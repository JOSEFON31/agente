"""
Entrenamiento offline para warm-start del agente online (nuclear2.py).

Uso:
  python entrenar_offline.py

Variables opcionales:
  OFFLINE_DATA_FILE=btc_usdt_1m_con_indicadores.parquet
  OFFLINE_EPISODES=2000
"""

import os
import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


# ───────────────── CONFIG ─────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

STATE_DIM = 14
ACTION_DIM = 3

GAMMA = 0.95
LR = 1e-4
BATCH_SIZE = 128
BUFFER_SIZE = 150_000
MIN_REPLAY_FOR_TRAIN = 1_000
TARGET_UPDATE_TAU = 0.01
GRAD_CLIP = 1.2

EPS_START = 0.98
EPS_END = 0.01
EPS_DECAY = 0.9992

NUM_EPISODES = int(os.getenv("OFFLINE_EPISODES", "2000"))
TRAIN_EVERY_N_STEPS = 3
DATA_FILE = os.getenv("OFFLINE_DATA_FILE", "btc_usdt_1m_con_indicadores.parquet")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[offline] device={DEVICE}")


# ───────────────── MODEL (igual arquitectura online) ─────────────────
class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)


model = DuelingDQN(STATE_DIM, ACTION_DIM).to(DEVICE)
target_model = DuelingDQN(STATE_DIM, ACTION_DIM).to(DEVICE)
target_model.load_state_dict(model.state_dict())
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(args)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


memory = ReplayBuffer(BUFFER_SIZE)


def normalize_state(features: list) -> np.ndarray:
    return np.clip(np.array(features, dtype=np.float32), -5.0, 5.0)


def select_action(state: np.ndarray, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(ACTION_DIM)
    with torch.no_grad():
        q = model(torch.from_numpy(state).unsqueeze(0).to(DEVICE))
    return int(q.argmax(1).item())


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float = TARGET_UPDATE_TAU) -> None:
    for t, s in zip(target_net.parameters(), source_net.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "close" not in out.columns:
        raise ValueError("El dataset offline debe incluir columna 'close'.")

    # fallback indicators if not precomputed
    if "ret1" not in out.columns:
        out["ret1"] = out["close"].pct_change(1)
    if "ret3" not in out.columns:
        out["ret3"] = out["close"].pct_change(3)
    if "ret6" not in out.columns:
        out["ret6"] = out["close"].pct_change(6)

    # map common alternative names from your script example
    if "ret4" in out.columns and "ret3" not in df.columns:
        out["ret3"] = out["ret4"]
    if "ret12" in out.columns and "ret6" not in df.columns:
        out["ret6"] = out["ret12"]

    for col, default in {
        "rsi": 50.0,
        "macd": 0.0,
        "ema_fast": out["close"],
        "ema_slow": out["close"],
        "atr_pct": 0.0,
        "vol": 0.0,
        "volume_z": 0.0,
    }.items():
        if col not in out.columns:
            out[col] = default

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


class TradingEnvOffline:
    def __init__(self, df: pd.DataFrame):
        self.df = ensure_features(df).reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.current_step = 0
        self.capital = 100.0
        self.peak_capital = 100.0
        self.in_position = False
        self.entry = None

    def reset(self) -> np.ndarray:
        min_start = 120
        max_start = max(min_start, self.max_steps - 2500)
        self.current_step = random.randint(min_start, max_start)
        self.capital = 100.0
        self.peak_capital = 100.0
        self.in_position = False
        self.entry = None
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        row = self.df.iloc[min(self.current_step, self.max_steps)]

        trend = (float(row["ema_fast"]) - float(row["ema_slow"])) / max(float(row["close"]), 1e-9)
        age = 0.0
        unrealized = 0.0
        if self.in_position and self.entry:
            age = self.current_step - self.entry["step_open"]
            unrealized = (float(row["close"]) - self.entry["price_open"]) / max(self.entry["price_open"], 1e-9)

        drawdown = 1 - (self.capital / max(self.peak_capital, 1e-9))

        return normalize_state([
            float(row["rsi"]) / 100.0,
            float(row["macd"]),
            trend * 50.0,
            float(row["ret1"]),
            float(row["ret3"]),
            float(row["ret6"]),
            float(row["atr_pct"]) * 20.0,
            float(row["vol"]) * 10.0,
            float(row["volume_z"]) / 4.0,
            1.0 if self.in_position else 0.0,
            np.log1p(age),
            drawdown * 10.0,
            np.clip(unrealized * 10.0, -5.0, 5.0),
            np.clip((self.capital - 100.0) / 100.0 * 10.0, -5.0, 5.0),
        ])

    def step(self, action: int):
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True

        row = self.df.iloc[self.current_step]
        price = float(row["close"])
        reward = 0.0
        close_now = False

        if not self.in_position and action == 1:
            size = self.capital * 0.8
            qty = size / max(price, 1e-9)
            self.in_position = True
            self.entry = {
                "price_open": price,
                "qty": qty,
                "cost": size,
                "step_open": self.current_step,
                "min_price": price,
            }

        elif self.in_position and self.entry:
            self.entry["min_price"] = min(self.entry["min_price"], price)
            pnl_pct = (price - self.entry["price_open"]) / max(self.entry["price_open"], 1e-9)
            hold_steps = self.current_step - self.entry["step_open"]

            if action == 2 or pnl_pct <= -0.012 or hold_steps >= 25 or pnl_pct >= 0.009:
                close_now = True

        if close_now and self.in_position and self.entry:
            usdt_out = self.entry["qty"] * price
            pnl = usdt_out - self.entry["cost"]
            ret_pct = pnl / max(self.entry["cost"], 1e-9) * 100.0
            hold_steps = self.current_step - self.entry["step_open"]
            max_dd = (self.entry["min_price"] - self.entry["price_open"]) / max(self.entry["price_open"], 1e-9)

            # reward logic inspired by your provided script
            reward = ret_pct
            if ret_pct > 0:
                reward += (ret_pct ** 1.1) * 1.8
                if ret_pct > 0.5:
                    reward += (ret_pct - 0.5) * 7.5
                if ret_pct > 1.0:
                    reward += (ret_pct - 1.0) * 13.0
            if hold_steps > 600:
                reward -= (hold_steps - 600) * 0.002
            if max_dd < -0.6:
                reward -= abs(max_dd) * 2.0
            reward = float(np.clip(reward, -20.0, 60.0))

            self.capital += pnl
            self.peak_capital = max(self.peak_capital, self.capital)
            if self.capital > self.peak_capital:
                reward += (self.capital - self.peak_capital) * 20.0

            self.in_position = False
            self.entry = None

        self.current_step += 1
        done = self.current_step >= self.max_steps or self.capital < 10.0
        return self._get_state(), reward, done


def main() -> None:
    df = pd.read_parquet(DATA_FILE)
    env = TradingEnvOffline(df)

    epsilon = EPS_START
    global_step = 0

    ep_rewards = []
    ep_capitals = []

    print(f"[offline] episodes={NUM_EPISODES} | data={DATA_FILE}")

    for ep in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        max_steps_ep = random.randint(800, 2000)

        while not done and ep_len < max_steps_ep:
            action = select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, float(done))

            state = next_state
            ep_reward += reward
            ep_len += 1
            global_step += 1

            if len(memory) >= MIN_REPLAY_FOR_TRAIN and ep_len % TRAIN_EVERY_N_STEPS == 0:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                s = torch.from_numpy(states).to(DEVICE)
                ns = torch.from_numpy(next_states).to(DEVICE)
                a = torch.from_numpy(actions).long().to(DEVICE)
                r = torch.from_numpy(rewards).to(DEVICE)
                d = torch.from_numpy(dones).to(DEVICE)

                q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                next_actions = model(ns).detach().argmax(1)
                next_q = target_model(ns).detach().gather(1, next_actions.unsqueeze(1)).squeeze(1)
                expected = r + GAMMA * next_q * (1.0 - d)

                loss = nn.SmoothL1Loss()(q, expected)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                soft_update(target_model, model)

        epsilon = max(EPS_END, EPS_START * (EPS_DECAY ** ep))
        ep_rewards.append(ep_reward)
        ep_capitals.append(env.capital)

        avg_r = np.mean(ep_rewards[-50:])
        avg_c = np.mean(ep_capitals[-50:])
        print(
            f"Ep {ep:4d} | Reward {ep_reward:8.2f} | Cap {env.capital:7.2f} | Len {ep_len:4d} "
            f"| eps {epsilon:.4f} | avgR50 {avg_r:7.2f} | avgC50 {avg_c:7.2f}"
        )

        if ep % 50 == 0:
            ckpt = f"models/dqn_offline_ep{ep}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"[offline] saved {ckpt} | buffer={len(memory):,} | steps={global_step:,}")

    torch.save(model.state_dict(), "models/dqn_offline_final.pt")
    print("[offline] entrenamiento finalizado -> models/dqn_offline_final.pt")


if __name__ == "__main__":
    main()
