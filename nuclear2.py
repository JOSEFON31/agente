"""
Nuclear2 - Trading RL agent (improved baseline)

⚠️ Important:
- No strategy can guarantee profits.
- This script is for research/education and should be validated with extensive backtesting/paper trading.
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Deque, Optional, Tuple

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL
from binance.exceptions import BinanceAPIException
from torch.nn.utils import clip_grad_norm_


@dataclass
class Config:
    # RL
    state_dim: int = 14
    action_dim: int = 3  # 0 hold, 1 buy, 2 sell
    gamma: float = 0.98
    lr: float = 1.5e-4
    batch_size: int = 128
    buffer_size: int = 200_000
    min_replay_for_train: int = 2_000
    eps_start: float = 0.20
    eps_end: float = 0.02
    eps_decay: float = 0.999
    target_update_tau: float = 0.01

    # Trading
    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    interval: str = Client.KLINE_INTERVAL_1MINUTE
    history_bars: int = 300
    taker_fee: float = 0.001
    take_profit: float = 0.008
    stop_loss: float = -0.012
    max_hold_steps: int = 25
    min_order_usdt: float = 10.0

    # Risk
    risk_per_trade: float = 0.01       # 1% of equity at stop-loss distance
    max_position_notional: float = 0.35  # max 35% of equity
    max_daily_drawdown: float = 0.05
    cooldown_steps_after_loss: int = 3

    # Rewards
    hold_penalty: float = -0.00015
    inaction_penalty: float = -0.00005
    long_hold_penalty: float = -0.001
    reward_scale: float = 4.0

    # Runtime
    target_equity: float = float(os.getenv("TARGET_EQUITY", "1000000"))
    model_dir: str = "models"
    sleep_seconds: float = 0.8


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG.model_dir, exist_ok=True)


def getenv_required(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required env var: {name}")
    value = value.strip()
    if not value:
        raise RuntimeError(f"Env var is empty: {name}")
    return value


def load_binance_credentials() -> tuple[str, str]:
    """Load and validate Binance credentials from environment variables."""
    api_key = getenv_required("BINANCE_API_KEY")
    api_secret = getenv_required("BINANCE_API_SECRET")
    return api_key, api_secret


def sync_binance_time(client: Client) -> None:
    try:
        server_time = client.get_server_time()["serverTime"]
        local_time = int(time.time() * 1000)
        client.timestamp_offset = server_time - local_time
        print(f"🕒 Binance clock sync offset = {client.timestamp_offset}ms")
    except Exception as exc:
        print(f"⚠ Clock sync failed: {exc}")


def safe_api_call(client: Client, func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except BinanceAPIException as exc:
            if exc.code == -1021:
                print("⚠ Timestamp drift detected; re-syncing.")
                sync_binance_time(client)
                continue
            print(f"⚠ Binance API error: {exc}")
            time.sleep(1.0)
        except Exception as exc:
            print(f"⚠ Unexpected API error: {exc}")
            time.sleep(1.0)


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


class ReplayBuffer:
    def __init__(self, cap: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=cap)

    def push(self, *args) -> None:
        self.buffer.append(args)

    def sample(self, batch: int):
        samples = random.sample(self.buffer, batch)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def normalize_state(features) -> np.ndarray:
    return np.clip(np.array(features, dtype=np.float32), -5, 5)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = ta.momentum.RSIIndicator(out["close"], 14).rsi()
    out["macd"] = ta.trend.MACD(out["close"]).macd_diff()
    out["ema_fast"] = ta.trend.EMAIndicator(out["close"], 12).ema_indicator()
    out["ema_slow"] = ta.trend.EMAIndicator(out["close"], 26).ema_indicator()
    out["ret1"] = out["close"].pct_change(1)
    out["ret3"] = out["close"].pct_change(3)
    out["ret6"] = out["close"].pct_change(6)
    out["atr"] = ta.volatility.AverageTrueRange(out["high"], out["low"], out["close"], 14).average_true_range()
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["vol"] = out["close"].rolling(20).std() / out["close"].rolling(20).mean()
    out["volume_z"] = (out["volume"] - out["volume"].rolling(30).mean()) / out["volume"].rolling(30).std()
    return out.fillna(0)


class TradingEnv:
    def __init__(self, client: Client) -> None:
        self.client = client
        self.df = calculate_indicators(self._fetch(CFG.history_bars))
        self.step_idx = len(self.df) - 1
        self.in_position = False
        self.entry: Optional[dict] = None
        self.hold_steps = 0
        self.loss_cooldown = 0

        self.initial_equity = self._equity()
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.total_profit = 0.0
        self.daily_stop = False

    def _fetch(self, limit: int) -> pd.DataFrame:
        klines = safe_api_call(self.client, self.client.get_klines, symbol=CFG.symbol, interval=CFG.interval, limit=limit)
        df = pd.DataFrame(klines, columns=["t", "o", "h", "l", "c", "v", "ct", "q", "n", "tb", "tq", "i"])
        df[["o", "h", "l", "c", "v"]] = df[["o", "h", "l", "c", "v"]].astype(float)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return df

    def _equity(self) -> float:
        usdt = float(safe_api_call(self.client, self.client.get_asset_balance, asset="USDT")["free"])
        btc = float(safe_api_call(self.client, self.client.get_asset_balance, asset="BTC")["free"])
        price = float(safe_api_call(self.client, self.client.get_symbol_ticker, symbol=CFG.symbol)["price"])
        return usdt + btc * price

    def _risk_position_size_usdt(self, price: float, atr_pct: float) -> float:
        if atr_pct <= 0:
            return 0.0
        max_loss = self.equity * CFG.risk_per_trade
        stop_distance = max(abs(CFG.stop_loss), atr_pct * 1.5)
        raw_notional = max_loss / stop_distance
        capped_notional = min(raw_notional, self.equity * CFG.max_position_notional)
        return max(0.0, capped_notional)

    def update(self) -> None:
        new = self._fetch(1)
        self.df = pd.concat([self.df, new]).tail(CFG.history_bars).reset_index(drop=True)
        self.df = calculate_indicators(self.df)
        self.step_idx = len(self.df) - 1
        self.equity = self._equity()
        self.total_profit = self.equity - self.initial_equity
        self.peak_equity = max(self.peak_equity, self.equity)

        drawdown = 1 - (self.equity / self.peak_equity)
        if drawdown >= CFG.max_daily_drawdown:
            self.daily_stop = True

    def state(self) -> np.ndarray:
        r = self.df.iloc[self.step_idx]
        trend = (r.ema_fast - r.ema_slow) / max(r.close, 1e-9)
        age = self.hold_steps if self.in_position else 0
        drawdown = 1 - (self.equity / max(self.peak_equity, 1e-9))
        return normalize_state(
            [
                r.rsi / 100,
                r.macd,
                trend * 50,
                r.ret1,
                r.ret3,
                r.ret6,
                r.atr_pct * 20,
                r.vol * 10,
                r.volume_z / 4,
                1.0 if self.in_position else 0.0,
                np.log1p(age),
                drawdown * 10,
                self.loss_cooldown / max(CFG.cooldown_steps_after_loss, 1),
                (self.total_profit / max(self.initial_equity, 1e-9)) * 10,
            ]
        )

    def _sell_market(self, qty: float) -> None:
        qty_str = str(Decimal(str(qty)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
        safe_api_call(
            self.client,
            self.client.create_order,
            symbol=CFG.symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=qty_str,
        )

    def _buy_market(self, quote_usdt: float):
        quote_str = str(Decimal(str(quote_usdt)).quantize(Decimal("0.01"), rounding=ROUND_DOWN))
        return safe_api_call(
            self.client,
            self.client.create_order,
            symbol=CFG.symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quoteOrderQty=quote_str,
        )

    def step(self, action: int):
        self.update()
        row = self.df.iloc[self.step_idx]
        price = float(row.close)
        atr_pct = float(max(row.atr_pct, 1e-6))
        reward = 0.0

        if self.daily_stop:
            return self.state(), -1.0, True

        if not self.in_position:
            reward += CFG.inaction_penalty
            if self.loss_cooldown > 0:
                self.loss_cooldown -= 1

        if (not self.in_position) and action == 1 and self.loss_cooldown == 0:
            usdt = float(safe_api_call(self.client, self.client.get_asset_balance, asset="USDT")["free"])
            size_usdt = min(usdt, self._risk_position_size_usdt(price, atr_pct))
            if size_usdt >= CFG.min_order_usdt:
                order = self._buy_market(size_usdt)
                qty = float(order["executedQty"])
                cost = float(order["cummulativeQuoteQty"])
                fee_in = cost * CFG.taker_fee
                self.entry = {"price": price, "qty": qty, "cost": cost + fee_in}
                self.in_position = True
                self.hold_steps = 0
                print(f"🟢 BUY {qty:.6f} BTC @ ${price:,.2f} | Notional ${cost:,.2f}")

        if self.in_position and self.entry:
            self.hold_steps += 1
            reward += CFG.hold_penalty
            if self.hold_steps > 12:
                reward += CFG.long_hold_penalty

            pnl_gross = price * self.entry["qty"] - self.entry["cost"]
            fee_out = price * self.entry["qty"] * CFG.taker_fee
            pnl_net = pnl_gross - fee_out
            pnl_pct = (price - self.entry["price"]) / self.entry["price"]

            should_close = (
                pnl_pct >= CFG.take_profit
                or pnl_pct <= CFG.stop_loss
                or self.hold_steps >= CFG.max_hold_steps
                or action == 2
            )

            if should_close:
                self._sell_market(self.entry["qty"])
                print(f"🔴 SELL {self.entry['qty']:.6f} BTC @ ${price:,.2f} | PnL ${pnl_net:+,.2f} ({pnl_pct:+.2%})")

                # Stable reward shaping: positive pnl => positive reward, negative pnl => negative reward.
                reward += (pnl_net / max(self.initial_equity, 1e-9)) * CFG.reward_scale

                if pnl_net < 0:
                    self.loss_cooldown = CFG.cooldown_steps_after_loss

                self.in_position = False
                self.entry = None
                self.hold_steps = 0

        return self.state(), float(reward), False




def load_checkpoint_compatible(model: nn.Module, checkpoint_path: str) -> tuple[int, int]:
    """Load only checkpoint tensors with matching key+shape.

    Returns: (loaded_tensor_count, skipped_tensor_count).
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_state = model.state_dict()

    compatible = {}
    skipped = 0
    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped += 1

    model_state.update(compatible)
    model.load_state_dict(model_state)
    return len(compatible), skipped


def select_action(model: nn.Module, state: np.ndarray, eps: float) -> int:
    if random.random() < eps:
        return random.randrange(CFG.action_dim)
    with torch.no_grad():
        q = model(torch.from_numpy(state).unsqueeze(0).to(DEVICE))
    return int(q.argmax(1).item())


def main() -> None:
    api_key, api_secret = load_binance_credentials()

    client = Client(api_key, api_secret, testnet=True, requests_params={"timeout": 30})
    sync_binance_time(client)

    model = DuelingDQN(CFG.state_dim, CFG.action_dim).to(DEVICE)
    target_model = DuelingDQN(CFG.state_dim, CFG.action_dim).to(DEVICE)

    checkpoints = sorted([f for f in os.listdir(CFG.model_dir) if f.endswith(".pt")])
    if checkpoints:
        latest = os.path.join(CFG.model_dir, checkpoints[-1])
        loaded, skipped = load_checkpoint_compatible(model, latest)
        print(f"✅ Loaded checkpoint: {latest} | tensors loaded={loaded}, skipped={skipped}")

    target_model.load_state_dict(model.state_dict())
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
    memory = ReplayBuffer(CFG.buffer_size)

    env = TradingEnv(client)
    epsilon = CFG.eps_start
    state = env.state()
    steps = 0

    print(f"🚀 RL Trader running on {DEVICE} | target equity ${CFG.target_equity:,.2f}")

    while True:
        action = select_action(model, state, epsilon)
        next_state, reward, done = env.step(action)
        memory.push(state, action, reward, next_state, float(done))
        state = next_state
        steps += 1

        if len(memory) >= CFG.min_replay_for_train:
            states, actions, rewards, next_states, dones = memory.sample(CFG.batch_size)
            s = torch.from_numpy(states).to(DEVICE)
            ns = torch.from_numpy(next_states).to(DEVICE)
            a = torch.from_numpy(actions).long().to(DEVICE)
            r = torch.from_numpy(rewards).to(DEVICE)
            d = torch.from_numpy(dones).to(DEVICE)

            q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
            next_actions = model(ns).argmax(1)
            next_q = target_model(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = r + (1.0 - d) * CFG.gamma * next_q

            loss = nn.SmoothL1Loss()(q, target.detach())
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            soft_update(target_model, model, CFG.target_update_tau)

        epsilon = max(CFG.eps_end, epsilon * CFG.eps_decay)
        progress = (env.equity / max(CFG.target_equity, 1e-9)) * 100
        print(
            f"Step {steps} | Eq ${env.equity:,.2f} | PnL ${env.total_profit:+,.2f} "
            f"| DD {(1 - env.equity / max(env.peak_equity, 1e-9)):.2%} | {progress:.4f}% | eps {epsilon:.3f}"
        )

        if done:
            print("🛑 Trading halted due to max drawdown control.")
            break

        if env.equity >= CFG.target_equity:
            print(f"🏆 Target reached: ${env.equity:,.2f}")
            break

        if steps % 200 == 0:
            ckpt = os.path.join(CFG.model_dir, f"dqn_{steps}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved model: {ckpt}")

        time.sleep(CFG.sleep_seconds)


if __name__ == "__main__":
    main()
