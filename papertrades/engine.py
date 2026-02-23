import json
import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .client import CachedClient, GeckoClient
from .dex import SimulatedDex
from .price_history import PriceHistory
from .wallet import SimulatedWallet


@dataclass
class BacktestResult:
    value_history: list[float]
    trade_count: int
    dates: list


def _portfolio_value(wallet: SimulatedWallet, dex: SimulatedDex,
                     token_a: str, token_b: str) -> float:
    """Portfolio value denominated in token A."""
    return wallet.balance(token_a) + wallet.balance(token_b) * dex.prices[token_b] / dex.prices[token_a]


class BacktestEngine:
    def __init__(self, token_a: str, token_b: str, swap_fee: float = 0.005,
                 network: str = "solana", client: CachedClient | None = None):
        self._client = client or CachedClient(GeckoClient())
        self.token_a = token_a
        self.token_b = token_b
        self.swap_fee = swap_fee
        self.network = network

    def run(self, strategy, start_date,
            interval: str = "hourly") -> BacktestResult:
        """Run a strategy over historical bars."""
        token_a, token_b = self.token_a, self.token_b

        history_a = PriceHistory(token_a, self.network, self._client, start_date)
        history_b = PriceHistory(token_b, self.network, self._client, start_date)

        # Align series on common timestamps
        common_idx = history_a.all_prices().index.intersection(
            history_b.all_prices().index)
        if interval == "daily":
            common_idx = pd.Series(index=common_idx).resample("D").first().index

        p_a0 = history_a.price_at(common_idx[0])
        p_b0 = history_b.price_at(common_idx[0])
        dex = SimulatedDex({token_a: p_a0, token_b: p_b0}, fee=self.swap_fee)
        wallet = SimulatedWallet.balanced(token_a, token_b, dex)

        value_history = [_portfolio_value(wallet, dex, token_a, token_b)]
        trades = 0
        dates = list(common_idx)

        for i in range(1, len(common_idx)):
            tick = common_idx[i]
            history_a.set_cursor(
                history_a.all_prices().index.get_indexer([tick], method="ffill")[0])
            history_b.set_cursor(
                history_b.all_prices().index.get_indexer([tick], method="ffill")[0])
            p_a = history_a.current_price
            p_b = history_b.current_price

            dex.prices[token_a] = p_a
            dex.prices[token_b] = p_b

            signal = strategy.step(wallet, history_a, history_b, tick=tick)
            if signal != 0:
                if signal > 0:
                    wallet.swap(token_b, signal, token_a, dex)
                else:
                    wallet.swap(token_a, abs(signal), token_b, dex)
                trades += 1

            value_history.append(_portfolio_value(wallet, dex, token_a, token_b))

        return BacktestResult(
            value_history=value_history,
            trade_count=trades,
            dates=dates,
        )


class LiveEngine:
    def __init__(self, token_a: str, token_b: str, strategy,
                 swap_fee: float = 0.005, interval: str = "hourly",
                 network: str = "solana", client: CachedClient | None = None,
                 trades_file: str = "trades.jsonl"):
        self._client = client or CachedClient(GeckoClient())
        self.token_a = token_a
        self.token_b = token_b
        self.strategy = strategy
        self.swap_fee = swap_fee
        self.interval = interval
        self.network = network
        self.trades_file = trades_file

    def run(self):
        token_a, token_b = self.token_a, self.token_b
        interval_seconds = 3600 if self.interval == "hourly" else 86400

        print(f"--- Live Paper Trading ---")
        print(f"Strategy:  {self.strategy.name}")
        print(f"Token A:   {token_a[:12]}...")
        print(f"Token B:   {token_b[:12]}...")
        print(f"Interval:  {self.interval} ({interval_seconds}s)")
        print(f"Swap fee:  {self.swap_fee}")
        print(f"Logging to: {self.trades_file}")
        print("Press Ctrl+C to stop.\n")

        print("Fetching initial prices...")
        hist_a = PriceHistory(token_a, self.network, self._client)
        hist_b = PriceHistory(token_b, self.network, self._client)
        p_a = hist_a.poll()
        p_b = hist_b.poll()
        print(f"  Price A: ${p_a:.6f}  |  Price B: ${p_b:.6f}")

        dex = SimulatedDex({token_a: p_a, token_b: p_b}, fee=self.swap_fee)
        wallet = SimulatedWallet.balanced(token_a, token_b, dex)
        pv = _portfolio_value(wallet, dex, token_a, token_b)
        print(f"  Initial portfolio value: {pv:.6f} (in Token A units)\n")

        tick = 0
        try:
            while True:
                time.sleep(interval_seconds)
                tick += 1
                now = datetime.utcnow()

                try:
                    p_a = hist_a.poll()
                    p_b = hist_b.poll()
                except Exception as e:
                    print(f"[{now.isoformat()}] Error fetching prices: {e}. Retrying next interval.")
                    continue

                dex.prices[token_a] = p_a
                dex.prices[token_b] = p_b

                signal = self.strategy.step(wallet, hist_a, hist_b, tick=now)
                traded = signal != 0
                if traded:
                    if signal > 0:
                        wallet.swap(token_b, signal, token_a, dex)
                    else:
                        wallet.swap(token_a, abs(signal), token_b, dex)

                pv = _portfolio_value(wallet, dex, token_a, token_b)

                log_entry = {
                    "timestamp": now.isoformat(),
                    "tick": tick,
                    "price_a": p_a,
                    "price_b": p_b,
                    "bal_a": wallet.balance(token_a),
                    "bal_b": wallet.balance(token_b),
                    "portfolio_value": pv,
                    "traded": traded,
                    "signal": signal,
                }

                action = "TRADE" if traded else "HOLD"
                print(
                    f"[{now.isoformat()}] #{tick} {action} | "
                    f"A=${p_a:.6f} B=${p_b:.6f} | "
                    f"bal_a={wallet.balance(token_a):.6f} "
                    f"bal_b={wallet.balance(token_b):.6f} | "
                    f"value={pv:.6f}"
                )

                with open(self.trades_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

        except KeyboardInterrupt:
            print(f"\nStopped after {tick} ticks. Final value: {pv:.6f}")
