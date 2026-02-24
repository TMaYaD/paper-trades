import json
import time
from dataclasses import dataclass
from datetime import timedelta

from .client import CachedClient, GeckoClient
from .dex import SimulatedDex
from .price_history import PriceHistory
from .timestamps import ceil_hour, parse_ts, utcnow
from .wallet import SimulatedWallet


@dataclass
class BacktestResult:
    value_history: list[float]
    wallet: SimulatedWallet
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

    def run(self, strategy, start_date, end_date=None) -> BacktestResult:
        """Run a strategy over historical bars."""
        token_a, token_b = self.token_a, self.token_b

        history_a = PriceHistory(token_a, self.network, self._client)
        history_b = PriceHistory(token_b, self.network, self._client)

        # Clamp start to the later pool_created_at so we don't request
        # data from before a pool existed.
        effective_start = parse_ts(start_date)
        for h in (history_a, history_b):
            if h.pool_created_at is not None and h.pool_created_at > effective_start:
                effective_start = h.pool_created_at

        # Walk hourly from effective_start to end_date (default: now)
        tick = ceil_hour(effective_start)
        end = parse_ts(end_date) if end_date else utcnow()

        value_history = []
        dates = []
        dex = wallet = None

        while tick <= end:
            try:
                p_a = history_a.price_at(tick)
                p_b = history_b.price_at(tick)
            except ValueError:
                tick += timedelta(hours=1)
                continue

            if dex is None:
                dex = SimulatedDex({token_a: p_a, token_b: p_b}, fee=self.swap_fee)
                wallet = SimulatedWallet.balanced(token_a, token_b, dex)
            else:
                dex.prices[token_a] = p_a
                dex.prices[token_b] = p_b

                wallet.set_time(tick)
                signal = strategy.step(wallet, history_a, history_b, tick=tick)
                if signal != 0:
                    if signal > 0:
                        wallet.swap(token_b, signal, token_a, dex)
                    else:
                        wallet.swap(token_a, abs(signal), token_b, dex)

            value_history.append(_portfolio_value(wallet, dex, token_a, token_b))
            dates.append(tick)
            tick += timedelta(hours=1)

        return BacktestResult(
            value_history=value_history,
            wallet=wallet,
            dates=dates,
        )


class LiveEngine:
    def __init__(self, token_a: str, token_b: str, strategy,
                 swap_fee: float = 0.005,
                 network: str = "solana", client: CachedClient | None = None,
                 trades_file: str = "trades.jsonl"):
        self._client = client or CachedClient(GeckoClient())
        self.token_a = token_a
        self.token_b = token_b
        self.strategy = strategy
        self.swap_fee = swap_fee
        self.network = network
        self.trades_file = trades_file

    def run(self):
        token_a, token_b = self.token_a, self.token_b
        interval_seconds = 3600

        print(f"--- Live Paper Trading ---")
        print(f"Strategy:  {self.strategy.name}")
        print(f"Token A:   {token_a[:12]}...")
        print(f"Token B:   {token_b[:12]}...")
        print(f"Interval:  hourly ({interval_seconds}s)")
        print(f"Swap fee:  {self.swap_fee}")
        print(f"Logging to: {self.trades_file}")
        print("Press Ctrl+C to stop.\n")

        print("Fetching initial prices...")
        hist_a = PriceHistory(token_a, self.network, self._client)
        hist_b = PriceHistory(token_b, self.network, self._client)
        p_a = hist_a.current_price
        p_b = hist_b.current_price
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
                now = utcnow()

                try:
                    p_a = hist_a.current_price
                    p_b = hist_b.current_price
                except Exception as e:
                    print(f"[{now.isoformat()}] Error fetching prices: {e}. Retrying next interval.")
                    continue

                dex.prices[token_a] = p_a
                dex.prices[token_b] = p_b

                wallet.set_time(now)
                prev_trades = len(wallet.activity)
                signal = self.strategy.step(wallet, hist_a, hist_b, tick=now)
                if signal != 0:
                    if signal > 0:
                        wallet.swap(token_b, signal, token_a, dex)
                    else:
                        wallet.swap(token_a, abs(signal), token_b, dex)

                traded = len(wallet.activity) > prev_trades
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
