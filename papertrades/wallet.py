from abc import ABC, abstractmethod

from .dex import Dex


class Wallet(ABC):
    """Interface for wallets. Read balances, execute swaps."""

    @abstractmethod
    def balance(self, token: str) -> float: ...

    @property
    @abstractmethod
    def balances(self) -> dict[str, float]: ...

    @abstractmethod
    def swap(self, sell_token: str, sell_amount: float, buy_token: str, dex: Dex) -> float:
        """Sell sell_amount of sell_token, buy buy_token via dex. Returns amount received."""


class SimulatedWallet(Wallet):
    """Paper trading wallet. Delegates pricing to the dex."""

    def __init__(self, balances: dict[str, float]):
        self._balances = dict(balances)

    def balance(self, token: str) -> float:
        return self._balances.get(token, 0.0)

    @property
    def balances(self) -> dict[str, float]:
        return dict(self._balances)

    def swap(self, sell_token: str, sell_amount: float, buy_token: str, dex: Dex) -> float:
        received = dex.swap(sell_token, sell_amount, buy_token)
        self._balances[sell_token] -= sell_amount
        self._balances[buy_token] = self._balances.get(buy_token, 0.0) + received
        return received

    @classmethod
    def balanced(cls, token_a: str, token_b: str, dex: Dex) -> "SimulatedWallet":
        """Factory: 50-50 split by value. 1 unit of token_a, equivalent value of token_b."""
        return cls({token_a: 1.0, token_b: dex.prices[token_a] / dex.prices[token_b]})
