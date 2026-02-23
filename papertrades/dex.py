from abc import ABC, abstractmethod


class Dex(ABC):
    """Interface for decentralized exchanges. Provides pricing and executes swaps."""

    @abstractmethod
    def swap(self, sell_token: str, sell_amount: float, buy_token: str) -> float:
        """Execute swap, return amount of buy_token received."""


class SimulatedDex(Dex):
    """Paper trading DEX. Swaps use current prices with a flat fee."""

    def __init__(self, prices: dict[str, float], fee: float = 0.005):
        self.prices = prices  # mutable — engine updates each tick
        self.fee = fee

    def swap(self, sell_token: str, sell_amount: float, buy_token: str) -> float:
        value = sell_amount * self.prices[sell_token]
        return (value / self.prices[buy_token]) * (1 - self.fee)
