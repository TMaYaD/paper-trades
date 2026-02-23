import importlib
import pkgutil
from abc import ABC, abstractmethod


class Strategy(ABC):
    name: str  # e.g. "trade-half", used in --strategy CLI flag

    @abstractmethod
    def step(self, wallet, history_a, history_b, tick=None) -> float:
        """Return a trade signal (absolute units to sell):
          +n  → sell n units of B, buy A with proceeds
           0  → no trade
          -n  → sell n units of A, buy B with proceeds
        """


# --- Auto-discovery registry ---
REGISTRY = {}


def _discover():
    """Import all modules in this package and register Strategy subclasses."""
    for _, modname, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f".{modname}", __package__)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type)
                    and issubclass(attr, Strategy)
                    and attr is not Strategy
                    and hasattr(attr, 'name')):
                REGISTRY[attr.name] = attr


_discover()
