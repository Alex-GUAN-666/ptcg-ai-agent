"""Public callback and per-game controller; original selection behavior retained."""

from .controller import DecisionController
from .weights import ASSETS, load_deck


class Agent(DecisionController):
    """Owns one game's history. Not intended to be shared by concurrent games."""

    def __init__(self):
        super().__init__(ASSETS, 0, load_deck())


_default_agent = None


def agent(observation: dict, configuration=None) -> list[int]:
    """Competition-shaped callback; the first call lazily loads the weights."""
    global _default_agent
    if _default_agent is None:
        _default_agent = Agent()
    return _default_agent(observation, configuration)
