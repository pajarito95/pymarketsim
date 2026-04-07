from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RLParticipationAgent:
    agent_id: int
    tickers: List[str]
    initial_cash: float = 100_000.0

    cash: float = field(init=False)
    inventory: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.cash = float(self.initial_cash)
        self.inventory = {ticker: 0 for ticker in self.tickers}

    def update_position(self, ticker: str, quantity: int, cash_delta: float):
        self.inventory[ticker] += int(quantity)
        self.cash += float(cash_delta)

    def get_inventory(self, ticker: str) -> int:
        return int(self.inventory[ticker])

    def net_worth(self, mark_prices: Dict[str, float]) -> float:
        nw = self.cash
        for ticker, qty in self.inventory.items():
            nw += qty * float(mark_prices[ticker])
        return float(nw)