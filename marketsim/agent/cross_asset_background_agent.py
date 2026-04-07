import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues


@dataclass
class CrossAssetBackgroundAgent:
    agent_id: int
    tickers: List[str]
    q_max: int
    pv_var: float
    shade: List[float]
    eta: float = 1.0
    initial_cash: float = 100_000.0
    latency: int = 0

    cash: float = field(init=False)
    inventory: Dict[str, int] = field(init=False)
    pv: Dict[str, PrivateValues] = field(init=False)
    _order_counter: int = field(init=False, default=0)

    def __post_init__(self):
        self.latency = max(0, int(self.latency))
        self.reset()

    def reset(self):
        self.cash = float(self.initial_cash)
        self.inventory = {ticker: 0 for ticker in self.tickers}
        self.pv = {
            ticker: PrivateValues(self.q_max, self.pv_var)
            for ticker in self.tickers
        }
        self._order_counter = 0

    def update_position(self, ticker: str, q: int, cash_delta: float):
        self.inventory[ticker] += int(q)
        self.cash += float(cash_delta)

    def get_inventory(self, ticker: str) -> int:
        return int(self.inventory[ticker])

    def get_observation(self, market):
        return market.get_observation(latency_steps=self.latency)

    def estimate_fundamental(self, market) -> float:
        mean, r, T = market.get_info()
        t = market.get_time()
        val = market.get_fundamental_value()
        rho = (1 - r) ** (T - t)
        return float((1 - rho) * mean + rho * val)

    def _score_action_for_asset(self, ticker: str, market) -> Tuple[Optional[int], float, float, dict]:
        """
        Returns:
            best_side, best_surplus, candidate_price, delayed_obs
        """
        obs = self.get_observation(market)
        book = obs["book"]

        estimate = self.estimate_fundamental(market)
        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread * random.random() + self.shade[0]

        pos = self.inventory[ticker]
        pv_buy = self.pv[ticker].value_for_exchange(pos, BUY)
        pv_sell = self.pv[ticker].value_for_exchange(pos, SELL)

        best_ask = book.get("best_ask", None)
        best_bid = book.get("best_bid", None)

        buy_surplus = -np.inf
        sell_surplus = -np.inf

        buy_price = estimate + pv_buy - valuation_offset
        sell_price = estimate + pv_sell + valuation_offset

        if best_ask is not None:
            buy_surplus = (estimate + pv_buy) - best_ask
            if buy_surplus > self.eta * valuation_offset:
                buy_price = best_ask
            else:
                buy_surplus = (estimate + pv_buy) - buy_price
        else:
            buy_surplus = (estimate + pv_buy) - buy_price

        if best_bid is not None:
            sell_surplus = best_bid - (estimate + pv_sell)
            if sell_surplus > self.eta * valuation_offset:
                sell_price = best_bid
            else:
                sell_surplus = sell_price - (estimate + pv_sell)
        else:
            sell_surplus = sell_price - (estimate + pv_sell)

        if self.cash < max(buy_price, 0.0):
            buy_surplus = -np.inf

        if self.inventory[ticker] <= 0:
            sell_surplus = -np.inf

        if buy_surplus <= 0 and sell_surplus <= 0:
            return None, 0.0, 0.0, obs

        if buy_surplus >= sell_surplus:
            return BUY, float(buy_surplus), float(buy_price), obs
        else:
            return SELL, float(sell_surplus), float(sell_price), obs

    def choose_action(self, markets: Dict[str, object]) -> Optional[Tuple[str, Order]]:
        """
        Choose at most one order across all assets.
        """
        best_choice = None
        best_surplus = -np.inf

        for ticker, market in markets.items():
            side, surplus, price, obs = self._score_action_for_asset(ticker, market)
            if side is None:
                continue

            if surplus > best_surplus:
                self._order_counter += 1
                order_id = self.agent_id * 1_000_000 + self._order_counter
                order = Order(
                    price=float(price),
                    quantity=1.0,
                    agent_id=self.agent_id,
                    time=market.get_time(),
                    order_type=side,
                    order_id=order_id,
                    asset_id=0,
                )

                # attach latency / stale-book metadata
                order.latency = self.latency
                order.true_time = obs["true_time"]
                order.true_moment = obs["true_moment"]
                order.observed_time = obs["observed_time"]
                order.observed_moment = obs["observed_moment"]
                order.book_staleness = obs["true_time"] - obs["observed_time"]

                best_choice = (ticker, order)
                best_surplus = surplus

        return best_choice

    def get_pos_value(self, mark_prices: Dict[str, float]) -> float:
        total = 0.0
        for ticker in self.tickers:
            total += self.pv[ticker].value_at_position(self.inventory[ticker])
            total += self.inventory[ticker] * mark_prices[ticker]
        return float(total)

    def net_worth(self, mark_prices: Dict[str, float]) -> float:
        return float(self.cash + self.get_pos_value(mark_prices))