import random
from typing import List

import numpy as np

from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL


class ZIAgent(Agent):
    def __init__(
        self,
        agent_id: int,
        market: Market,
        q_max: int,
        shade: List,
        pv_var: float,
        eta: float = 1.0,
        latency: int = 0,
    ):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.shade = shade
        self.cash = 0
        self.eta = eta
        self.latency = max(0, int(latency))
        self._order_counter = 0  # Counter for unique order IDs

    def get_id(self) -> int:
        return self.agent_id

    def get_observation(self):
        """
        Delayed market observation according to fixed absolute latency.
        """
        return self.market.get_observation(latency_steps=self.latency)

    def estimate_fundamental(self):
        """
        For now, the agent still uses the market/fundamental process info directly.
        The order-book observation is delayed; the fundamental estimate remains aligned
        with the current simulator state unless you later choose to delay that too.
        """
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1 - r) ** (T - t)
        estimate = (1 - rho) * mean + rho * val
        return estimate

    def take_action(self, estimate=None):
        side = random.choice([BUY, SELL])
        t = self.market.get_time()

        obs = self.get_observation()
        observed_book = obs["book"]

        if estimate is None:
            estimate = self.estimate_fundamental()

        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread * random.random() + self.shade[0]

        # Cache private value lookup
        pv_value = self.pv.value_for_exchange(self.position, side)

        if side == BUY:
            price = estimate + pv_value - valuation_offset
        else:
            price = estimate + pv_value + valuation_offset

        if self.eta != 1.0:
            base_price = estimate + pv_value

            if side == BUY:
                best_price = observed_book.get("best_ask", None)
                if best_price is not None and (base_price - best_price) > self.eta * valuation_offset:
                    price = best_price
            else:
                best_price = observed_book.get("best_bid", None)
                if best_price is not None and (best_price - base_price) > self.eta * valuation_offset:
                    price = best_price

        # Use counter for order ID
        self._order_counter += 1
        order_id = self.agent_id * 1000000 + self._order_counter

        order = Order(
            price=price,
            quantity=1,
            agent_id=self.agent_id,
            time=t,
            order_type=side,
            order_id=order_id
        )

        # attach latency / observation metadata for later analysis
        order.latency = self.latency
        order.true_time = obs["true_time"]
        order.true_moment = obs["true_moment"]
        order.observed_time = obs["observed_time"]
        order.observed_moment = obs["observed_moment"]
        order.book_staleness = obs["true_time"] - obs["observed_time"]

        return [order]

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'ZI{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self.position = 0
        self.cash = 0
        self.pv = PrivateValues(self.q_max, self.pv_var)
        self._order_counter = 0