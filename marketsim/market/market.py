from copy import deepcopy

from marketsim.event.event_queue import EventQueue
from marketsim.fourheap.fourheap import FourHeap
from marketsim.fundamental.fundamental_abc import Fundamental
from marketsim.fourheap import constants


class Market:
    """
    Market with lightweight order-book snapshot history for fixed-latency observations.
    """

    def __init__(self, fundamental: Fundamental, time_steps):
        self.order_book = FourHeap()
        self.matched_orders = []
        self.fundamental = fundamental
        self.event_queue = EventQueue()
        self.end_time = time_steps

        # latency / observation-history state
        self.book_history = {}
        self.book_moment = 0

        # save initial snapshot at time 0
        self._record_book_snapshot()

    def get_fundamental_value(self):
        t = self.get_time()
        return self.fundamental.get_value_at(t)

    def get_final_fundamental(self):
        return self.fundamental.get_final_fundamental()

    def withdraw_all(self, agent_id: int):
        self.order_book.withdraw_all(agent_id)

    def clear_market(self):
        new_orders = self.order_book.market_clear(self.get_time())
        self.matched_orders += new_orders
        return new_orders

    def add_orders(self, orders):
        for order in orders:
            self.event_queue.schedule_activity(order)

    def get_time(self):
        return self.event_queue.get_current_time()

    def get_info(self):
        return self.fundamental.get_info()

    def get_current_book_moment(self) -> int:
        return self.book_moment

    def _build_snapshot(self):
        """
        Lightweight snapshot of observable market state.
        """
        book_state = self.order_book.snapshot_top_of_book()
        return {
            "time": int(self.get_time()),
            "moment": int(self.book_moment),
            **book_state,
        }

    def _record_book_snapshot(self):
        """
        Record the current visible order-book state at the current simulation time.
        If multiple updates occur at the same time, the newest one overwrites the
        prior snapshot for that time.
        """
        t = int(self.get_time())
        self.book_history[t] = self._build_snapshot()

    def get_snapshot_at_time(self, query_time: int):
        """
        Return the most recent snapshot available at or before query_time.
        """
        if not self.book_history:
            return None

        query_time = max(0, int(query_time))
        valid_times = [t for t in self.book_history.keys() if t <= query_time]
        if not valid_times:
            earliest_t = min(self.book_history.keys())
            return deepcopy(self.book_history[earliest_t])

        chosen_t = max(valid_times)
        return deepcopy(self.book_history[chosen_t])

    def get_observation(self, latency_steps: int = 0):
        """
        Return delayed market observation based on fixed absolute latency.
        If latency_steps = 0, the agent sees the current visible book.
        """
        true_time = int(self.get_time())
        latency_steps = max(0, int(latency_steps))
        observed_time = max(0, true_time - latency_steps)

        snapshot = self.get_snapshot_at_time(observed_time)
        if snapshot is None:
            snapshot = self._build_snapshot()

        return {
            "true_time": true_time,
            "true_moment": int(self.book_moment),
            "observed_time": int(snapshot["time"]),
            "observed_moment": int(snapshot["moment"]),
            "latency_steps": latency_steps,
            "book": snapshot,
        }

    def step(self):
        # TODO Need to figure out how to handle ties for price and time
        orders = self.event_queue.step()
        self.buy_init_volume, self.sell_init_volume = 0, 0

        for order in orders:
            if order.quantity <= 0:
                continue
            self.order_book.insert(order)

        new_orders = self.clear_market()

        # Compute midprices.
        self.order_book.update_midprice()

        # Increment book moment and save snapshot after each visible market update.
        self.book_moment += 1
        self._record_book_snapshot()

        return new_orders

    def get_midprices(self):
        return self.order_book.midprices

    def reset(self, fundamental):
        self.order_book = FourHeap()
        self.matched_orders = []
        self.event_queue = EventQueue()
        self.fundamental = fundamental

        # reset latency / observation-history state
        self.book_history = {}
        self.book_moment = 0
        self._record_book_snapshot()