import numpy as np
import pandas as pd
from marketsim.fundamental.fundamental_abc import Fundamental


class HistoricalFundamental(Fundamental):
    def __init__(self, prices, final_time: int = None):
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) == 0:
            raise ValueError("HistoricalFundamental received no valid prices.")

        if final_time is None:
            final_time = len(prices) - 1

        if len(prices) < final_time + 1:
            raise ValueError(
                f"Need at least {final_time + 1} prices, got {len(prices)}."
            )

        self.final_time = final_time
        self.fundamental_values = prices[: final_time + 1].astype(float)

        # compatibility values for agents that still expect mean-reversion info
        self.mean = float(np.mean(self.fundamental_values))
        self.r = 0.0

    def get_value_at(self, time: int) -> float:
        return float(self.fundamental_values[time])

    def get_fundamental_values(self):
        return self.fundamental_values

    def get_final_fundamental(self) -> float:
        return float(self.fundamental_values[-1])

    def get_info(self):
        return self.mean, self.r, self.final_time

    # not required by Fundamental ABC, but useful for compatibility
    def get_mean(self) -> float:
        return self.mean

    def get_r(self) -> float:
        return self.r