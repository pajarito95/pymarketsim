import numpy as np
import pandas as pd
from marketsim.fundamental.fundamental_abc import Fundamental


class HistoricalFundamental(Fundamental):
    def __init__(self, prices, final_time: int = None):
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) == 0: raise ValueError("HistoricalFundamental received no valid prices.")

        if final_time is None: final_time = len(prices) - 1

        if len(prices) < final_time + 1: raise ValueError(f"Need at least {final_time + 1} prices, got {len(prices)}.")

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


class ConstantHistoricalFundamental(Fundamental):
    """
    Pick one historical value and hold it fixed for the entire simulation.
    """
    def __init__(self, prices, final_time: int, selected_idx: int = 0):
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]

        if len(prices) == 0: raise ValueError("ConstantHistoricalFundamental received no valid prices.")

        selected_idx = int(max(0, min(selected_idx, len(prices) - 1)))
        self.final_time = int(final_time)
        self.constant_value = float(prices[selected_idx])
        self.fundamental_values = np.full(self.final_time + 1, self.constant_value, dtype=float)
        self.mean = self.constant_value
        self.r = 0.0

    def get_value_at(self, time: int) -> float:
        return self.constant_value

    def get_fundamental_values(self):
        return self.fundamental_values

    def get_final_fundamental(self) -> float:
        return self.constant_value

    def get_info(self):
        return self.mean, self.r, self.final_time

    def get_mean(self) -> float:
        return self.mean

    def get_r(self) -> float:
        return self.r


class HistoricalPiecewiseFundamental:
    """
    Holds one historical daily value constant over each bundle of intraday steps.
    Example: bundle_size=390 means one daily value is used for 390 simulation steps.
    """
    def __init__(self, prices: np.ndarray, final_time: int, bundle_size: int = 390):
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]
        if len(prices) == 0:
            raise ValueError("Prices must contain at least one valid value.")

        self.prices = prices
        self.final_time = int(final_time)
        self.bundle_size = int(bundle_size)

    def _daily_idx(self, time: int) -> int:
        idx = int(time) // self.bundle_size
        return min(idx, len(self.prices) - 1)

    def get_value_at(self, time: int) -> float:
        return float(self.prices[self._daily_idx(time)])

    def get_fundamental_values(self) -> np.ndarray:
        vals = np.zeros(self.final_time + 1, dtype=float)
        for t in range(self.final_time + 1):
            vals[t] = self.get_value_at(t)
        return vals

    def get_final_fundamental(self) -> float:
        return float(self.get_value_at(self.final_time))

    def get_mean(self) -> float:
        return float(np.mean(self.prices))

    def get_r(self) -> float:
        return 0.0

    def get_info(self):
        return self.get_mean(), self.get_r(), self.final_time


class HistoricalInterpolatedDriftFundamental:
    """
    Uses the historical daily value as an anchor for each intraday bundle, but allows within-bundle stochastic movement around that anchor using a mean-reverting update.

    f_t = anchor_t + (1-kappa)*(f_{t-1} - anchor_t) + eps_t
    """
    def __init__(
        self,
        prices: np.ndarray,
        final_time: int,
        bundle_size: int = 390,
        kappa: float = 0.05,
        shock_var: float = 1.0,
        shock_mean: float = 0.0,
        seed: int | None = None,
    ):
        prices = np.asarray(prices, dtype=float)
        prices = prices[~np.isnan(prices)]
        if len(prices) == 0:
            raise ValueError("Prices must contain at least one valid value.")

        self.prices = prices
        self.final_time = int(final_time)
        self.bundle_size = int(bundle_size)
        self.kappa = float(kappa)
        self.shock_var = float(max(shock_var, 1e-8))
        self.shock_std = float(np.sqrt(self.shock_var))
        self.shock_mean = float(shock_mean)

        rng = np.random.default_rng(seed)
        self.fundamental_values = np.zeros(self.final_time + 1, dtype=float)

        # initialize at first anchor
        self.fundamental_values[0] = float(self.prices[0])

        for t in range(1, self.final_time + 1):
            anchor = self._anchor_value(t)
            shock = rng.normal(loc=self.shock_mean, scale=self.shock_std)
            prev = self.fundamental_values[t - 1]
            new_val = anchor + (1.0 - self.kappa) * (prev - anchor) + shock
            self.fundamental_values[t] = max(0.0, float(new_val))

    def _daily_idx(self, time: int) -> int:
        idx = int(time) // self.bundle_size
        return min(idx, len(self.prices) - 1)

    def _anchor_value(self, time: int) -> float:
        return float(self.prices[self._daily_idx(time)])

    def get_value_at(self, time: int) -> float:
        return float(self.fundamental_values[int(time)])

    def get_fundamental_values(self) -> np.ndarray:
        return self.fundamental_values.copy()

    def get_final_fundamental(self) -> float:
        return float(self.fundamental_values[-1])

    def get_mean(self) -> float:
        return float(np.mean(self.prices))

    def get_r(self) -> float:
        return float(self.kappa)

    def get_info(self):
        return self.get_mean(), self.get_r(), self.final_time