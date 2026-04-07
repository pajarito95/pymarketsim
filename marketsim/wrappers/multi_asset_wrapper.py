import math
import random
from collections import defaultdict
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from marketsim.agent.cross_asset_background_agent import CrossAssetBackgroundAgent
from marketsim.agent.rl_participation_agent import RLParticipationAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.fundamental.historical import HistoricalFundamental
from marketsim.market.market import Market
from marketsim.wrappers.metrics import midprice_move, realized_volatility, relative_strength_index


def sample_arrivals_numpy(p: float, num_samples: int) -> np.ndarray:
    return np.random.geometric(p, size=num_samples) - 1


class MultiAssetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        historical_series: Dict[str, np.ndarray],
        sim_time: int,
        max_decision_events: int = 200,
        num_background_agents: int = 50,
        lam_bg: float = 0.10,
        lam_rl: float = 0.05,
        q_max: int = 10,
        pv_var: float = 1.0,
        zi_shade: List[float] | None = None,
        initial_cash: float = 100_000.0,
        lambda_invalid: float = 1.0,
        seed: int | None = None,
        bg_latency: int = 0,
        rl_latency: int = 0,
    ):
        super().__init__()

        if len(historical_series) != 2:
            raise ValueError("historical_series must contain exactly 2 assets.")

        if zi_shade is None:
            zi_shade = [0.05, 0.5]

        self.tickers = list(historical_series.keys())
        self.historical_series = historical_series
        self.sim_time = int(sim_time)
        self.max_decision_events = int(max_decision_events)
        self.num_background_agents = int(num_background_agents)
        self.lam_bg = float(lam_bg)
        self.lam_rl = float(lam_rl)
        self.q_max = int(q_max)
        self.pv_var = float(pv_var)
        self.zi_shade = zi_shade
        self.initial_cash = float(initial_cash)
        self.lambda_invalid = float(lambda_invalid)
        self.seed_value = seed
        self.bg_latency = max(0, int(bg_latency))
        self.rl_latency = max(0, int(rl_latency))

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.rl_agent_id = 900_000
        self.rl_agent = RLParticipationAgent(
            agent_id=self.rl_agent_id,
            tickers=self.tickers,
            initial_cash=self.initial_cash,
        )

        self.markets: Dict[str, Market] = {}
        self.background_agents: Dict[int, CrossAssetBackgroundAgent] = {}

        self.arrivals_bg = defaultdict(list)
        self.arrival_times_bg = None
        self.arrival_idx_bg = 0

        self.arrivals_rl = defaultdict(list)
        self.arrival_times_rl = None
        self.arrival_idx_rl = 0

        self.time = 0
        self.decision_event_count = 0
        self.last_net_worth = None

        # obs = time_left, cash_norm, nw_norm,
        # then per asset:
        # fundamental, best_bid, best_ask, inv_norm, midprice_move, volatility, rsi
        obs_dim = 3 + 2 * 7
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        self._build_env()

    def _build_env(self):
        self.markets = {}
        for ticker in self.tickers:
            prices = self.historical_series[ticker]
            fundamental = HistoricalFundamental(prices=prices, final_time=self.sim_time)
            self.markets[ticker] = Market(fundamental=fundamental, time_steps=self.sim_time)

        self.background_agents = {}
        for agent_id in range(self.num_background_agents):
            self.background_agents[agent_id] = CrossAssetBackgroundAgent(
                agent_id=agent_id,
                tickers=self.tickers,
                q_max=self.q_max,
                pv_var=self.pv_var,
                shade=self.zi_shade,
                eta=1.0,
                initial_cash=self.initial_cash,
                latency=self.bg_latency,
            )

        self.arrivals_bg = defaultdict(list)
        self.arrival_times_bg = sample_arrivals_numpy(self.lam_bg, 10000)
        self.arrival_idx_bg = 0
        for agent_id in range(self.num_background_agents):
            self.arrivals_bg[int(self.arrival_times_bg[self.arrival_idx_bg])].append(agent_id)
            self.arrival_idx_bg += 1

        self.arrivals_rl = defaultdict(list)
        self.arrival_times_rl = sample_arrivals_numpy(self.lam_rl, 10000)
        self.arrival_idx_rl = 0
        self.arrivals_rl[int(self.arrival_times_rl[self.arrival_idx_rl])].append(self.rl_agent_id)
        self.arrival_idx_rl += 1

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.time = 0
        self.decision_event_count = 0
        self.rl_agent.reset()
        self._build_env()

        self._advance_to_next_rl_event()
        self.last_net_worth = self._get_net_worth()

        return self._get_obs(), {}

    def step(self, action: int):
        if self.time >= self.sim_time or self.decision_event_count >= self.max_decision_events:
            return self._get_obs(), 0.0, True, False, {}

        nw_before = self._get_net_worth()
        invalid = 0

        if self.rl_agent_id in self.arrivals_rl[self.time]:
            invalid = self._execute_rl_action(action)
            self._schedule_next_rl_arrival()

        self._process_background_agents_for_current_time()
        self._process_all_markets_for_current_time()

        self.time += 1
        self.decision_event_count += 1

        done = self._advance_to_next_rl_event()
        nw_after = self._get_net_worth()

        reward = (nw_after - nw_before) - self.lambda_invalid * invalid
        self.last_net_worth = nw_after

        terminated = done or (self.decision_event_count >= self.max_decision_events)
        return self._get_obs(), float(reward), terminated, False, {
            "net_worth": nw_after,
            "invalid_action": invalid,
            "time": self.time,
        }

    def _schedule_next_bg_arrival(self, agent_id: int):
        if self.arrival_idx_bg >= len(self.arrival_times_bg):
            self.arrival_times_bg = sample_arrivals_numpy(self.lam_bg, 10000)
            self.arrival_idx_bg = 0

        next_t = int(self.arrival_times_bg[self.arrival_idx_bg]) + 1 + self.time
        self.arrivals_bg[next_t].append(agent_id)
        self.arrival_idx_bg += 1

    def _schedule_next_rl_arrival(self):
        if self.arrival_idx_rl >= len(self.arrival_times_rl):
            self.arrival_times_rl = sample_arrivals_numpy(self.lam_rl, 10000)
            self.arrival_idx_rl = 0

        next_t = int(self.arrival_times_rl[self.arrival_idx_rl]) + 1 + self.time
        self.arrivals_rl[next_t].append(self.rl_agent_id)
        self.arrival_idx_rl += 1

    def _execute_rl_action(self, action: int) -> int:
        if action == 0:
            return 0

        if action == 1:
            ticker = self.tickers[0]
            side = BUY
        elif action == 2:
            ticker = self.tickers[0]
            side = SELL
        elif action == 3:
            ticker = self.tickers[1]
            side = BUY
        elif action == 4:
            ticker = self.tickers[1]
            side = SELL
        else:
            return 1

        market = self.markets[ticker]
        obs = self._get_rl_market_observation(ticker)
        book = obs["book"]

        best_ask = market.order_book.get_best_ask()
        best_bid = market.order_book.get_best_bid()
        fallback_price = market.get_fundamental_value()

        if side == BUY:
            exec_price = best_ask if not math.isinf(best_ask) else fallback_price
            if self.rl_agent.cash < exec_price:
                return 1

            # direct execution
            self.rl_agent.update_position(ticker, 1, -float(exec_price))
            return 0

        else:
            if self.rl_agent.get_inventory(ticker) <= 0:
                return 1

            exec_price = best_bid if not math.isinf(best_bid) else fallback_price

            # direct execution
            self.rl_agent.update_position(ticker, -1, float(exec_price))
            return 0

    def _process_background_agents_for_current_time(self):
        agents = self.arrivals_bg[self.time]
        for agent_id in agents:
            agent = self.background_agents[agent_id]

            for ticker, market in self.markets.items():
                market.event_queue.set_time(self.time)
                market.withdraw_all(agent_id)

            choice = agent.choose_action(self.markets)
            if choice is not None:
                ticker, order = choice
                self.markets[ticker].add_orders([order])

            self._schedule_next_bg_arrival(agent_id)

    def _process_all_markets_for_current_time(self):
        for ticker, market in self.markets.items():
            market.event_queue.set_time(self.time)
            new_orders = market.step()

            for matched_order in new_orders:
                agent_id = matched_order.order.agent_id
                quantity = int(matched_order.order.order_type * matched_order.order.quantity)
                cash_delta = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type

                if agent_id == self.rl_agent_id:
                    self.rl_agent.update_position(ticker, quantity, cash_delta)
                else:
                    self.background_agents[agent_id].update_position(ticker, quantity, cash_delta)

    def _advance_to_next_rl_event(self) -> bool:
        while self.time < self.sim_time and len(self.arrivals_rl[self.time]) == 0:
            self._process_background_agents_for_current_time()
            self._process_all_markets_for_current_time()
            self.time += 1

        return self.time >= self.sim_time

    def _mark_prices(self) -> Dict[str, float]:
        marks = {}
        for ticker, market in self.markets.items():
            bid = market.order_book.get_best_bid()
            ask = market.order_book.get_best_ask()
            if not math.isinf(bid) and not math.isinf(ask):
                marks[ticker] = float((bid + ask) / 2.0)
            else:
                marks[ticker] = float(market.get_fundamental_value())
        return marks

    def _get_net_worth(self) -> float:
        return self.rl_agent.net_worth(self._mark_prices())

    def _get_rl_market_observation(self, ticker: str) -> dict:
        market = self.markets[ticker]
        return market.get_observation(latency_steps=self.rl_latency)

    def _safe_metric(self, fn, market: Market) -> float:
        try:
            val = fn(market)
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                return 0.0
            return float(val)
        except Exception:
            return 0.0

    def _get_obs(self) -> np.ndarray:
        time_left = max(self.sim_time - self.time, 0) / max(self.sim_time, 1)
        cash_norm = self.rl_agent.cash / self.initial_cash
        nw_norm = self._get_net_worth() / self.initial_cash

        obs = [time_left, cash_norm, nw_norm]

        for ticker in self.tickers:
            market = self.markets[ticker]
            delayed_obs = self._get_rl_market_observation(ticker)
            book = delayed_obs["book"]
            fundamental = float(market.get_fundamental_value())

            best_bid = book.get("best_bid", None)
            best_ask = book.get("best_ask", None)

            if best_bid is None:
                best_bid = fundamental
            if best_ask is None:
                best_ask = fundamental

            inv = self.rl_agent.get_inventory(ticker) / max(self.q_max, 1)
            mpm = self._safe_metric(midprice_move, market)
            vol = self._safe_metric(realized_volatility, market)
            rsi = self._safe_metric(relative_strength_index, market) / 100.0

            scale = max(self.historical_series[ticker][0], 1e-8)

            obs.extend([
                fundamental / scale,
                float(best_bid) / scale,
                float(best_ask) / scale,
                inv,
                mpm,
                vol,
                rsi,
            ])
        
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(obs, dtype=np.float32)