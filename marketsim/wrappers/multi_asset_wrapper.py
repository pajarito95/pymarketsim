import math
import random
from collections import defaultdict
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from marketsim.agent.cross_asset_background_agent import CrossAssetBackgroundAgent
from marketsim.agent.rl_participation_agent import RLParticipationAgent
from marketsim.agent.market_maker import MMAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.fundamental.historical import HistoricalFundamental
from marketsim.fundamental.mean_reverting import GaussianMeanReverting
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
        use_market_makers: bool = True,
        lam_mm: float = 0.10,
        mm_xi: float = 0.5,
        mm_K: int = 3,
        mm_omega: float = 2.0,
        seed: int | None = None,
        bg_latency: int = 0,
        rl_latency: int = 0,
        fundamental_mode: str = "historical",   # "historical" or "synthetic"
        synthetic_kappa: float = 0.05,          # mean-reversion strength r
        synthetic_sigma_scale: float = 1.0,     # multiplier on estimated shock variance
    ):
        super().__init__()

        if len(historical_series) != 2:
            raise ValueError("historical_series must contain exactly 2 assets.")
        if fundamental_mode not in {"historical", "synthetic"}:
            raise ValueError("fundamental_mode must be 'historical' or 'synthetic'.")

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

        self.use_market_makers = bool(use_market_makers)
        self.lam_mm = float(lam_mm)
        self.mm_xi = float(mm_xi)
        self.mm_K = int(mm_K)
        self.mm_omega = float(mm_omega)

        self.seed_value = seed
        self.bg_latency = max(0, int(bg_latency))
        self.rl_latency = max(0, int(rl_latency))

        self.fundamental_mode = fundamental_mode
        self.synthetic_kappa = float(synthetic_kappa)
        self.synthetic_sigma_scale = float(synthetic_sigma_scale)

        self._rl_order_counter = 0

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
        self.market_makers: Dict[str, MMAgent] = {}
        self.mm_agent_ids: Dict[str, int] = {}

        self.arrivals_mm = defaultdict(list)
        self.arrival_times_mm = None
        self.arrival_idx_mm = 0

        self.arrivals_bg = defaultdict(list)
        self.arrival_times_bg = None
        self.arrival_idx_bg = 0

        self.arrivals_rl = defaultdict(list)
        self.arrival_times_rl = None
        self.arrival_idx_rl = 0

        self.time = 0
        self.decision_event_count = 0
        self.last_net_worth = None

        # step diagnostics
        self.last_step_matched_counts = {ticker: 0 for ticker in self.tickers}

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

    def _make_fundamental(self, ticker: str):
        ref_prices = np.asarray(self.historical_series[ticker], dtype=float)
        ref_prices = ref_prices[~np.isnan(ref_prices)]

        if len(ref_prices) == 0:
            raise ValueError(f"No valid prices available for ticker {ticker}.")

        if self.fundamental_mode == "historical":
            return HistoricalFundamental(prices=ref_prices, final_time=self.sim_time)

        # synthetic = Gaussian mean-reverting process calibrated loosely to the
        # scale of the historical series
        mean = float(np.mean(ref_prices))

        if len(ref_prices) > 1:
            log_rets = np.diff(np.log(np.maximum(ref_prices, 1e-8)))
            ret_var = float(np.var(log_rets))
            # convert return-scale variation to price-scale shock variance
            shock_var = ret_var * (mean ** 2)
        else:
            shock_var = (0.01 * mean) ** 2

        shock_var = max(shock_var * self.synthetic_sigma_scale, 1e-8)

        # GaussianMeanReverting allocates exactly `final_time` slots, so use
        # sim_time + 1 to allow indexing from t=0,...,sim_time
        return GaussianMeanReverting(
            final_time=self.sim_time + 1,
            mean=mean,
            r=self.synthetic_kappa,
            shock_var=shock_var,
            shock_mean=0.0,
        )

    def _build_env(self):
        self.markets = {}
        for ticker in self.tickers:
            fundamental = self._make_fundamental(ticker)
            self.markets[ticker] = Market(fundamental=fundamental, time_steps=self.sim_time)

        self.market_makers = {}
        self.mm_agent_ids = {}
        if self.use_market_makers:
            for idx, ticker in enumerate(self.tickers):
                mm_id = 800_000 + idx
                self.mm_agent_ids[ticker] = mm_id
                self.market_makers[ticker] = MMAgent(
                    agent_id=mm_id,
                    market=self.markets[ticker],
                    xi=self.mm_xi,
                    K=self.mm_K,
                    omega=self.mm_omega,
                )

        self.arrivals_mm = defaultdict(list)
        if self.use_market_makers:
            self.arrival_times_mm = sample_arrivals_numpy(self.lam_mm, 10000)
            self.arrival_idx_mm = 0
            for ticker in self.tickers:
                next_t = int(self.arrival_times_mm[self.arrival_idx_mm]) + 1
                self.arrivals_mm[next_t].append(ticker)
                self.arrival_idx_mm += 1
        else:
            self.arrival_times_mm = None
            self.arrival_idx_mm = 0

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

    def _schedule_next_mm_arrival(self, ticker: str):
        if not self.use_market_makers:
            return

        if self.arrival_idx_mm >= len(self.arrival_times_mm):
            self.arrival_times_mm = sample_arrivals_numpy(self.lam_mm, 10000)
            self.arrival_idx_mm = 0

        next_t = int(self.arrival_times_mm[self.arrival_idx_mm]) + 1 + self.time
        self.arrivals_mm[next_t].append(ticker)
        self.arrival_idx_mm += 1

    def _process_market_makers_for_current_time(self):
        if not self.use_market_makers:
            return

        tickers_now = self.arrivals_mm[self.time]
        for ticker in tickers_now:
            mm = self.market_makers[ticker]
            market = self.markets[ticker]

            market.event_queue.set_time(self.time)
            market.withdraw_all(mm.get_id())

            orders = mm.take_action()
            market.add_orders(orders)

            self._schedule_next_mm_arrival(ticker)

    def _seed_market_makers_at_t0(self):
        if not self.use_market_makers:
            return

        for ticker, mm in self.market_makers.items():
            market = self.markets[ticker]
            market.event_queue.set_time(0)

            market.withdraw_all(mm.get_id())
            orders = mm.take_action()
            market.add_orders(orders)

        self._process_all_markets_for_current_time()

    def _cancel_rl_orders(self):
        for market in self.markets.values():
            market.withdraw_all(self.rl_agent_id)

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.time = 0
        self.decision_event_count = 0
        self.rl_agent.reset()
        self._build_env()

        self._seed_market_makers_at_t0()

        self._advance_to_next_rl_event()
        self.last_net_worth = self._get_net_worth()

        self._rl_order_counter = 0
        self.last_step_matched_counts = {ticker: 0 for ticker in self.tickers}

        return self._get_obs(), {}

    def step(self, action: int):
        if self.time >= self.sim_time or self.decision_event_count >= self.max_decision_events:
            return self._get_obs(), 0.0, True, False, {}

        nw_before = self._get_net_worth()
        invalid = 0
        order_submitted = 0
        submitted_order_id = None

        if self.rl_agent_id in self.arrivals_rl[self.time]:
            invalid, order_submitted, submitted_order_id = self._execute_rl_action(action)
            self._schedule_next_rl_arrival()

        self._process_market_makers_for_current_time()
        self._process_background_agents_for_current_time()
        rl_filled_order_ids = self._process_all_markets_for_current_time()

        order_filled = int(submitted_order_id is not None and submitted_order_id in rl_filled_order_ids)
        self.time += 1
        self.decision_event_count += 1

        done = self._advance_to_next_rl_event()
        nw_after = self._get_net_worth()

        reward = (nw_after - nw_before) - self.lambda_invalid * invalid
        self.last_net_worth = nw_after

        terminated = done or (self.decision_event_count >= self.max_decision_events)

        info = {
            "net_worth": nw_after,
            "invalid_action": invalid,
            "time": self.time,
            "order_submitted": order_submitted,
            "order_filled": order_filled,
        }

        return self._get_obs(), float(reward), terminated, False, info

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

    def _execute_rl_action(self, action: int):
        """
        Submit the RL action as a real limit order into the LOB.

        Returns
        -------
        invalid : int
            1 if the action is invalid, else 0
        order_submitted : int
            1 if an RL order was submitted to the book, else 0
        submitted_order_id : int | None
            The RL order id if submitted, else None
        """
        self._cancel_rl_orders()

        if action == 0:
            return 0, 0, None

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
            return 1, 0, None

        market = self.markets[ticker]
        market.event_queue.set_time(self.time)

        best_ask = market.order_book.get_best_ask()
        best_bid = market.order_book.get_best_bid()
        fundamental = market.get_fundamental_value()

        if side == BUY:
            if not math.isinf(best_ask):
                limit_price = float(best_ask)
            elif not math.isinf(best_bid):
                limit_price = float(best_bid)
            else:
                limit_price = float(fundamental)

            if self.rl_agent.cash < limit_price:
                return 1, 0, None

            quantity = 1

        else:  # SELL
            if self.rl_agent.get_inventory(ticker) <= 0:
                return 1, 0, None

            if not math.isinf(best_bid):
                limit_price = float(best_bid)
            elif not math.isinf(best_ask):
                limit_price = float(best_ask)
            else:
                limit_price = float(fundamental)

            quantity = 1

        self._rl_order_counter += 1
        order_id = self.rl_agent_id * 1_000_000 + self._rl_order_counter

        order = Order(
            price=limit_price,
            quantity=quantity,
            agent_id=self.rl_agent_id,
            time=self.time,
            order_type=side,
            order_id=order_id,
        )

        market.add_orders([order])

        return 0, 1, order_id

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
        rl_filled_order_ids = set()
        self.last_step_matched_counts = {ticker: 0 for ticker in self.tickers}

        for ticker, market in self.markets.items():
            market.event_queue.set_time(self.time)
            new_orders = market.step()
            self.last_step_matched_counts[ticker] = len(new_orders)

            for matched_order in new_orders:
                agent_id = matched_order.order.agent_id
                quantity = int(matched_order.order.order_type * matched_order.order.quantity)
                cash_delta = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type

                if agent_id == self.rl_agent_id:
                    self.rl_agent.update_position(ticker, quantity, cash_delta)
                    rl_filled_order_ids.add(matched_order.order.order_id)

                elif agent_id in self.background_agents:
                    self.background_agents[agent_id].update_position(ticker, quantity, cash_delta)

                elif self.use_market_makers and agent_id in self.mm_agent_ids.values():
                    self.market_makers[ticker].update_position(quantity, cash_delta)

        return rl_filled_order_ids

    def _advance_to_next_rl_event(self) -> bool:
        while self.time < self.sim_time and len(self.arrivals_rl[self.time]) == 0:
            self._process_market_makers_for_current_time()
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

            scale = max(float(self.historical_series[ticker][0]), 1e-8)

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

    # ------------------------------------------------------------------
    # Snapshot helpers for logging / later distributional analysis
    # ------------------------------------------------------------------

    def get_market_snapshot(self, episode: int | None = None) -> List[dict]:
        rows = []
        for ticker, market in self.markets.items():
            bid = market.order_book.get_best_bid()
            ask = market.order_book.get_best_ask()
            bid_out = None if math.isinf(bid) else float(bid)
            ask_out = None if math.isinf(ask) else float(ask)

            if bid_out is not None and ask_out is not None:
                mid = 0.5 * (bid_out + ask_out)
                spread = ask_out - bid_out
            else:
                mid = float(market.get_fundamental_value())
                spread = None

            rows.append({
                "episode": episode,
                "time": int(self.time),
                "ticker": ticker,
                "fundamental_mode": self.fundamental_mode,
                "fundamental": float(market.get_fundamental_value()),
                "best_bid": bid_out,
                "best_ask": ask_out,
                "midprice": float(mid),
                "spread": spread,
                "matched_orders": int(self.last_step_matched_counts.get(ticker, 0)),
            })
        return rows

    def get_agent_snapshot(self, episode: int | None = None) -> List[dict]:
        rows = []
        marks = self._mark_prices()

        # RL agent
        rl_row = {
            "episode": episode,
            "time": int(self.time),
            "agent_id": int(self.rl_agent_id),
            "agent_type": "rl",
            "latency": int(self.rl_latency),
            "cash": float(self.rl_agent.cash),
            "net_worth": float(self.rl_agent.net_worth(marks)),
        }
        for ticker in self.tickers:
            inv = int(self.rl_agent.get_inventory(ticker))
            rl_row[f"{ticker}_holdings"] = inv
            rl_row[f"participates_{ticker}"] = int(inv > 0)
        rows.append(rl_row)

        # Background agents
        for agent_id, agent in self.background_agents.items():
            row = {
                "episode": episode,
                "time": int(self.time),
                "agent_id": int(agent_id),
                "agent_type": "background",
                "latency": int(getattr(agent, "latency", 0)),
                "cash": float(agent.cash),
                "net_worth": float(agent.net_worth(marks)),
            }
            for ticker in self.tickers:
                inv = int(agent.inventory[ticker])
                row[f"{ticker}_holdings"] = inv
                row[f"participates_{ticker}"] = int(inv > 0)
            rows.append(row)

        # Market makers
        if self.use_market_makers:
            for ticker, mm in self.market_makers.items():
                mark = marks[ticker]
                position = int(mm.position)
                net_worth = float(mm.cash + position * mark)

                row = {
                    "episode": episode,
                    "time": int(self.time),
                    "agent_id": int(mm.get_id()),
                    "agent_type": "market_maker",
                    "latency": 0,
                    "cash": float(mm.cash),
                    "net_worth": net_worth,
                }
                for tk in self.tickers:
                    inv = position if tk == ticker else 0
                    row[f"{tk}_holdings"] = int(inv)
                    row[f"participates_{tk}"] = int(inv > 0)
                rows.append(row)

        return rows