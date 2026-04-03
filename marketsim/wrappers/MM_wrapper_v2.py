import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import torch
import torch.distributions as dist
from collections import defaultdict

from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.fundamental.historical import HistoricalFundamental
from marketsim.agent.noise_ZI_agent import ZIAgent
from marketsim.agent.informed_ZI import ZIAgent as InformedZIAgent
from marketsim.agent.market_maker_beta import MMAgent
from marketsim.wrappers.metrics import volume_imbalance, queue_imbalance, realized_volatility, relative_strength_index, midprice_move


def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()


class MMEnv(gym.Env):
    def __init__(
        self,
        num_background_agents: int,
        sim_time: int,
        num_assets: int = 1,
        lam: float = 75e-3,
        lamMM: float = 5e-3,
        informedZI: bool = False,
        mean: float = 1e5,
        r: float = 0.05,
        shock_var: float = 5e6,
        q_max: int = 10,
        est_var: float = 1e6,
        pv_var: float = 5e6,
        shade=None,
        n_levels: int = 21,
        total_volume: int = 100,
        xi: float = 50,
        omega: float = 10,
        beta_params: dict = None,
        policy: bool = False,
        normalizers=None,
        fundamental_type: str = "mean_reverting",
        historical_prices=None,
        seed: int | None = None,
    ):
        super().__init__()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.seed_value = seed

        if shade is None:
            shade = [10, 30]

        self.num_agents = num_background_agents
        self.total_num_agents = self.num_agents + 1
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.time = 0

        self.lamMM = lamMM
        self.normalizers = normalizers

        self.mean = mean
        self.shock_var = shock_var
        self.r = r

        self.fundamental_type = fundamental_type
        self.historical_prices = historical_prices

        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.arrivals_MM = defaultdict(list)
        self.arrival_times_MM = sample_arrivals(lamMM, self.arrivals_sampled)
        self.arrival_index_MM = 0

        self.markets = []
        if num_assets > 1:
            raise NotImplementedError("Only support single market currently")

        for _ in range(num_assets):
            fundamental = self._build_fundamental()
            self.markets.append(Market(fundamental=fundamental, time_steps=sim_time))

        self.agents = {}
        for agent_id in range(num_background_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

            if informedZI and agent_id >= int(num_background_agents / 2):
                self.agents[agent_id] = InformedZIAgent(
                    agent_id=agent_id,
                    market=self.markets[0],
                    q_max=q_max,
                    shade=shade,
                    pv_var=pv_var,
                )
            else:
                self.agents[agent_id] = ZIAgent(
                    agent_id=agent_id,
                    market=self.markets[0],
                    q_max=q_max,
                    shade=shade,
                    pv_var=pv_var,
                    est_var=est_var,
                )

        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item()].append(self.num_agents)
        self.arrival_index_MM += 1

        self.MM = MMAgent(
            agent_id=self.num_agents,
            market=self.markets[0],
            n_levels=n_levels,
            total_volume=total_volume,
            xi=xi,
            omega=omega,
            beta_params=beta_params,
            policy=policy,
        )

        self.spreads = []
        self.midprices = []
        self.inventory = []
        self.value_MM = 0
        self.total_quantity = 0
        self.MM_quantity = 0

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(5,),
            dtype=np.float64,
        )

        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1, 0.1, 0.1]),
            high=np.array([5.0, 5.0, 5.0, 5.0]),
            dtype=np.float64,
        )

        self.observation = None

    def _build_fundamental(self):
        if self.fundamental_type == "mean_reverting":
            return LazyGaussianMeanReverting(
                mean=self.mean,
                final_time=self.sim_time + 1,
                r=self.r,
                shock_var=self.shock_var,
            )

        elif self.fundamental_type == "historical":
            if self.historical_prices is None:
                raise ValueError(
                    "historical_prices must be provided when fundamental_type='historical'"
                )
            return HistoricalFundamental(
                prices=self.historical_prices,
                final_time=self.sim_time,
            )

        else:
            raise ValueError(f"Unknown fundamental_type: {self.fundamental_type}")

    def get_obs(self):
        return self.observation

    def update_obs(self):
        time_left = self.sim_time - self.time
        fundamental_value = self.markets[0].fundamental.get_value_at(self.time)
        best_ask = self.markets[0].order_book.get_best_ask()
        best_bid = self.markets[0].order_book.get_best_bid()
        MMinvt = self.MM.position

        midprice_delta = midprice_move(self.MM.market)
        vol_imbalance = volume_imbalance(self.MM.market)
        que_imbalance = queue_imbalance(self.MM.market)
        vr = realized_volatility(self.MM.market)
        rsi = relative_strength_index(self.MM.market)

        self.observation = self.normalization(
            time_left=time_left,
            fundamental_value=fundamental_value,
            best_ask=best_ask,
            best_bid=best_bid,
            MMinvt=MMinvt,
            midprice_delta=midprice_delta,
            vol_imbalance=vol_imbalance,
            que_imbalance=que_imbalance,
            vr=vr,
            rsi=rsi,
        )

    def normalization(
        self,
        time_left: int,
        fundamental_value: float,
        best_ask: float,
        best_bid: float,
        MMinvt: float,
        midprice_delta: float,
        vol_imbalance: float,
        que_imbalance: float,
        vr: float,
        rsi: float,
    ):
        if self.normalizers is None:
            return np.array([time_left, fundamental_value, best_ask, best_bid, MMinvt])

        time_left /= self.sim_time
        fundamental_value /= self.normalizers["fundamental"]

        if math.isinf(best_ask):
            best_ask = 1
        else:
            best_ask /= self.normalizers["fundamental"]

        if math.isinf(best_bid):
            best_bid = 0
        else:
            best_bid /= self.normalizers["fundamental"]

        MMinvt /= self.normalizers["invt"]

        midprice_delta /= 1e2
        rsi /= 100

        return np.array([time_left, fundamental_value, best_ask, best_bid, MMinvt])

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.time = 0
        self.observation = None

        for market in self.markets:
            fundamental = self._build_fundamental()
            market.reset(fundamental=fundamental)

        for agent_id in self.agents:
            self.agents[agent_id].reset()

        self.MM.reset()

        self.spreads = []
        self.midprices = []
        self.inventory = []
        self.value_MM = 0
        self.total_quantity = 0
        self.MM_quantity = 0

        self.reset_arrivals()

        _, end = self.run_until_next_MM_arrival()
        if end:
            raise ValueError("An episode without MM. Length of an episode should be set large.")

        return self.get_obs(), {}

    def reset_arrivals(self):
        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
        self.arrival_index = 0

        self.arrivals_MM = defaultdict(list)
        self.arrival_times_MM = sample_arrivals(self.lamMM, self.arrivals_sampled)
        self.arrival_index_MM = 0

        for agent_id in range(self.num_agents):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1

        self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item()].append(self.num_agents)
        self.arrival_index_MM += 1

    def step(self, action):
        if self.time < self.sim_time:
            self.MM_step(action)
            self.agents_step()
            self.market_step(agent_only=False)
            self.time += 1
            reward, end = self.run_until_next_MM_arrival()
            if end:
                return self.end_sim()
            return self.get_obs(), reward, False, False, {}
        else:
            return self.end_sim()

    def agents_step(self):
        agents = self.arrivals[self.time]
        if len(agents) != 0:
            for market in self.markets:
                market.event_queue.set_time(self.time)
                for agent_id in agents:
                    agent = self.agents[agent_id]
                    market.withdraw_all(agent_id)
                    side = random.choice([BUY, SELL])
                    orders = agent.take_action(side)
                    market.add_orders(orders)

                    if self.arrival_index == self.arrivals_sampled:
                        self.arrival_times = sample_arrivals(self.lam, self.arrivals_sampled)
                        self.arrival_index = 0
                    self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                    self.arrival_index += 1

    def MM_step(self, action):
        for market in self.markets:
            market.event_queue.set_time(self.time)
            market.withdraw_all(self.num_agents)
            #orders = self.MM.take_action(action)
            action = np.asarray(action, dtype=float)
            action = np.clip(action, 0.1, 5.0)
            orders = self.MM.take_action(action)
            market.add_orders(orders)

            if self.arrival_index_MM == self.arrivals_sampled:
                self.arrival_times_MM = sample_arrivals(self.lamMM, self.arrivals_sampled)
                self.arrival_index_MM = 0
            self.arrivals_MM[self.arrival_times_MM[self.arrival_index_MM].item() + 1 + self.time].append(self.num_agents)
            self.arrival_index_MM += 1

    def market_step(self, agent_only=True, verbose=False):
        for market in self.markets:
            new_orders = market.step()
            for matched_order in new_orders:
                agent_id = matched_order.order.agent_id
                quantity = matched_order.order.order_type * matched_order.order.quantity
                cash = -matched_order.price * matched_order.order.quantity * matched_order.order.order_type
                if agent_id == self.num_agents:
                    self.MM.update_position(quantity, cash)
                else:
                    self.agents[agent_id].update_position(quantity, cash)

                self.total_quantity += abs(quantity)
                if agent_id == self.num_agents:
                    self.MM_quantity += abs(quantity)

            best_ask = market.order_book.get_best_ask()
            best_bid = market.order_book.get_best_bid()
            self.spreads.append(best_ask - best_bid)
            self.midprices.append((best_ask + best_bid) / 2)
            self.inventory.append(self.MM.position)

    def end_sim(self):
        fundamental_val = self.markets[0].get_final_fundamental()
        current_value = self.MM.position * fundamental_val + self.MM.cash
        reward = current_value - self.MM.last_value
        self.MM.last_value = current_value
        self.value_MM = current_value

        reward_scale = 1.0 if self.normalizers is None else self.normalizers["reward"]
        return self.get_obs(), reward / reward_scale, True, False, {}

    def run_until_next_MM_arrival(self):
        while len(self.arrivals_MM[self.time]) == 0 and self.time < self.sim_time:
            self.agents_step()
            self.market_step(agent_only=True)
            self.time += 1

        if self.time >= self.sim_time:
            return 0, True
        else:
            fundamental_val = self.markets[0].get_final_fundamental()
            current_value = self.MM.position * fundamental_val + self.MM.cash
            reward = current_value - self.MM.last_value
            self.MM.last_value = current_value
            self.update_obs()

            reward_scale = 1.0 if self.normalizers is None else self.normalizers["reward"]
            return reward / reward_scale, False

    def get_stats(self):
        return {
            "spreads": self.spreads.copy(),
            "midprices": self.midprices.copy(),
            "inventory": self.inventory.copy(),
            "total_quantity": self.total_quantity,
            "MM_quantity": self.MM_quantity,
            "MM_value": self.value_MM,
        }