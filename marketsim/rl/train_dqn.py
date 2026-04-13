import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

from marketsim.data.load_historical import load_two_asset_series
from marketsim.rl.dqn_agent import DQNAgent, DQNConfig
from marketsim.rl.replay_buffer import ReplayBuffer
from marketsim.wrappers.multi_asset_wrapper import MultiAssetEnv


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def summarize_run(summary_path: str) -> dict:
    records = []
    if not os.path.exists(summary_path):
        return {}

    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        return {}

    n = len(records)
    first_n = min(100, n)
    last_n = min(100, n)

    first = records[:first_n]
    last = records[-last_n:]

    def mean_of(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def max_of(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.max(vals)) if vals else None

    summary = {
        "episodes": n,
        "mean_return_first": mean_of("episode_return", first),
        "mean_return_last": mean_of("episode_return", last),
        "mean_invalid_first": mean_of("invalid_actions", first),
        "mean_invalid_last": mean_of("invalid_actions", last),
        "best_episode_return": max_of("episode_return", records),
        "mean_steps_last": mean_of("episode_steps", last),
        "mean_net_worth_last": mean_of("final_net_worth", last),
    }
    return summary


def derive_run_metrics(
    market_log_path: str | None = None,
    agent_log_path: str | None = None,
) -> dict:
    """
    Lightweight derived summary from market and agent logs.
    This is optional but useful for later analysis.
    """
    out: Dict[str, Any] = {}

    if market_log_path and os.path.exists(market_log_path):
        market_rows = []
        with open(market_log_path, "r", encoding="utf-8") as f:
            for line in f:
                market_rows.append(json.loads(line))

        if market_rows:
            by_ticker = {}
            for row in market_rows:
                by_ticker.setdefault(row["ticker"], []).append(row)

            for ticker, rows in by_ticker.items():
                spreads = [r["spread"] for r in rows if r.get("spread") is not None]
                mids = [r["midprice"] for r in rows if r.get("midprice") is not None]
                matched = [r.get("matched_orders", 0) for r in rows]

                if spreads:
                    out[f"{ticker}_avg_spread"] = float(np.mean(spreads))
                if mids and len(mids) > 1:
                    mids_arr = np.asarray(mids, dtype=float)
                    rets = np.diff(np.log(np.maximum(mids_arr, 1e-8)))
                    out[f"{ticker}_midprice_vol"] = float(np.std(rets))
                out[f"{ticker}_total_matched_orders"] = int(np.sum(matched))

    if agent_log_path and os.path.exists(agent_log_path):
        agent_rows = []
        with open(agent_log_path, "r", encoding="utf-8") as f:
            for line in f:
                agent_rows.append(json.loads(line))

        if agent_rows:
            # final time snapshots only
            max_ep = max(r["episode"] for r in agent_rows)
            ep_rows = [r for r in agent_rows if r["episode"] == max_ep]
            if ep_rows:
                max_t = max(r["time"] for r in ep_rows)
                final_rows = [r for r in ep_rows if r["time"] == max_t]

                for key in [k for k in final_rows[0].keys() if k.startswith("participates_")]:
                    vals = [r[key] for r in final_rows]
                    out[f"final_mean_{key}"] = float(np.mean(vals))

                net_worths = [r["net_worth"] for r in final_rows if r.get("net_worth") is not None]
                if net_worths:
                    out["final_mean_net_worth_all_agents"] = float(np.mean(net_worths))
                    out["final_std_net_worth_all_agents"] = float(np.std(net_worths))

    return out


def train(
    ticker_a="AAPL",
    ticker_b="XOM",
    sim_time=200,
    max_decision_events=100,
    num_background_agents=25,
    lam_bg=0.10,
    lam_rl=0.05,
    q_max=10,
    pv_var=1.0,
    zi_shade=(0.05, 0.5),
    initial_cash=100_000.0,
    lambda_invalid=1.0,
    bg_latency=0,
    rl_latency=0,
    gamma=0.99,
    lr=5e-4,
    batch_size=64,
    target_update_freq=1000,
    learning_starts=500,
    train_freq=4,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=5000,
    grad_clip=10.0,
    replay_capacity=100_000,
    episodes=2000,
    checkpoint_every=0,
    seed=42,
    run_tag=None,
    use_market_makers=True,
    lam_mm=0.10,
    mm_xi=0.5,
    mm_K=3,
    mm_omega=2.0,
    fundamental_mode: str = "historical",   # "historical" or "synthetic"
    synthetic_kappa: float = 0.05,
    synthetic_sigma_scale: float = 0.02,
    log_market: bool = True,
    log_agents: bool = True,
    output_root: str = "runs",
):
    csv_path = r"R:\sescott1\Masters\thesis\potential_codes\StockMARL\Stock-MARL-main\resources\datasets\train_dataV1.csv"

    historical_series = load_two_asset_series(
        csv_path=csv_path,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        sim_time=sim_time,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if run_tag is None:
        run_tag = "manual"

    run_name = f"dqn_{ticker_a}_{ticker_b}_{run_tag}_{timestamp}"
    out_dir = os.path.join(output_root, run_name)
    ensure_dir(out_dir)

    print("Saving outputs to:", os.path.abspath(out_dir))

    config_record = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "csv_path": csv_path,
        "sim_time": sim_time,
        "max_decision_events": max_decision_events,
        "num_background_agents": num_background_agents,
        "lam_bg": lam_bg,
        "lam_rl": lam_rl,
        "q_max": q_max,
        "pv_var": pv_var,
        "zi_shade": list(zi_shade),
        "initial_cash": initial_cash,
        "lambda_invalid": lambda_invalid,
        "bg_latency": bg_latency,
        "rl_latency": rl_latency,
        "gamma": gamma,
        "lr": lr,
        "batch_size": batch_size,
        "target_update_freq": target_update_freq,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_steps": epsilon_decay_steps,
        "grad_clip": grad_clip,
        "replay_capacity": replay_capacity,
        "episodes": episodes,
        "checkpoint_every": checkpoint_every,
        "seed": seed,
        "run_tag": run_tag,
        "use_market_makers": use_market_makers,
        "lam_mm": lam_mm,
        "mm_xi": mm_xi,
        "mm_K": mm_K,
        "mm_omega": mm_omega,
        "fundamental_mode": fundamental_mode,
        "synthetic_kappa": synthetic_kappa,
        "synthetic_sigma_scale": synthetic_sigma_scale,
        "log_market": log_market,
        "log_agents": log_agents,
        "output_root": output_root,
    }

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    env = MultiAssetEnv(
        historical_series=historical_series,
        sim_time=sim_time,
        max_decision_events=max_decision_events,
        num_background_agents=num_background_agents,
        lam_bg=lam_bg,
        lam_rl=lam_rl,
        q_max=q_max,
        pv_var=pv_var,
        zi_shade=list(zi_shade),
        initial_cash=initial_cash,
        lambda_invalid=lambda_invalid,
        use_market_makers=use_market_makers,
        lam_mm=lam_mm,
        mm_xi=mm_xi,
        mm_K=mm_K,
        mm_omega=mm_omega,
        seed=seed,
        bg_latency=bg_latency,
        rl_latency=rl_latency,
        fundamental_mode=fundamental_mode,
        synthetic_kappa=synthetic_kappa,
        synthetic_sigma_scale=synthetic_sigma_scale,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dqn = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        config=DQNConfig(
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            learning_starts=learning_starts,
            train_freq=train_freq,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            grad_clip=grad_clip,
        ),
    )
    replay_buffer = ReplayBuffer(capacity=replay_capacity)

    episode_log_path = os.path.join(out_dir, "episode_log.jsonl")
    market_log_path = os.path.join(out_dir, "market_log.jsonl")
    agent_log_path = os.path.join(out_dir, "agent_log.jsonl")

    latest_ckpt = os.path.join(out_dir, "latest.pt")
    best_ckpt = os.path.join(out_dir, "best.pt")

    best_return = -np.inf

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)

        # log initial state snapshot
        if log_market:
            for row in env.get_market_snapshot(episode=episode):
                append_jsonl(market_log_path, row)

        if log_agents:
            for row in env.get_agent_snapshot(episode=episode):
                append_jsonl(agent_log_path, row)

        done = False
        ep_return = 0.0
        ep_steps = 0
        losses = []
        invalid_count = 0
        last_info = {}

        while not done:
            action = dqn.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            ep_return += reward
            ep_steps += 1
            invalid_count += int(info.get("invalid_action", 0))
            last_info = info

            if dqn.total_steps > dqn.config.learning_starts and dqn.total_steps % dqn.config.train_freq == 0:
                loss = dqn.update(replay_buffer)
                if loss is not None:
                    losses.append(loss)

            if log_market:
                for row in env.get_market_snapshot(episode=episode):
                    append_jsonl(market_log_path, row)

            if log_agents:
                for row in env.get_agent_snapshot(episode=episode):
                    append_jsonl(agent_log_path, row)

            state = next_state

        avg_loss = float(np.mean(losses)) if losses else None
        record = {
            "episode": episode,
            "episode_return": ep_return,
            "episode_steps": ep_steps,
            "invalid_actions": invalid_count,
            "final_net_worth": last_info.get("net_worth", None),
            "avg_loss": avg_loss,
            "epsilon": dqn.epsilon(),
            "buffer_size": len(replay_buffer),
            "total_steps": dqn.total_steps,
        }
        append_jsonl(episode_log_path, record)

        dqn.save(latest_ckpt)

        if ep_return > best_return:
            best_return = ep_return
            dqn.save(best_ckpt)

        if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
            dqn.save(os.path.join(out_dir, f"checkpoint_ep_{episode+1}.pt"))

        print(
            f"Ep {episode:04d} | Return {ep_return:10.4f} | Steps {ep_steps:3d} "
            f"| Invalid {invalid_count:3d} | Eps {dqn.epsilon():.4f} | Loss {avg_loss}"
        )

    summary = summarize_run(episode_log_path)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    derived = derive_run_metrics(
        market_log_path=market_log_path if log_market else None,
        agent_log_path=agent_log_path if log_agents else None,
    )
    with open(os.path.join(out_dir, "derived_summary.json"), "w", encoding="utf-8") as f:
        json.dump(derived, f, indent=2)

    print("Run summary:", summary)
    print("Derived summary:", derived)

    return {
        "out_dir": out_dir,
        "episode_log_path": episode_log_path,
        "market_log_path": market_log_path if log_market else None,
        "agent_log_path": agent_log_path if log_agents else None,
        "summary": summary,
        "derived_summary": derived,
        "config": config_record,
    }


def _run_grid(grid: List[dict], base_kwargs: dict) -> List[dict]:
    results = []
    for params in grid:
        kwargs = dict(base_kwargs)
        kwargs.update(params)
        result = train(**kwargs)
        results.append({
            "run_tag": kwargs["run_tag"],
            **result["summary"],
            **result["derived_summary"],
            "out_dir": result["out_dir"],
        })
    return results


def run_rl_parameter_doe(output_root="runs_rl_param"):
    """
    (c) RL parameter sensitivity.
    """
    base_kwargs = dict(
        ticker_a="AAPL",
        ticker_b="XOM",
        sim_time=200,
        max_decision_events=100,
        num_background_agents=25,
        lam_bg=0.10,
        lam_rl=0.05,
        q_max=10,
        pv_var=1.0,
        zi_shade=(0.05, 0.5),
        initial_cash=100_000.0,
        gamma=0.99,
        lr=5e-4,
        batch_size=64,
        target_update_freq=1000,
        learning_starts=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=5000,
        grad_clip=10.0,
        replay_capacity=100_000,
        episodes=2000,
        checkpoint_every=10,
        seed=42,
        fundamental_mode="historical",
        bg_latency=0,
        rl_latency=0,
        use_market_makers=True,
        output_root=output_root,
    )

    grid = [
        {"run_tag": "A_pen1_tf4", "lambda_invalid": 1.0, "train_freq": 4},
        {"run_tag": "B_pen2_tf4", "lambda_invalid": 2.0, "train_freq": 4},
        {"run_tag": "C_pen1_tf1", "lambda_invalid": 1.0, "train_freq": 1},
        {"run_tag": "D_pen2_tf1", "lambda_invalid": 2.0, "train_freq": 1},
    ]
    return _run_grid(grid, base_kwargs)


def run_latency_doe(output_root="runs_latency"):
    """
    (b) latency vs no latency, keeping historical fundamental.
    """
    base_kwargs = dict(
        ticker_a="AAPL",
        ticker_b="XOM",
        sim_time=200,
        max_decision_events=100,
        num_background_agents=25,
        lam_bg=0.10,
        lam_rl=0.05,
        q_max=10,
        pv_var=1.0,
        zi_shade=(0.05, 0.5),
        initial_cash=100_000.0,
        gamma=0.99,
        lr=5e-4,
        batch_size=64,
        target_update_freq=1000,
        learning_starts=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=5000,
        grad_clip=10.0,
        replay_capacity=100_000,
        episodes=2000,
        checkpoint_every=10,
        seed=42,
        fundamental_mode="historical",
        use_market_makers=True,
        output_root=output_root,
    )

    grid = [
        {"run_tag": "hist_lat0_pen1_tf1", "lambda_invalid": 1.0, "train_freq": 1, "bg_latency": 0, "rl_latency": 0},
        {"run_tag": "hist_lat1_pen1_tf1", "lambda_invalid": 1.0, "train_freq": 1, "bg_latency": 1, "rl_latency": 1},
        {"run_tag": "hist_lat0_pen2_tf1", "lambda_invalid": 2.0, "train_freq": 1, "bg_latency": 0, "rl_latency": 0},
        {"run_tag": "hist_lat1_pen2_tf1", "lambda_invalid": 2.0, "train_freq": 1, "bg_latency": 1, "rl_latency": 1},
    ]
    return _run_grid(grid, base_kwargs)


def run_fundamental_doe(output_root="runs_fundamental"):
    """
    (a) synthetic vs historical fundamental.
    Uses a fixed RL setting.
    """
    base_kwargs = dict(
        ticker_a="AAPL",
        ticker_b="XOM",
        sim_time=200,
        max_decision_events=100,
        num_background_agents=25,
        lam_bg=0.10,
        lam_rl=0.05,
        q_max=10,
        pv_var=1.0,
        zi_shade=(0.05, 0.5),
        initial_cash=100_000.0,
        gamma=0.99,
        lr=5e-4,
        batch_size=64,
        target_update_freq=1000,
        learning_starts=500,
        train_freq=1,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=5000,
        grad_clip=10.0,
        replay_capacity=100_000,
        episodes=2000,
        checkpoint_every=10,
        seed=42,
        lambda_invalid=1.0,
        bg_latency=0,
        rl_latency=0,
        use_market_makers=True,
        output_root=output_root,
    )

    grid = [
        {"run_tag": "hist_pen1_tf1", "fundamental_mode": "historical"},
        {"run_tag": "synthetic_pen1_tf1", "fundamental_mode": "synthetic"},
    ]
    return _run_grid(grid, base_kwargs)


def run_fundamental_latency_doe(output_root="runs_fundamental_latency"):
    """
    (d) combination of (a) and (b): historical/synthetic x latency/no-latency
    """
    base_kwargs = dict(
        ticker_a="AAPL",
        ticker_b="XOM",
        sim_time=200,
        max_decision_events=100,
        num_background_agents=25,
        lam_bg=0.10,
        lam_rl=0.05,
        q_max=10,
        pv_var=1.0,
        zi_shade=(0.05, 0.5),
        initial_cash=100_000.0,
        gamma=0.99,
        lr=5e-4,
        batch_size=64,
        target_update_freq=1000,
        learning_starts=500,
        train_freq=1,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=5000,
        grad_clip=10.0,
        replay_capacity=100_000,
        episodes=2000,
        checkpoint_every=10,
        seed=42,
        lambda_invalid=1.0,
        use_market_makers=True,
        output_root=output_root,
    )

    grid = [
        {"run_tag": "hist_lat0", "fundamental_mode": "historical", "bg_latency": 0, "rl_latency": 0},
        {"run_tag": "hist_lat1", "fundamental_mode": "historical", "bg_latency": 1, "rl_latency": 1},
        {"run_tag": "synthetic_lat0", "fundamental_mode": "synthetic", "bg_latency": 0, "rl_latency": 0},
        {"run_tag": "synthetic_lat1", "fundamental_mode": "synthetic", "bg_latency": 1, "rl_latency": 1},
    ]
    return _run_grid(grid, base_kwargs)


if __name__ == "__main__":
    train()