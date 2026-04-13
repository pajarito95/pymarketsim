import json
import os
from datetime import datetime

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
    }
    return summary


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
    checkpoint_every=10,
    seed=42,
    run_tag=None,
    use_market_makers=False,
    lam_mm=0.10,
    mm_xi=0.5,
    mm_K=3,
    mm_omega=2.0,
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
    out_dir = os.path.join("runs", run_name)
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
        bg_latency=bg_latency,
        rl_latency=rl_latency,
        use_market_makers=use_market_makers,
        lam_mm=lam_mm,
        mm_xi=mm_xi,
        mm_K=mm_K,
        mm_omega=mm_omega,
        seed=seed,
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

    summary_path = os.path.join(out_dir, "episode_log.jsonl")
    latest_ckpt = os.path.join(out_dir, "latest.pt")
    best_ckpt = os.path.join(out_dir, "best.pt")

    best_return = -np.inf

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)

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
        append_jsonl(summary_path, record)

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

    summary = summarize_run(summary_path)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Run summary:", summary)
    return {
        "out_dir": out_dir,
        "summary_path": summary_path,
        "summary": summary,
        "config": config_record,
    }


def run_doe_no_latency():
    runs = [
        {"run_tag": "A_pen1_tf4", "lambda_invalid": 1.0, "train_freq": 4},
        {"run_tag": "B_pen2_tf4", "lambda_invalid": 2.0, "train_freq": 4},
        {"run_tag": "C_pen1_tf1", "lambda_invalid": 1.0, "train_freq": 1},
        {"run_tag": "D_pen2_tf1", "lambda_invalid": 2.0, "train_freq": 1},
    ]

    results = []
    for params in runs:
        result = train(
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
            **params,
        )
        results.append({
            "run_tag": params["run_tag"],
            **result["summary"],
            "out_dir": result["out_dir"],
        })

    return results


def run_doe_latency():
    runs = [
        {"run_tag": "A_pen1_tf4_lat1", "lambda_invalid": 1.0, "train_freq": 4, "bg_latency": 1, "rl_latency": 1},
        {"run_tag": "B_pen2_tf4_lat1", "lambda_invalid": 2.0, "train_freq": 4, "bg_latency": 1, "rl_latency": 1},
        {"run_tag": "C_pen1_tf1_lat1", "lambda_invalid": 1.0, "train_freq": 1, "bg_latency": 1, "rl_latency": 1},
        {"run_tag": "D_pen2_tf1_lat1", "lambda_invalid": 2.0, "train_freq": 1, "bg_latency": 1, "rl_latency": 1},
    ]

    results = []
    for params in runs:
        result = train(
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
            **params,
        )
        results.append({
            "run_tag": params["run_tag"],
            **result["summary"],
            "out_dir": result["out_dir"],
        })

    return results


if __name__ == "__main__":
    train()