# tests/random_baseline_per_stage.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import gfootball.env as football_env
from tqdm import tqdm

@dataclass
class TrainingStage:
    name: str
    env_name: str
    representation: str
    left_players: int
    right_players: int
    target_reward: float
    max_timesteps: int
    description: str = ""

TRAINING_STAGES: List[TrainingStage] =[
    TrainingStage("stage_1_basic", "academy_empty_goal_close", "simple115v2", 1, 0, 0.75, 1_000_000, "1 attacker, no opponents: finishes into an empty goal from close range."),
    TrainingStage("stage_2_basic", "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 0.75, 200_000_000, "1 attacker versus a goalkeeper: dribbles towards goal and finishes under light pressure."),
    TrainingStage("stage_3_basic", "academy_pass_and_shoot_with_keeper", "simple115v2", 1, 0, 0.75, 5_000_000, "1 attacker facing a goalkeeper and nearby defender: focuses on control, positioning, and finishing."),
    TrainingStage("stage_4_1v1", "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 0.75, 10_000_000, "3 attackers versus 1 defender and a goalkeeper: encourages passing combinations and shot creation."),
    TrainingStage("stage_5_3v0", "academy_single_goal_versus_lazy", "simple115v2", 1, 0, 1.0, 50_000_000, "3 vs 0 on a full field against static opponents: focuses on offensive buildup and team coordination."),
    TrainingStage("stage_6_transition", "11_vs_11_easy_stochastic", "simple115v2", 3, 3, 1.0, 100_000_000, "Small-sided (3-player) team in 11v11 environment with easy opponents: transition toward full gameplay."),
    TrainingStage("stage_7_midgame", "11_vs_11_easy_stochastic", "simple115v2", 5, 5, 1.0, 500_000_000, "3 vs 3 within a full 11v11 match (easy mode): focuses on spacing, positioning, and transitions."),
    TrainingStage("stage_8_fullgame", "11_vs_11_stochastic", "simple115v2", 5, 5, 1.0, 1_000_000_000, "Full 11v11 stochastic match: standard difficulty with dynamic and realistic gameplay.")
]

def _unpack_step(step_tuple) -> Tuple[np.ndarray, float, bool, bool, dict]:
    if len(step_tuple) == 5:
        obs, rew, terminated, truncated, info = step_tuple
        done = bool(terminated) or bool(truncated)
        return obs, float(rew if np.isscalar(rew) else np.sum(rew)), bool(terminated), bool(truncated), info
    elif len(step_tuple) == 4:
        obs, rew, done, info = step_tuple
        return obs, float(rew if np.isscalar(rew) else np.sum(rew)), bool(done), False, info
    else:
        raise ValueError(f"Unexpected step() format: len={len(step_tuple)}")

def _sample_actions(env, n_agents: int, rng: np.random.Generator):
    act_space = env.action_space
    if hasattr(act_space, "n"):
        return [int(rng.integers(0, act_space.n)) for _ in range(n_agents)]
    if hasattr(act_space, "spaces"):
        first = act_space.spaces[0]
        n = first.n if hasattr(first, "n") else 19
        return [int(rng.integers(0, n)) for _ in range(n_agents)]
    return [int(rng.integers(0, 19)) for _ in range(n_agents)]

def _infer_n_controlled_agents(env_cfg: dict) -> int:
    return int(env_cfg.get("number_of_left_players_agent_controls", 0)) + int(env_cfg.get("number_of_right_players_agent_controls", 0))

def evaluate_random_baseline(
    stage: TrainingStage,
    episodes: int = 200,
    max_steps: int = 400,
    rewards: str = "scoring,checkpoints",
    seed: int = 1234,
    end_on_score: bool = True,
):
    rng = np.random.default_rng(seed)

    env_cfg = dict(
        env_name=stage.env_name,
        representation=stage.representation,
        number_of_left_players_agent_controls=stage.left_players,
        number_of_right_players_agent_controls=stage.right_players,
        rewards=rewards,
        render=False,
        stacked=True,
    )
    env = football_env.create_environment(**env_cfg)
    n_agents = _infer_n_controlled_agents(env_cfg)
    if n_agents <= 0:
        print(f"[WARN] Stage '{stage.name}': n_agents=0 -> Random-Baseline macht wenig Sinn.")
        n_agents = 1

    returns = []
    goal_flags = []
    steps_to_goal = []

    # Progress bar for episodes within this stage
    episode_progress_bar = tqdm(
        range(episodes),
        desc=f"Stage {stage.name[:18]:<18}",
        unit="ep",
        leave=False,
        ncols=100,
    )

    for ep in episode_progress_bar:
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        ep_return = 0.0
        terminated = truncated = done = False
        step_count = 0

        while not done and step_count < max_steps:
            actions = _sample_actions(env, n_agents, rng)
            step_tuple = env.step(actions)
            obs, rew, term, trunc, info = _unpack_step(step_tuple)
            done = term or trunc
            ep_return += rew
            step_count += 1

        returns.append(ep_return)

        scored = ep_return >= 0.99
        goal_flags.append(scored)
        if scored:
            steps_to_goal.append(step_count)

        # Update progress bar postfix with current stats (e.g., every 10 episodes)
        if (ep + 1) % 10 == 0 or ep == episodes - 1:
            current_mean_ret = np.mean(returns) if returns else 0.0
            current_goal_rate = np.mean(goal_flags) if goal_flags else 0.0
            episode_progress_bar.set_postfix_str(
                f"Ret={current_mean_ret:.2f}, Goal%={current_goal_rate*100:.1f}%",
                refresh=True
            )

    env.close()

    ret = np.asarray(returns, dtype=float)
    gf = np.asarray(goal_flags, dtype=bool)
    stg = np.asarray(steps_to_goal, dtype=float) if steps_to_goal else np.array([])

    summary = {
        "stage": stage.name,
        "env_name": stage.env_name,
        "episodes": episodes,
        "mean_return": float(ret.mean()),
        "median_return": float(np.median(ret)),
        "p95_return": float(np.percentile(ret, 95)),
        "max_return": float(ret.max()),
        "goal_rate": float(gf.mean()),
        "avg_steps_to_goal": (float(stg.mean()) if stg.size else None),
        "rewards": rewards,
        "seed": seed,
    }
    return summary

def run_all_random_baselines(
    stages: List[TrainingStage],
    episodes: int = 200,
    rewards: str = "scoring,checkpoints",
    seed: int = 1234,
    max_steps: int = 400,
):
    print("\nStarting Random Baseline Evaluation...")
    print(f"Running {episodes} episodes per stage. This may take a while.\n")
    
    summaries = []
    # Loop over stages; progress bar is now *inside* evaluate_random_baseline
    for s in stages:
        summary = evaluate_random_baseline(
            s,
            episodes=episodes,
            max_steps=max_steps,
            rewards=rewards,
            seed=seed,
            end_on_score=True,
        )
        summaries.append(summary)

    # Print final summary table after all evaluations are done
    print("\n\n" + "="*40)
    print("=== FINAL RANDOM BASELINE SUMMARY ===")
    print("="*40)
    print(f"Episodes={episodes} (per stage), Rewards='{rewards}', Seed={seed}, MaxSteps={max_steps}\n")

    header = (
        "stage".ljust(18),
        "env".ljust(32),
        "mean".rjust(6),
        "med".rjust(6),
        "p95".rjust(6),
        "max".rjust(6),
        "goal%".rjust(7),
        "steps_goal".rjust(11),
    )
    print(" | ".join(header))
    print("-" * (len(" | ".join(header)) + 2))

    for sm in summaries:
        mean_ = f"{sm['mean_return']:.3f}"
        med_  = f"{sm['median_return']:.3f}"
        p95_  = f"{sm['p95_return']:.3f}"
        max_  = f"{sm['max_return']:.3f}"
        goal_ = f"{sm['goal_rate']*100:5.1f}%"
        stg   = f"{sm['avg_steps_to_goal']:.1f}" if sm['avg_steps_to_goal'] is not None else "   n/a"
        row = (
            sm["stage"].ljust(18)[:18],
            sm["env_name"].ljust(32)[:32],
            mean_.rjust(6),
            med_.rjust(6),
            p95_.rjust(6),
            max_.rjust(6),
            goal_.rjust(7),
            stg.rjust(11),
        )
        print(" | ".join(row))

    print("\nHinweis:")
    print("- Random-Baseline ist in GRF oft sehr niedrig; Tor-Rate ~0–10% je nach Stage ist normal.")
    print("- Theoretisches Max pro Episode (mit 'scoring,checkpoints'): ~2.0 (1.0 Tor + bis zu ~1.0 Checkpoints).")
    print("- Wenn du getrennt 'scoring' vs. 'scoring,checkpoints' vergleichen willst: run_all_random_baselines zweimal mit unterschiedlichem 'rewards'.")

if __name__ == "__main__":
    run_all_random_baselines(
        TRAINING_STAGES,
        episodes=100,
        rewards="scoring,checkpoints",
        seed=42,
        max_steps=400,
    )

# ========================================
# === FINAL RANDOM BASELINE SUMMARY ===
# ========================================
# Episodes=100 (per stage), Rewards='scoring,checkpoints', Seed=42, MaxSteps=400

# stage              | env                              |   mean |    med |    p95 |    max |   goal% |  steps_goal
# -------------------------------------------------------------------------------------------------------------------
# stage_1_basic      | academy_empty_goal_close         |  0.924 |  0.900 |  2.000 |  2.000 |   23.0% |        49.9
# stage_2_basic      | academy_run_to_score_with_keeper |  0.031 |  0.100 |  0.100 |  0.200 |    0.0% |         n/a
# stage_3_basic      | academy_pass_and_shoot_with_keep |  0.450 |  0.700 |  0.800 |  2.000 |    1.0% |        34.0
# stage_4_1v1        | academy_3_vs_1_with_keeper       |  1.050 |  0.800 |  1.705 |  6.000 |   32.0% |       110.0
# stage_5_3v0        | academy_single_goal_versus_lazy  |  0.194 |  0.000 |  0.700 |  2.000 |    1.0% |       201.0
# stage_6_transition | 11_vs_11_easy_stochastic         |  1.208 |  0.800 |  3.600 |  6.000 |   46.0% |       400.0
# stage_7_midgame    | 11_vs_11_easy_stochastic         |  1.186 |  0.850 |  5.105 |  5.900 |   44.0% |       400.0
# stage_8_fullgame   | 11_vs_11_stochastic              |  1.473 |  0.800 |  5.400 |  6.200 |   46.0% |       400.0

# Hinweis:
# - Random-Baseline ist in GRF oft sehr niedrig; Tor-Rate ~0–10% je nach Stage ist normal.
# - Theoretisches Max pro Episode (mit 'scoring,checkpoints'): ~2.0 (1.0 Tor + bis zu ~1.0 Checkpoints).
# - Wenn du getrennt 'scoring' vs. 'scoring,checkpoints' vergleichen willst: run_all_random_baselines zweimal mit unterschiedlichem 'rewards'.
