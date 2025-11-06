# experiments.py
# Runners for Task 4 Experiments (1–4), plus Q-learning & SARSA updates.
# Works with:
#   - world.py exposing: reset(seed), applicable_actions(state,agent), step(state,action,agent), set_pickups(pickups)
#   - policies.py exposing: prandom, pgreedy, pexploit

import random
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---- Bind to your project modules ----
import world as W
import policies as P

# ---- Types ----
State = Tuple
QTable = Dict[Tuple[Any, str], float]

# ---- Q helpers ----
def q_get(Q: QTable, s: State, a: str) -> float:
    return Q.get((s, a), 0.0)

def q_set(Q: QTable, s: State, a: str, v: float) -> None:
    Q[(s, a)] = v

def argmax_q(Q: QTable, state: State, actions: Sequence[str]) -> str:
    best_q = -math.inf
    best = []
    for a in actions:
        q = q_get(Q, state, a)
        if q > best_q:
            best_q = q; best = [a]
        elif q == best_q:
            best.append(a)
    return random.choice(best) if best else random.choice(list(actions))

# ---- Updates ----
def update_qlearning(Q: QTable, s: State, a: str, r: float, s_next: State, alpha: float, gamma: float,
                     next_actions: Sequence[str]) -> None:
    target = r + (gamma * max(q_get(Q, s_next, an) for an in next_actions)) if next_actions else r
    new_q = (1 - alpha) * q_get(Q, s, a) + alpha * target
    q_set(Q, s, a, new_q)

def update_sarsa(Q: QTable, s: State, a: str, r: float, s_next: State, a_next: Optional[str],
                 alpha: float, gamma: float) -> None:
    target = r + (gamma * q_get(Q, s_next, a_next) if a_next is not None else 0.0)
    new_q = (1 - alpha) * q_get(Q, s, a) + alpha * target
    q_set(Q, s, a, new_q)

# ---- Metrics ----
def manhattan_between_agents(s: State) -> int:
    # Your state: (i, j, i', j', x, x', a, b, c, d, e, f)
    return abs(s[0] - s[2]) + abs(s[1] - s[3])

@dataclass
class EpisodeLog:
    steps: int
    total_reward: float
    avg_manhattan: float

@dataclass
class RunResult:
    seed: int
    total_steps: int
    terminals_reached: int
    episodes: List[EpisodeLog]
    Q: QTable  # final Q

# ---- Core rollout ----
def rollout(
    seed: int,
    total_steps: int,
    warmup_steps: int,
    alpha: float,
    gamma: float,
    learner: str,                 # 'q' or 'sarsa'
    schedule_after_warmup: str,   # 'prandom' | 'pgreedy' | 'pexploit'
) -> RunResult:
    random.seed(seed)
    state = W.reset(seed)
    Q: QTable = {}
    current_agent = 'F'  # Female starts
    step_count = 0
    terminals = 0

    # episode buffers
    episode_states: List[State] = [state]
    episode_rewards = 0.0
    episode_steps = 0
    episode_dist_sum = 0
    episodes: List[EpisodeLog] = []

    def pick_policy():
        if step_count < warmup_steps:
            return P.prandom
        return {'prandom': P.prandom, 'pgreedy': P.pgreedy, 'pexploit': P.pexploit}.get(schedule_after_warmup, P.pgreedy)

    while step_count < total_steps:
        actions = W.applicable_actions(state, current_agent)
        if not actions:
            current_agent = 'M' if current_agent == 'F' else 'F'
            continue

        policy = pick_policy()
        action = policy(state, actions, Q)

        next_state, reward, done = W.step(state, action, current_agent)
        step_count += 1
        episode_steps += 1
        episode_rewards += reward
        episode_dist_sum += manhattan_between_agents(next_state)
        episode_states.append(next_state)

        if learner == 'q':
            next_actions = W.applicable_actions(next_state, 'M' if current_agent == 'F' else 'F')
            update_qlearning(Q, state, action, reward, next_state, alpha, gamma, next_actions)
        else:
            next_actions_curr = W.applicable_actions(next_state, 'M' if current_agent == 'F' else 'F')
            a_next = None
            if next_actions_curr:
                a_next = pick_policy()(next_state, next_actions_curr, Q)
            update_sarsa(Q, state, action, reward, next_state, a_next, alpha, gamma)

        state = next_state
        current_agent = 'M' if current_agent == 'F' else 'F'

        if done:
            terminals += 1
            avg_manh = episode_dist_sum / max(1, episode_steps)
            episodes.append(EpisodeLog(steps=episode_steps, total_reward=episode_rewards, avg_manhattan=avg_manh))
            # new episode, keep Q-table
            state = W.reset(seed + terminals)   # small perturbation per episode
            episode_states = [state]
            episode_rewards = 0.0
            episode_steps = 0
            episode_dist_sum = 0
            current_agent = 'F'

    return RunResult(seed=seed, total_steps=step_count, terminals_reached=terminals, episodes=episodes, Q=Q)

# ---- Experiments ----
def run_experiment1_variant(variant: str, seed: int) -> RunResult:
    # α=0.3, γ=0.5, 8000 steps, warmup 500; variant = 'prandom'| 'pgreedy' | 'pexploit'
    return rollout(seed=seed, total_steps=8000, warmup_steps=500,
                   alpha=0.3, gamma=0.5, learner='q', schedule_after_warmup=variant)

def run_experiment2(seed: int) -> RunResult:
    # SARSA; same as 1(c): post-warmup PGREEDY
    return rollout(seed=seed, total_steps=8000, warmup_steps=500,
                   alpha=0.3, gamma=0.5, learner='sarsa', schedule_after_warmup='pgreedy')

def run_experiment3(seed: int, base: str = 'q') -> RunResult:
    # like 1(c) if base='q' or like 2 if base='sarsa' but α=0.15, γ=0.45
    learner = 'q' if base == 'q' else 'sarsa'
    return rollout(seed=seed, total_steps=8000, warmup_steps=500,
                   alpha=0.15, gamma=0.45, learner=learner, schedule_after_warmup='pgreedy')

def run_experiment4(seed: int, base: str = 'q') -> RunResult:
    # Start like 1(c) or 2; after 3rd terminal, set_pickups([(1,2),(4,5)]); continue to 6th
    random.seed(seed)
    state = W.reset(seed)
    Q: QTable = {}
    current_agent = 'F'
    step_count = 0
    terminals = 0

    alpha, gamma = 0.3, 0.5
    warmup_steps = 500
    learner = 'q' if base == 'q' else 'sarsa'

    episode_states: List[State] = [state]
    episode_rewards = 0.0
    episode_steps = 0
    episode_dist_sum = 0
    episodes: List[EpisodeLog] = []

    def pick_policy():
        if step_count < warmup_steps:
            return P.prandom
        return P.pgreedy

    # generous cap to avoid infinite loops
    while terminals < 6 and step_count < 20000:
        actions = W.applicable_actions(state, current_agent)
        if not actions:
            current_agent = 'M' if current_agent == 'F' else 'F'
            continue

        action = pick_policy()(state, actions, Q)
        next_state, reward, done = W.step(state, action, current_agent)
        step_count += 1
        episode_steps += 1
        episode_rewards += reward
        episode_dist_sum += manhattan_between_agents(next_state)
        episode_states.append(next_state)

        if learner == 'q':
            next_actions = W.applicable_actions(next_state, 'M' if current_agent == 'F' else 'F')
            update_qlearning(Q, state, action, reward, next_state, alpha, gamma, next_actions)
        else:
            next_actions_curr = W.applicable_actions(next_state, 'M' if current_agent == 'F' else 'F')
            a_next = None
            if next_actions_curr:
                a_next = pick_policy()(next_state, next_actions_curr, Q)
            update_sarsa(Q, state, action, reward, next_state, a_next, alpha, gamma)

        state = next_state
        current_agent = 'M' if current_agent == 'F' else 'F'

        if done:
            terminals += 1
            avg_manh = episode_dist_sum / max(1, episode_steps)
            episodes.append(EpisodeLog(steps=episode_steps, total_reward=episode_rewards, avg_manhattan=avg_manh))

            if terminals == 3:
                W.set_pickups([(1, 2), (4, 5)])  # change pickups, keep Q

            state = W.reset(seed + terminals)
            episode_states = [state]
            episode_rewards = 0.0
            episode_steps = 0
            episode_dist_sum = 0
            current_agent = 'F'

    return RunResult(seed=seed, total_steps=step_count, terminals_reached=terminals, episodes=episodes, Q=Q)

# ---- Two-run helpers ----
def two_runs(fn_runner: Callable[[int], RunResult], seed_a: int, seed_b: int):
    return fn_runner(seed_a), fn_runner(seed_b)

def two_runs_exp1_prandom(seed_a: int = 7, seed_b: int = 19):
    return two_runs(lambda s: run_experiment1_variant('prandom', s), seed_a, seed_b)

def two_runs_exp1_pgreedy(seed_a: int = 7, seed_b: int = 19):
    return two_runs(lambda s: run_experiment1_variant('pgreedy', s), seed_a, seed_b)

def two_runs_exp1_pexploit(seed_a: int = 7, seed_b: int = 19):
    return two_runs(lambda s: run_experiment1_variant('pexploit', s), seed_a, seed_b)

def two_runs_exp2(seed_a: int = 11, seed_b: int = 23):
    return two_runs(run_experiment2, seed_a, seed_b)

def two_runs_exp3(base: str = 'q', seed_a: int = 13, seed_b: int = 29):
    return two_runs(lambda s: run_experiment3(s, base=base), seed_a, seed_b)

def two_runs_exp4(base: str = 'q', seed_a: int = 17, seed_b: int = 31):
    return two_runs(lambda s: run_experiment4(s, base=base), seed_a, seed_b)

# ---- Pretty summary ----
def summarize_run(run: RunResult) -> str:
    lines = [
        f"Seed: {run.seed}",
        f"Total Steps: {run.total_steps}",
        f"Terminals: {run.terminals_reached}",
    ]
    if run.episodes:
        avg_reward = sum(e.total_reward for e in run.episodes) / len(run.episodes)
        avg_len = sum(e.steps for e in run.episodes) / len(run.episodes)
        avg_manh = sum(e.avg_manhattan for e in run.episodes) / len(run.episodes)
        lines += [
            f"Avg Reward/episode: {avg_reward:.2f}",
            f"Avg Steps/episode: {avg_len:.1f}",
            f"Avg Manhattan/step: {avg_manh:.2f}",
        ]
    return "\n".join(lines)

if __name__ == "__main__":
    # smoke test (comment out if you prefer)
    r1a, r1b = two_runs_exp1_pgreedy()
    print("=== Exp1(c) PGREEDY ===")
    print(summarize_run(r1a)); print(summarize_run(r1b))