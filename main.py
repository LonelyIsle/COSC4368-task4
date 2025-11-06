# main.py
# Entry point for running Task 4 experiments.
# You can run: python main.py --exp 1b | 1c | 1d | 2 | 3q | 3sarsa | 4q | 4sarsa

import argparse
from experiments import (
    two_runs_exp1_prandom,
    two_runs_exp1_pgreedy,
    two_runs_exp1_pexploit,
    two_runs_exp2,
    two_runs_exp3,
    two_runs_exp4,
    summarize_run,
)
from visualize import (
    plot_world,
    plot_episode_trace,
    plot_distance_over_time,
    plot_q_arrows,
)
import world as W

def run_and_summarize(experiment_fn, seedA, seedB, label):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    rA, rB = experiment_fn(seedA, seedB)
    print("Run A Summary:")
    print(summarize_run(rA))
    print("\nRun B Summary:")
    print(summarize_run(rB))
    return rA, rB

def visualize_results(run, grid_size=(5, 5)):
    # Visualize world state & learned Q
    print("\nGenerating visuals for one sample run...")

    if run.episodes:
        # Mock a sample trace (you can log real episode_states if you want more detail)
        state = W.get_initial_state()
        plot_world(
            state,
            grid_size=grid_size,
            pickup_cells=W.PICKUP_LOCATIONS,
            dropoff_cells=W.DROPOFF_LOCATIONS,
            title="Initial World State"
        )

    # Q-table visualization
    plot_q_arrows(
        run.Q,
        grid_size=grid_size,
        for_agent='F',
        title="Greedy Policy Arrows (F)"
    )
    plot_q_arrows(
        run.Q,
        grid_size=grid_size,
        for_agent='M',
        title="Greedy Policy Arrows (M)"
    )
    print("Visuals displayed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 4 experiments.")
    parser.add_argument("--exp", required=True,
                        help="1b | 1c | 1d | 2 | 3q | 3sarsa | 4q | 4sarsa")
    parser.add_argument("--seedA", type=int, default=7)
    parser.add_argument("--seedB", type=int, default=19)
    args = parser.parse_args()

    exp_map = {
        "1b": (two_runs_exp1_prandom, "Experiment 1(b) - Q-Learning PRANDOM"),
        "1c": (two_runs_exp1_pgreedy, "Experiment 1(c) - Q-Learning PGREEDY"),
        "1d": (two_runs_exp1_pexploit, "Experiment 1(d) - Q-Learning PEXPLOIT"),
        "2": (two_runs_exp2, "Experiment 2 - SARSA (PGREEDY)"),
        "3q": (lambda a, b: two_runs_exp3(base="q", seed_a=a, seed_b=b), "Experiment 3 - Q-Learning (lower α, γ)"),
        "3sarsa": (lambda a, b: two_runs_exp3(base="sarsa", seed_a=a, seed_b=b), "Experiment 3 - SARSA (lower α, γ)"),
        "4q": (lambda a, b: two_runs_exp4(base="q", seed_a=a, seed_b=b), "Experiment 4 - Q-Learning Pickup Switch"),
        "4sarsa": (lambda a, b: two_runs_exp4(base="sarsa", seed_a=a, seed_b=b), "Experiment 4 - SARSA Pickup Switch"),
    }

    if args.exp not in exp_map:
        print("Unknown experiment ID. Use one of: 1b | 1c | 1d | 2 | 3q | 3sarsa | 4q | 4sarsa")
        exit(1)

    exp_fn, label = exp_map[args.exp]
    resultA, resultB = run_and_summarize(exp_fn, args.seedA, args.seedB, label)

    # Optional: visualize one run
    visualize_results(resultA)