# main.py
# Entry point for running Task 4 experiments.
# You can run: python3 main.py --exp 1b | 1c | 1d | 2 | 3q | 3sarsa | 4q | 4sarsa

import argparse
from experiments import (
    two_runs_exp1_prandom,
    two_runs_exp1_pgreedy,
    two_runs_exp1_pexploit,
    two_runs_exp2,
    two_runs_exp3,
    two_runs_exp4,
    summarize_run,
    sample_q_entries,
)
from visualize import visualize_run_package
import world as W

def run_and_summarize(experiment_fn, seedA, seedB, label):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    rA, rB = experiment_fn(seedA, seedB)
    print("Run A Summary:")
    print(summarize_run(rA))
    print("\nRun B Summary:")
    print(summarize_run(rB))

    # Show a tiny Q-table sample in the console for grading
    print("\nTop Q entries (Run A):")
    for ((s, a), q) in sample_q_entries(rA, top_n=8):
        print(a, f"{q:8.3f}", s)

    return rA, rB

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 4 experiments.")
    parser.add_argument("--exp", required=True,
                        help="1b | 1c | 1d | 2 | 3q | 3sarsa | 4q | 4sarsa")
    parser.add_argument("--seedA", type=int, default=7)
    parser.add_argument("--seedB", type=int, default=19)
    parser.add_argument("--show", action="store_true",
                        help="Show plots in windows instead of saving PNGs.")
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

    # ----- Visuals: save PNGs by default (works in headless terminals) -----
    outA = f"plots/{args.exp}_seed{resultA.seed}"
    outB = f"plots/{args.exp}_seed{resultB.seed}"

    initial_state = W.get_initial_state() if hasattr(W, "get_initial_state") else None
    pickups = getattr(W, "PICKUP_LOCATIONS", [])
    dropoffs = getattr(W, "DROPOFF_LOCATIONS", [])

    visualize_run_package(
        resultA,
        grid_size=(5, 5),
        initial_state=initial_state,
        pickup_cells=pickups,
        dropoff_cells=dropoffs,
        show=args.show,
        outdir=outA,
        prefix=f"{args.exp}_seed{resultA.seed}"
    )
    visualize_run_package(
        resultB,
        grid_size=(5, 5),
        initial_state=initial_state,
        pickup_cells=pickups,
        dropoff_cells=dropoffs,
        show=args.show,
        outdir=outB,
        prefix=f"{args.exp}_seed{resultB.seed}"
    )

    print(f"\nPNG plots saved under:\n - {outA}\n - {outB}\n(Use --show to open windows instead.)")