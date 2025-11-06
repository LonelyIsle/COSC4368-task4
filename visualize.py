# visualize.py
# Matplotlib-only helpers for world snapshots, episode traces, distance, and Q arrows.
# Works in headless terminals by default: plots are saved to PNGs if show=False.

from typing import Dict, Iterable, List, Tuple, Callable, Any, Sequence
import os
import math
import matplotlib.pyplot as plt

GridPos = Tuple[int, int]
State = Tuple  # (i, j, i', j', x, x', a, b, c, d, e, f)

# --- You can tweak these defaults if needed ---
DEFAULT_GRID = (5, 5)

# ---------- internal helpers ----------
def _maybe_save_or_show(fig, show: bool, outdir: str | None, name: str):
    if show:
        plt.show(block=True)
    else:
        os.makedirs(outdir or "plots", exist_ok=True)
        path = os.path.join(outdir or "plots", f"{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[saved] {path}")

def extract_positions_from_state(state: State) -> Tuple[GridPos, GridPos]:
    # Matches your state: (i, j, i', j', ...)
    return (int(state[0]), int(state[1])), (int(state[2]), int(state[3]))

def _draw_grid(ax, width: int, height: int):
    ax.set_xlim(0.5, width + 0.5)
    ax.set_ylim(0.5, height + 0.5)
    ax.set_xticks(range(1, width + 1))
    ax.set_yticks(range(1, height + 1))
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_xlabel('col')
    ax.set_ylabel('row')

# ---------- world snapshots / traces ----------
def plot_world(
    state: State,
    grid_size: Tuple[int, int] = DEFAULT_GRID,
    pickup_cells: Iterable[GridPos] = ((3, 5), (4, 2)),
    dropoff_cells: Iterable[GridPos] = ((1, 1), (1, 5), (3, 3), (5, 5)),
    position_extractor: Callable[[State], Tuple[GridPos, GridPos]] = extract_positions_from_state,
    title: str = 'PD-World snapshot',
    show: bool = False,
    outdir: str | None = None,
    name: str = "world"
):
    width, height = grid_size
    f_pos, m_pos = position_extractor(state)

    fig, ax = plt.subplots()
    _draw_grid(ax, width, height)

    for (x, y) in pickup_cells:
        ax.scatter([x], [y], marker='s', s=200, label='pickup')
    for (x, y) in dropoff_cells:
        ax.scatter([x], [y], marker='D', s=200, label='dropoff')

    ax.scatter([f_pos[1]], [f_pos[0]], marker='o', s=200, label='F')  # (row,col) -> (y,x)
    ax.scatter([m_pos[1]], [m_pos[0]], marker='^', s=200, label='M')

    ax.legend(loc='upper right')
    ax.set_title(title)
    _maybe_save_or_show(fig, show, outdir, name)

def plot_episode_trace(
    states: List[State],
    grid_size: Tuple[int, int] = DEFAULT_GRID,
    pickup_cells: Iterable[GridPos] = ((3, 5), (4, 2)),
    dropoff_cells: Iterable[GridPos] = ((1, 1), (1, 5), (3, 3), (5, 5)),
    position_extractor: Callable[[State], Tuple[GridPos, GridPos]] = extract_positions_from_state,
    title: str = 'Episode trace (agent paths)',
    show: bool = False,
    outdir: str | None = None,
    name: str = "trace"
):
    width, height = grid_size
    fig, ax = plt.subplots()
    _draw_grid(ax, width, height)

    f_path: List[GridPos] = []
    m_path: List[GridPos] = []
    for s in states:
        f_pos, m_pos = position_extractor(s)
        f_path.append(f_pos); m_path.append(m_pos)

    for (x, y) in pickup_cells:
        ax.scatter([x], [y], marker='s', s=120)
    for (x, y) in dropoff_cells:
        ax.scatter([x], [y], marker='D', s=120)

    # paths: convert (row,col) -> (x=col, y=row)
    if len(f_path) >= 2:
        xs = [p[1] for p in f_path]; ys = [p[0] for p in f_path]
        ax.plot(xs, ys, linewidth=2, label='F path')
    if len(m_path) >= 2:
        xs = [p[1] for p in m_path]; ys = [p[0] for p in m_path]
        ax.plot(xs, ys, linewidth=2, linestyle='--', label='M path')

    if f_path:
        ax.scatter([f_path[0][1]], [f_path[0][0]], s=100, label='F start')
        ax.scatter([f_path[-1][1]], [f_path[-1][0]], s=100, marker='x', label='F end')
    if m_path:
        ax.scatter([m_path[0][1]], [m_path[0][0]], s=100, label='M start')
        ax.scatter([m_path[-1][1]], [m_path[-1][0]], s=100, marker='x', label='M end')

    ax.legend(loc='upper right')
    ax.set_title(title)
    _maybe_save_or_show(fig, show, outdir, name)

# ---------- learning curves (per EPISODE) ----------
def plot_reward_per_episode(episodes, show: bool = False, outdir: str | None = None, name: str = "reward_per_episode"):
    fig, ax = plt.subplots()
    rewards = [e.total_reward for e in episodes]
    ax.plot(range(1, len(rewards) + 1), rewards)
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward")
    ax.set_title("Reward per episode")
    _maybe_save_or_show(fig, show, outdir, name)

def plot_steps_per_episode(episodes, show: bool = False, outdir: str | None = None, name: str = "steps_per_episode"):
    fig, ax = plt.subplots()
    steps = [e.steps for e in episodes]
    ax.plot(range(1, len(steps) + 1), steps)
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
    ax.set_title("Steps per episode")
    _maybe_save_or_show(fig, show, outdir, name)

def plot_avg_manhattan_per_episode(episodes, show: bool = False, outdir: str | None = None, name: str = "avg_manhattan_per_episode"):
    fig, ax = plt.subplots()
    dists = [e.avg_manhattan for e in episodes]
    ax.plot(range(1, len(dists) + 1), dists)
    ax.set_xlabel("Episode"); ax.set_ylabel("Avg Manhattan distance")
    ax.set_title("Average Manhattan distance per episode")
    _maybe_save_or_show(fig, show, outdir, name)

# ---------- Q-table arrows ----------
def _arrow_for_action(action: str) -> Tuple[float, float]:
    if action == 'north':  return (0, -0.3)
    if action == 'south':  return (0,  0.3)
    if action == 'west':   return (-0.3, 0)
    if action == 'east':   return (0.3,  0)
    return (0.0, 0.0)  # pickup/dropoff/none

def plot_q_arrows(
    q_table: Dict[Tuple[Any, str], float],
    grid_size: Tuple[int, int] = DEFAULT_GRID,
    for_agent: str = 'F',
    position_extractor: Callable[[State], Tuple[GridPos, GridPos]] = extract_positions_from_state,
    title: str = 'Greedy action by cell from Q-table',
    show_values: bool = True,
    show: bool = False,
    outdir: str | None = None,
    name: str = "q_arrows"
):
    # Aggregate across all states that place the chosen agent at (row,col)
    width, height = grid_size
    fig, ax = plt.subplots()
    _draw_grid(ax, width, height)

    # collect best (action, q) per cell
    best_per_cell: Dict[GridPos, Tuple[str, float]] = {}
    for (state, action), q in q_table.items():
        f_pos, m_pos = position_extractor(state)
        row, col = (f_pos if for_agent == 'F' else m_pos)
        cell = (row, col)
        if cell not in best_per_cell or q > best_per_cell[cell][1]:
            best_per_cell[cell] = (action, q)

    for (row, col), (action, q) in best_per_cell.items():
        dx, dy = _arrow_for_action(action)
        # convert to plotting coords: (x=col, y=row); dy sign is handled by inverted axis
        ax.arrow(col, row, dx, dy, head_width=0.12, length_includes_head=True)
        if show_values and math.isfinite(q):
            ax.text(col, row, f"{q:.2f}", ha='center', va='center')

    ax.set_title(title + f" (agent={for_agent})")
    _maybe_save_or_show(fig, show, outdir, f"{name}_{for_agent}")

# ---------- wrapper ----------
def visualize_run_package(
    run,
    grid_size: Tuple[int, int] = DEFAULT_GRID,
    initial_state: State | None = None,
    pickup_cells: Iterable[GridPos] = (),
    dropoff_cells: Iterable[GridPos] = (),
    show: bool = False,
    outdir: str | None = None,
    prefix: str = "run"
):
    # learning curves
    if run.episodes:
        plot_reward_per_episode(run.episodes, show, outdir, f"{prefix}_reward")
        plot_steps_per_episode(run.episodes, show, outdir, f"{prefix}_steps")
        plot_avg_manhattan_per_episode(run.episodes, show, outdir, f"{prefix}_avg_manhattan")

    # world snapshot if provided
    if initial_state is not None:
        plot_world(
            initial_state,
            grid_size=grid_size,
            pickup_cells=pickup_cells or ((3, 5), (4, 2)),
            dropoff_cells=dropoff_cells or ((1, 1), (1, 5), (3, 3), (5, 5)),
            title="Initial World State",
            show=show,
            outdir=outdir,
            name=f"{prefix}_world",
        )

    # Q arrows for both agents
    plot_q_arrows(run.Q, grid_size=grid_size, for_agent='F', title="Greedy Policy Arrows", show=show, outdir=outdir, name=f"{prefix}_q")
    plot_q_arrows(run.Q, grid_size=grid_size, for_agent='M', title="Greedy Policy Arrows", show=show, outdir=outdir, name=f"{prefix}_q")