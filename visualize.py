# visualize.py
# Matplotlib-only helpers for world snapshots, episode traces, distance, and Q arrows.
# IMPORTANT: One chart per function (no subplots), no seaborn, no custom colors.

from typing import Dict, Iterable, List, Tuple, Callable, Any
import math
import matplotlib.pyplot as plt

GridPos = Tuple[int, int]
State = Tuple  # (i, j, i', j', x, x', a, b, c, d, e, f)

# --- You can tweak these defaults if needed ---
DEFAULT_GRID = (5, 5)

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

def plot_world(
    state: State,
    grid_size: Tuple[int, int] = DEFAULT_GRID,
    pickup_cells: Iterable[GridPos] = ((3, 5), (4, 2)),
    dropoff_cells: Iterable[GridPos] = ((1, 1), (1, 5), (3, 3), (5, 5)),
    position_extractor: Callable[[State], Tuple[GridPos, GridPos]] = extract_positions_from_state,
    title: str = 'PD-World snapshot'
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
    plt.show()

def plot_episode_trace(
    states: List[State],
    grid_size: Tuple[int, int] = DEFAULT_GRID,
    pickup_cells: Iterable[GridPos] = ((3, 5), (4, 2)),
    dropoff_cells: Iterable[GridPos] = ((1, 1), (1, 5), (3, 3), (5, 5)),
    position_extractor: Callable[[State], Tuple[GridPos, GridPos]] = extract_positions_from_state,
    title: str = 'Episode trace (agent paths)'
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
    plt.show()

def plot_distance_over_time(
    states: List[State],
    position_extractor: Callable[[State], Tuple[GridPos, GridPos]] = extract_positions_from_state,
    title: str = 'Agent Manhattan distance over time'
):
    distances: List[int] = []
    for s in states:
        (fr, fc), (mr, mc) = position_extractor(s)
        distances.append(abs(fr - mr) + abs(fc - mc))

    fig, ax = plt.subplots()
    ax.plot(range(len(distances)), distances)
    ax.set_xlabel('Step')
    ax.set_ylabel('Manhattan distance')
    ax.set_title(title)
    plt.show()

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
    show_values: bool = True
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
    plt.show()