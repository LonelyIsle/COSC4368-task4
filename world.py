# world.py
# PD-World: 2-agent block transport environment
# State format: (i, j, i', j', x, x', a, b, c, d, e, f)
#   (i, j)   : Female agent position (row, col) in 1..GRID_SIZE
#   (i', j') : Male agent position (row, col)
#   x, x'    : Carry flags (0/1) for F and M
#   a, b, c  : Blocks at dropoffs (1,1), (1,5), (3,3)
#   d, e     : Blocks at the 2 pickup "slots" (mapped to current PICKUP_LOCATIONS[0/1])
#   f        : Blocks at dropoff (5,5)

import random
from typing import Tuple, Set, Dict, List

# ===========================================================
# World configuration
# ===========================================================

GRID_SIZE = 5  # valid rows/cols are 1..5 inclusive

# These coordinates can change during Exp 4 via set_pickups()
PICKUP_LOCATIONS: List[Tuple[int, int]] = [(3, 5), (4, 2)]

# Dropoffs are fixed
DROPOFF_LOCATIONS: List[Tuple[int, int]] = [(1, 1), (1, 5), (3, 3), (5, 5)]

# Capacities and initial counts
DROPOFF_CAPACITY = 5
INITIAL_PICKUP_BLOCKS = 10  # for both pickup slots at start

# Actions
ACTIONS = ['north', 'south', 'east', 'west', 'pickup', 'dropoff']

# Rewards
REWARD_MOVE = -1
REWARD_PICKUP = 13
REWARD_DROPOFF = 13


# ===========================================================
# State helpers
# ===========================================================

def get_initial_state() -> Tuple:
    """
    Initial state:
      F at (1,3), M at (5,3), both not carrying;
      dropoffs a,b,c,f start at 0;
      pickup-slot counts d,e start at 10 each.
    """
    # (i, j,  i', j',  x, x',  a, b, c,  d,  e,  f)
    return (1, 3,   5, 3,   0, 0,   0, 0, 0,  10, 10,  0)


def is_terminal_state(state: Tuple) -> bool:
    """
    Terminal when all dropoffs hit capacity: a=b=c=f=5.
    """
    _, _, _, _, _, _, a, b, c, _, _, f = state
    return (a == DROPOFF_CAPACITY
            and b == DROPOFF_CAPACITY
            and c == DROPOFF_CAPACITY
            and f == DROPOFF_CAPACITY)


def get_agent_position(state: Tuple, agent: str) -> Tuple[int, int]:
    """
    Return (row, col) for agent 'F' or 'M'.
    """
    i, j, i_p, j_p, *_ = state
    return (i, j) if agent == 'F' else (i_p, j_p)


def get_agent_carrying(state: Tuple, agent: str) -> int:
    """
    Return carry flag (0/1) for agent 'F' or 'M'.
    """
    _, _, _, _, x, x_p, *_ = state
    return x if agent == 'F' else x_p


def get_block_counts(state: Tuple) -> Dict[str, int]:
    """
    Human-readable dict of block counts at sites.
    """
    *_, a, b, c, d, e, f = state
    return {
        'dropoff_1_1': a,
        'dropoff_1_5': b,
        'dropoff_3_3': c,
        'pickup_slot0': d,  # mapped to PICKUP_LOCATIONS[0]
        'pickup_slot1': e,  # mapped to PICKUP_LOCATIONS[1]
        'dropoff_5_5': f,
    }


# ===========================================================
# Action applicability
# ===========================================================

def aplop(state: Tuple, agent: str) -> Set[str]:
    """
    Applicable operators for agent in current state.
    Enforces:
      - 1..GRID_SIZE boundaries
      - no occupying same cell
      - pickup only at pickup cells with stock and not carrying
      - dropoff only at dropoff cells with capacity and carrying
    """
    i, j, i_p, j_p, x, x_p, a, b, c, d, e, f = state

    if agent == 'F':
        r, col = i, j
        carrying = x
        other_r, other_c = i_p, j_p
    else:
        r, col = i_p, j_p
        carrying = x_p
        other_r, other_c = i, j

    ops: Set[str] = set()

    # Move north
    if r > 1 and not (r - 1 == other_r and col == other_c):
        ops.add('north')
    # Move south
    if r < GRID_SIZE and not (r + 1 == other_r and col == other_c):
        ops.add('south')
    # Move east
    if col < GRID_SIZE and not (r == other_r and col + 1 == other_c):
        ops.add('east')
    # Move west
    if col > 1 and not (r == other_r and col - 1 == other_c):
        ops.add('west')

    # Pickup (dynamic positions)
    if carrying == 0:
        if (r, col) == tuple(PICKUP_LOCATIONS[0]) and d > 0:
            ops.add('pickup')
        elif (r, col) == tuple(PICKUP_LOCATIONS[1]) and e > 0:
            ops.add('pickup')

    # Dropoff (fixed positions)
    if carrying == 1:
        if (r, col) == (1, 1) and a < DROPOFF_CAPACITY:
            ops.add('dropoff')
        elif (r, col) == (1, 5) and b < DROPOFF_CAPACITY:
            ops.add('dropoff')
        elif (r, col) == (3, 3) and c < DROPOFF_CAPACITY:
            ops.add('dropoff')
        elif (r, col) == (5, 5) and f < DROPOFF_CAPACITY:
            ops.add('dropoff')

    return ops


# ===========================================================
# Transition
# ===========================================================

def apply(state: Tuple, action: str, agent: str) -> Tuple[Tuple, int]:
    """
    Apply action for the given agent. Assumes action is applicable.
    Returns (new_state, reward).
    """
    i, j, i_p, j_p, x, x_p, a, b, c, d, e, f = state

    # copy locals
    new_i, new_j = i, j
    new_i_p, new_j_p = i_p, j_p
    new_x, new_x_p = x, x_p
    new_a, new_b, new_c, new_d, new_e, new_f = a, b, c, d, e, f

    # Current agent snapshot
    if agent == 'F':
        r, col = i, j
        carrying = x
    else:
        r, col = i_p, j_p
        carrying = x_p

    reward = 0

    if action == 'north':
        if agent == 'F':
            new_i = r - 1
        else:
            new_i_p = r - 1
        reward = REWARD_MOVE

    elif action == 'south':
        if agent == 'F':
            new_i = r + 1
        else:
            new_i_p = r + 1
        reward = REWARD_MOVE

    elif action == 'east':
        if agent == 'F':
            new_j = col + 1
        else:
            new_j_p = col + 1
        reward = REWARD_MOVE

    elif action == 'west':
        if agent == 'F':
            new_j = col - 1
        else:
            new_j_p = col - 1
        reward = REWARD_MOVE

    elif action == 'pickup':
        if agent == 'F':
            new_x = 1
        else:
            new_x_p = 1

        # Decrement whichever pickup slot matches current coordinate
        if (r, col) == tuple(PICKUP_LOCATIONS[0]):
            new_d = d - 1
        elif (r, col) == tuple(PICKUP_LOCATIONS[1]):
            new_e = e - 1

        reward = REWARD_PICKUP

    elif action == 'dropoff':
        if agent == 'F':
            new_x = 0
        else:
            new_x_p = 0

        if (r, col) == (1, 1):
            new_a = a + 1
        elif (r, col) == (1, 5):
            new_b = b + 1
        elif (r, col) == (3, 3):
            new_c = c + 1
        elif (r, col) == (5, 5):
            new_f = f + 1

        reward = REWARD_DROPOFF

    new_state = (
        new_i, new_j, new_i_p, new_j_p,
        new_x, new_x_p, new_a, new_b, new_c,
        new_d, new_e, new_f
    )
    return new_state, reward


# ===========================================================
# Adapters for experiment runners
# ===========================================================

def reset(seed: int = 0):
    """Reset world and return the initial state. Seed accepted for compatibility."""
    random.seed(seed)
    return get_initial_state()


def applicable_actions(state: Tuple, agent: str):
    """Adapter name expected by runners; wraps aplop()."""
    return aplop(state, agent)


def step(state: Tuple, action: str, agent: str):
    """
    One environment step for agent. Returns (next_state, reward, done).
    """
    next_state, reward = apply(state, action, agent)
    done = is_terminal_state(next_state)
    return next_state, reward, done


def set_pickups(pickups):
    """
    Experiment 4 hook: change pickup coordinates WITHOUT touching d/e counts in the state.
    The two slot counts (d,e) always map to PICKUP_LOCATIONS[0] and [1], respectively.
    """
    global PICKUP_LOCATIONS
    if not isinstance(pickups, (list, tuple)) or len(pickups) != 2:
        raise ValueError("set_pickups expects exactly two pickup coordinates, e.g., [(1,2),(4,5)]")
    PICKUP_LOCATIONS = [tuple(pickups[0]), tuple(pickups[1])]