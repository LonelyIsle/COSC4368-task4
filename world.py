# everything here is related to the PD-World
# The grid size, Its dropoff and pickup locations, initial state, terminal state, agent positions,
# block counts at locations allowed to have blocks
# determine the rules for how an agent moves in the world

import random
import copy
from typing import Tuple, Set, Dict # for hints/better documentation

'''
===========================================================
WORLD STATE SETUP
===========================================================
'''

# Max of grid dimensions
GRID_SIZE = 5   #Valid positions are (1,1) to (5,5). The alop() function will enforce that boundary - 5x5

# Pickup and dropoff locations
PICKUP_LOCATIONS = [(3, 5), (4, 2)]
DROPOFF_LOCATIONS = [(1, 1), (1, 5), (3, 3), (5, 5)]

# Initial configuration for blocks and drop off site
INITIAL_PICKUP_BLOCKS = 10
DROPOFF_CAPACITY = 5

# Action definitions
ACTIONS = ['north', 'south', 'east', 'west', 'pickup', 'dropoff']

# Rewards
REWARD_MOVE = -1
REWARD_PICKUP = 13
REWARD_DROPOFF = 13

def get_initial_state() -> Tuple:
    """
    Returns the initial state of the PD-World.

    State format: (i, j, i', j', x, x', a, b, c, d, e, f)
    - (i,j): Female agent position
    - (i',j'): Male agent position
    - x: Female carrying block (0 or 1)
    - x': Male carrying block (0 or 1)
    - a: blocks in dropoff (1,1)
    - b: blocks in dropoff (1,5)
    - c: blocks in dropoff (3,3)
    - d: blocks in pickup (3,5)
    - e: blocks in pickup (4,2)
    - f: blocks in dropoff (5,5)
    """
    return (1, 3, 5, 3, 0, 0, 0, 0, 0, 10, 10, 0)


def is_terminal_state(state: Tuple) -> bool:
    """
    Check if the state is a terminal state.
    Terminal state: all dropoff locations have 5 blocks each.

    Args:
        state: Full state tuple

    Returns:
        bool: True if terminal state reached
    """
    i, j, i_prime, j_prime, x, x_prime, a, b, c, d, e, f = state

    return a == 5 and b == 5 and c == 5 and f == 5


def get_agent_position(state: Tuple, agent: str) -> Tuple[int, int]:
    """
    Extract agent position from state.

    Args:
        state: Full state tuple
        agent: 'F' for female, 'M' for male

    Returns:
        Tuple[int, int]: (row, column) position
    """
    i, j, i_prime, j_prime, x, x_prime, a, b, c, d, e, f = state
    if agent == 'F':
        return (i, j)
    else:
        return (i_prime, j_prime)


def get_agent_carrying(state: Tuple, agent: str) -> int:
    """
    Check if agent is carrying a block.

    Args:
        state: Full state tuple
        agent: 'F' or 'M'

    Returns:
        int: 1 if carrying, 0 if not
    """
    i, j, i_prime, j_prime, x, x_prime, a, b, c, d, e, f = state
    if agent == 'F':
        return x
    else:
        return x_prime


def get_block_counts(state: Tuple) -> Dict:
    """
    Extract block counts from state.

    Args:
        state: Full state tuple

    Returns:
        Dict: Block counts for each location
    """
    i, j, i_prime, j_prime, x, x_prime, a, b, c, d, e, f = state
    return {
        'dropoff_1_1': a,
        'dropoff_1_5': b,
        'dropoff_3_3': c,
        'pickup_3_5': d,
        'pickup_4_2': e,
        'dropoff_5_5': f
    }


'''
===========================================================
AGENT MOVEMENT
===========================================================
'''

def aplop(state: Tuple, agent: str) -> Set[str]:
    """
    Returns the set of applicable operators for the given agent in the given state.

    Enforces:
    - Grid boundaries (can't leave 5x5 grid)
    - Pickup conditions (must be at pickup location, location has blocks, not carrying)
    - Dropoff conditions (must be at dropoff location, location not full, carrying block)
    - Collision avoidance (can't move to cell occupied by other agent)

    Args:
        state: Full state tuple (i, j, i', j', x, x', a, b, c, d, e, f)
        agent: 'F' for female agent, 'M' for male agent

    Returns:
        Set[str]: Set of applicable operators for an agent - {'north', 'south', 'pickup'}
    """

    # Unpack the state
    i, j, i_prime, j_prime, x, x_prime, a, b, c, d, e, f = state

    # Get the current agents position and carry status
    if agent == 'F': # female
        agent_row, agent_col = i, j
        agent_carrying = x
        other_row, other_col = i_prime, j_prime
    else: # male
        agent_row, agent_col = i_prime, j_prime
        agent_carrying = x_prime
        other_row, other_col = i, j

    # All movement actions
    applicable_operators = set()

    # Check if agent can move north - not at the top edge boundary
    if agent_row > 1:
        new_row = agent_row - 1
        # check if other agent is north of agent
        if not(new_row == other_row and agent_col == other_col):
            applicable_operators.add('north')

    # Check if agent can move south - not at the bottom edge boundary
    if agent_row < GRID_SIZE:
        new_row = agent_row + 1
        # check if other agent is south of agent
        if not (new_row == other_row and agent_col == other_col):
            applicable_operators.add('south')

    # Check if agent can move east - not at the right edge boundary
    if agent_col < GRID_SIZE:
        new_col = agent_col + 1
        # check if other agent is east of agent
        if not (agent_row == other_row and new_col == other_col):
            applicable_operators.add('east')

    # Check if agent can move west - not at the left edge boundary
    if agent_col > 1:
        new_col = agent_col - 1
        # check if other agent is west of agent
        if not (agent_row == other_row and new_col == other_col):
            applicable_operators.add('west')

    # Check pickup location
    #  Conditions: must be at pickup location, location has blocks, and not carrying block
    if agent_carrying == 0: # not carrying block
        if(agent_row, agent_col) == (3, 5) and d > 0:
            applicable_operators.add('pickup')
        elif (agent_row, agent_col) == (4, 2) and e > 0:
            applicable_operators.add('pickup')

    # Check DROPOFF

    if agent_carrying == 1:  # Is carrying a block
        if (agent_row, agent_col) == (1, 1) and a < DROPOFF_CAPACITY:
            applicable_operators.add('dropoff')
        elif (agent_row, agent_col) == (1, 5) and b < DROPOFF_CAPACITY:
            applicable_operators.add('dropoff')
        elif (agent_row, agent_col) == (3, 3) and c < DROPOFF_CAPACITY:
            applicable_operators.add('dropoff')
        elif (agent_row, agent_col) == (5, 5) and f < DROPOFF_CAPACITY:
            applicable_operators.add('dropoff')

    return applicable_operators


# a decision will be made after receiving the result from aplop() then call apply()


def apply(state: Tuple, action: str, agent: str) -> Tuple[Tuple, int]:
    """
    Apply the given action for the specified agent, returning new state and reward.

    This function assumes the action is applicable (should be checked by aplop first).

    see get_initial_state() for explanation of letter variables

    Args:
        state: Current full state tuple
        action: Action to apply ('north', 'south', 'east', 'west', 'pickup', 'dropoff')
        agent: Which agent is acting ('F' or 'M')

    Returns:
        Tuple[Tuple, int]: (new_state, reward)
            - new_state: Updated state tuple after action
            - reward: Immediate reward for this action
    """
    # unpack the state
    i, j, i_prime, j_prime, x, x_prime, a, b, c, d, e, f = state

    # Start with current state values
    new_i, new_j = i, j
    new_i_prime, new_j_prime = i_prime, j_prime
    new_x, new_x_prime = x, x_prime
    new_a, new_b, new_c, new_d, new_e, new_f = a, b, c, d, e, f

    # Default reward
    reward = 0

    # Determine which agent to move
    if agent == 'F': # female
        agent_row, agent_col = i, j
        agent_carrying = x
    else:  # male
        agent_row, agent_col = i_prime, j_prime
        agent_carrying = x_prime

    # Execute action
    if action == 'north':
        # decrease row (up)
        if agent == 'F':
            new_i = agent_row - 1
        else:
            new_i_prime = agent_row - 1
        reward = REWARD_MOVE

    elif action == 'south':
        # increase row (down)
        if agent == 'F':
            new_i = agent_row + 1
        else:
            new_i_prime = agent_row + 1
        reward = REWARD_MOVE

    elif action == 'east':
        # increase column (right)
        if agent == 'F':
            new_j = agent_col + 1
        else:
            new_j_prime = agent_col + 1
        reward = REWARD_MOVE

    elif action == 'west':
        # decrease column (left)
        if agent == 'F':
            new_j = agent_col - 1
        else:
            new_j_prime = agent_col - 1
        reward = REWARD_MOVE

    elif action == 'pickup':
        # Set carrying status to 1
        if agent == 'F':
            new_x = 1
        else:
            new_x_prime = 1

        # Decrease block count of pickup location
        if (agent_row, agent_col) == (3, 5):
            new_d = d - 1
        elif (agent_row, agent_col) == (4, 2):
            new_e = e - 1

        reward = REWARD_PICKUP

    elif action == 'dropoff':
        # Set carrying status to 0
        if agent == 'F':
            new_x = 0
        else:
            new_x_prime = 0

        # Increase block count of the dropoff location
        if (agent_row, agent_col) == (1, 1):
            new_a = a + 1
        elif (agent_row, agent_col) == (1, 5):
            new_b = b + 1
        elif (agent_row, agent_col) == (3, 3):
            new_c = c + 1
        elif (agent_row, agent_col) == (5, 5):
            new_f = f + 1

        reward = REWARD_DROPOFF

    # Update state of the world
    new_state = (new_i, new_j, new_i_prime, new_j_prime,
                 new_x, new_x_prime, new_a, new_b, new_c,
                 new_d, new_e, new_f)

    return new_state, reward
