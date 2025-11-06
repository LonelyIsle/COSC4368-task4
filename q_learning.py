"""
Everything in this file is related to Q_learning
contains the algorithm and the means to update tables

for reference:
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

import numpy as np
import random
from typing import Tuple, Set, Dict

def create_q_table():
    return {}


def set_q_values(q_table: Dict, state: Tuple, action: str, value: float):
    """
    q_table: Dictionary storing Q-values
    state: Simplified state tuple
    action: Action string
    value: New Q-value
    """

    key = (state, action)
    q_table[key] = value

def get_q_values(q_table: Dict, state: Tuple, action: str) -> float:
    """
    q_table: Dictionary storing q-values
    state: Simplified state tuple (i, j, x, delta_i, delta_j)
    action: Action string
    """

    key = (state, action)
    return q_table.get(key, 0.0) # Return key or 0

def get_best_action(q_table: Dict, state: Tuple, applicable_actions: Set[str]) -> str:
    """
    Return action with highest Q-value from applicable actions.
    Breaks ties randomly.

    q_table: Dictionary storing Q-values
    state: Simplified state tuple
    applicable_actions: Set of valid actions
    """
    # Get q-values for all applicable actions
    action_q_vals = {}
    for action in applicable_actions:
        q_value = get_q_values(q_table, state, action)
        action_q_vals[action] = q_value

    # find the max q_value
    max_value = max(action_q_vals.values())

    # tie-breaker function - if multiple actions have the same value select at random
    best_action = []
    for action, q_value in action_q_vals.items():
        if q_value == max_value:
            best_action.append(action)

    return random.choice(best_action) # randomly pick action, not an issue if only one option


def get_max_q_value(q_table: Dict, state: Tuple, applicable_actions: Set[str]) -> float:
    """
    Return the highest Q-value among applicable actions.
    this function is for returning the number for q-learning formula

    q_table: Dictionary storing Q-values
    state: Simplified state tuple
    applicable_actions: Set of valid actions

    """
    # Safety check
    if not applicable_actions:
        return 0.0

    # Get Q-values for all actions
    q_values = []
    for action in applicable_actions:
        q_value = get_q_values(q_table, state, action)
        q_values.append(q_value)

    # Return the maximum
    return max(q_values)

def simplify_state(full_state: Tuple, agent: str) -> Tuple:
    """
    simplify the state to only use what is needed for the q-learning algorithm
    refer to the top of this code to see what the full state is and what variables mean what
    simplified state space : (i, j, x, delta_i, delta_j) is the end result
        - (i, j) = agent position
        - x = agent carrying block (boolean)
        - delta_i, delta_j = Relative row/column position to other agent
    """
    i, j, i_prime, j_prime, x, x_prime, a, b, c, d, e, f = full_state

    # determine which agent we are dealing with
    if agent == 'F': # Female agent
        delta_i = i - i_prime
        delta_j = j - j_prime
        return i, j, x, delta_i, delta_j
    else: # Male agent
        delta_i = i_prime - i
        delta_j = j_prime - j
        return i_prime, j_prime, x_prime, delta_i, delta_j

def update_q_learning(q_table: Dict,
                     state: Tuple,
                     action: str,
                     reward: float,
                     next_state: Tuple,
                     applicable_next_actions: Set[str],
                     alpha: float,
                     gamma: float):
    """
    Formula: Q(s,a) ← (1-α)*Q(s,a) + α*[R + γ*max Q(s',a')]

    q_table: Dictionary storing Q-values
        state: Current simplified state
        action: Action taken
        reward: Immediate reward received
        next_state: Next simplified state
        applicable_next_actions: Set of valid actions in next state
        alpha: Learning rate (0 < α ≤ 1)
        gamma: Discount factor (0 ≤ γ ≤ 1)

        note from slides:
            "a’ has to be an applicable operator in s’; e.g. pickup and drop-off are not
            applicable in a pickup/dropoff states if empty/full! The q-values of
            non-applicable operators are therefore not considered! "
    """
    old_q = get_q_values(q_table, state, action)

    # Get max q-value in next state (ONLY from the applicable actions!)
    max_next_q = get_max_q_value(q_table, next_state, applicable_next_actions)

    # Calculate new q-value using Q-Learning formula
    new_q = (1 - alpha) * old_q + alpha * (reward + gamma * max_next_q)

    # Update the q-table
    set_q_values(q_table, state, action, new_q)

