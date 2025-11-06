from typing import Tuple, Set, Dict
import random
from q_learning import *

def PRandom(state: Tuple, applicable_actions: Set[str], q_table: Dict) -> str:
    """
    if the agent has the option to pickup or dropoff, select it
    applicable_actions: Set of valid actions
    """
    if 'pickup' in applicable_actions:
        return 'pickup'

    if 'dropoff' in applicable_actions:
        return 'dropoff'

    # if the above 2 actions are not an option select from the ones in the set
    actions = list(applicable_actions) # needs to be in a list for random.choice
    return random.choice(actions)


def PExploit(state: Tuple, applicable_actions: Set[str], q_table: Dict, epsilon: float = 0.15) -> str:

    """
    if the agent has the option to pickup or dropoff, select it
    otherwise, apply the applicable operator with the highest q-value with a 85% probability
    and choose randomly with a 15% probability
    """

    if 'pickup' in applicable_actions:
        return 'pickup'

    if 'dropoff' in applicable_actions:
        return 'dropoff'

    # Generate random number between 0 and 1. This will get the .85 and .15 probabilities needed for this policy
    random_value = random.random()

    if random_value < epsilon:
        # Choose randomly
        actions = list(applicable_actions) # needs to be in a list for random.choice
        return random.choice(actions)
    else:
        # Choose best Q-value
        return get_best_action(q_table, state, applicable_actions)


def PGreedy(state: Tuple, applicable_actions: Set[str], q_table: Dict) -> str:
    """
    If pickup and dropoff is applicable, choose this
    operator; otherwise, apply the applicable operator with the
    highest q-value (break ties by rolling a dice for operators
    with the same utility)
    """

    if 'pickup' in applicable_actions:
        return 'pickup'

    if 'dropoff' in applicable_actions:
        return 'dropoff'

    # return action with highest q-value
    return get_best_action(q_table, state, applicable_actions)
