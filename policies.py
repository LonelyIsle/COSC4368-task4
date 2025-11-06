# policies.py
from typing import Iterable, Dict, Tuple, List
import random
import math

State = Tuple  # your state tuple

def _as_list(actions: Iterable[str]) -> List[str]:
    lst = list(actions)
    if not lst:
        raise ValueError("No applicable actions available.")
    return lst

def _q(Q: Dict[Tuple[State, str], float], state: State, action: str) -> float:
    # experiments.py stores Q with keys: (state, action)
    return Q.get((state, action), 0.0)

def _best_action(Q: Dict[Tuple[State, str], float], state: State, actions: Iterable[str]) -> str:
    acts = _as_list(actions)
    best_q = -math.inf
    best: List[str] = []
    for a in acts:
        q = _q(Q, state, a)
        if q > best_q:
            best_q = q
            best = [a]
        elif q == best_q:
            best.append(a)
    return random.choice(best) if best else random.choice(acts)

# ---------------------------
# Policies used by the project
# ---------------------------

def prandom(state: State, applicable_actions: Iterable[str], q_table: Dict) -> str:
    """
    PRANDOM: If pickup or dropoff is applicable, do it; otherwise choose random.
    """
    acts = _as_list(applicable_actions)
    if 'pickup' in acts:
        return 'pickup'
    if 'dropoff' in acts:
        return 'dropoff'
    return random.choice(acts)

def pgreedy(state: State, applicable_actions: Iterable[str], q_table: Dict) -> str:
    """
    PGREEDY: If pickup/dropoff is applicable, do it; otherwise choose argmax-Q (break ties randomly).
    """
    acts = _as_list(applicable_actions)
    if 'pickup' in acts:
        return 'pickup'
    if 'dropoff' in acts:
        return 'dropoff'
    return _best_action(q_table, state, acts)

def pexploit(state: State, applicable_actions: Iterable[str], q_table: Dict, epsilon: float = 0.2) -> str:
    """
    PEXPLOIT: If pickup/dropoff is applicable, do it; otherwise:
      - with probability (1 - epsilon) pick greedy action
      - with probability epsilon pick a random applicable action
    Default epsilon=0.2 matches spec's 80/20 split.
    """
    acts = _as_list(applicable_actions)
    if 'pickup' in acts:
        return 'pickup'
    if 'dropoff' in acts:
        return 'dropoff'
    if random.random() < epsilon:
        return random.choice(acts)
    return _best_action(q_table, state, acts)

# ---------------------------
# Backward-compatible aliases (your older code can still call these)
# ---------------------------

def PRandom(state: State, applicable_actions: Iterable[str], q_table: Dict) -> str:
    return prandom(state, applicable_actions, q_table)

def PGreedy(state: State, applicable_actions: Iterable[str], q_table: Dict) -> str:
    return pgreedy(state, applicable_actions, q_table)

def PExploit(state: State, applicable_actions: Iterable[str], q_table: Dict, epsilon: float = 0.2) -> str:
    return pexploit(state, applicable_actions, q_table, epsilon=epsilon)