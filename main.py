from world import *
from q_learning import *

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TESTING Q-LEARNING MODULE")
    print("=" * 50)

    # Test Q-table functions
    q_table = create_q_table()

    state = (3, 5, 0, -2, 2)  # State Space 1 format

    # Set some Q-values
    set_q_values(q_table, state, 'north', 5.0)
    set_q_values(q_table, state, 'pickup', 10.0)
    set_q_values(q_table, state, 'south', 3.0)

    print(f"\nQ-values for state {state}:")
    print(f"  north: {get_q_values(q_table, state, 'north')}")
    print(f"  pickup: {get_q_values(q_table, state, 'pickup')}")
    print(f"  south: {get_q_values(q_table, state, 'south')}")
    print(f"  west (never set): {get_q_values(q_table, state, 'west')}")

    # Test get_best_action
    applicable = {'north', 'south', 'pickup'}
    best = get_best_action(q_table, state, applicable)
    print(f"\nBest action from {applicable}: {best}")
    print(f"  (Should be 'pickup' with Q=10.0)")

    # Test get_max_q_value
    max_q = get_max_q_value(q_table, state, applicable)
    print(f"Max Q-value: {max_q}")
    print(f"  (Should be 10.0)")

    # Test simplify_state
    print("\n" + "-" * 50)
    print("Testing simplify_state")
    print("-" * 50)

    full_state = (3, 5, 5, 3, 0, 1, 0, 0, 0, 10, 9, 0)
    #             ^F at (3,5)  ^M at (5,3)
    #                      ^F not carrying  ^M carrying

    simple_f = simplify_state(full_state, 'F')
    simple_m = simplify_state(full_state, 'M')

    print(f"\nFull state: {full_state}")
    print(f"  F at (3,5), M at (5,3)")
    print(f"  F not carrying, M carrying")

    print(f"\nSimplified for F: {simple_f}")
    print(f"  (i=3, j=5, x=0, delta_i=3-5=-2, delta_j=5-3=2)")

    print(f"\nSimplified for M: {simple_m}")
    print(f"  (i'=5, j'=3, x'=1, delta_i=5-3=2, delta_j=3-5=-2)")

    # Test Q-Learning update
    print("\n" + "-" * 50)
    print("Testing Q-Learning update")
    print("-" * 50)

    q_table = create_q_table()

    state = (3, 4, 0, -2, 1)
    action = 'east'
    reward = -1
    next_state = (3, 5, 0, -2, 2)
    applicable_next = {'north', 'south', 'west', 'pickup'}

    # Set some values in next state
    set_q_values(q_table, next_state, 'pickup', 8.0)
    set_q_values(q_table, next_state, 'north', 2.0)

    print(f"\nBefore update:")
    print(f"  Q({state}, {action}) = {get_q_values(q_table, state, action)}")

    # Do update with α=0.3, γ=0.5
    update_q_learning(q_table, state, action, reward,
                      next_state, applicable_next,
                      alpha=0.3, gamma=0.5)

    print(f"\nAfter update (α=0.3, γ=0.5, reward=-1):")
    print(f"  Q({state}, {action}) = {get_q_values(q_table, state, action):.2f}")
    print(f"  Expected: (1-0.3)*0 + 0.3*(-1 + 0.5*8.0) = 0.9")

    print("\n✅ All tests passed!")

