from world import *

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TESTING LAYER 1: World Representation")
    print("=" * 50)

    state = get_initial_state()
    print(f"\nInitial state: {state}")

    print(f"\nAgent F position: {get_agent_position(state, 'F')}")
    print(f"Agent M position: {get_agent_position(state, 'M')}")

    print(f"\nAgent F carrying: {get_agent_carrying(state, 'F')}")
    print(f"Agent M carrying: {get_agent_carrying(state, 'M')}")

    print(f"\nBlock counts: {get_block_counts(state)}")

    print(f"\nIs terminal? {is_terminal_state(state)}")

    # Test terminal state
    terminal_state = (1, 1, 5, 5, 0, 0, 5, 5, 5, 0, 0, 5)
    print(f"\nTerminal state: {terminal_state}")
    print(f"Is terminal? {is_terminal_state(terminal_state)}")

    print("\n" + "=" * 50)
    print("TESTING LAYER 2: Physics Engine")
    print("=" * 50)

    # Test aplop - What can Female agent do at start?
    state = get_initial_state()
    applicable_f = aplop(state, 'F')
    print(f"\nAt initial state, Agent F can do: {applicable_f}")

    # Test aplop - What can Male agent do at start?
    applicable_m = aplop(state, 'M')
    print(f"At initial state, Agent M can do: {applicable_m}")

    # Test apply - Move Female agent south
    print(f"\n--- Testing MOVE action ---")
    print(f"Before: F at {get_agent_position(state, 'F')}")
    new_state, reward = apply(state, 'south', 'F')
    print(f"After SOUTH: F at {get_agent_position(new_state, 'F')}")
    print(f"Reward: {reward}")

    # Test pickup - Move F to pickup location and pickup
    print(f"\n--- Testing PICKUP action ---")
    state = (3, 5, 5, 3, 0, 0, 0, 0, 0, 10, 10, 0)  # F at pickup (3,5)
    print(f"F at {get_agent_position(state, 'F')}, carrying: {get_agent_carrying(state, 'F')}")
    print(f"Pickup (3,5) has {get_block_counts(state)['pickup_3_5']} blocks")

    applicable = aplop(state, 'F')
    print(f"Applicable actions: {applicable}")

    if 'pickup' in applicable:
        new_state, reward = apply(state, 'pickup', 'F')
        print(f"After PICKUP: carrying: {get_agent_carrying(new_state, 'F')}")
        print(f"Pickup (3,5) now has {get_block_counts(new_state)['pickup_3_5']} blocks")
        print(f"Reward: {reward}")

    # Test dropoff - Move F to dropoff location with block and dropoff
    print(f"\n--- Testing DROPOFF action ---")
    state = (1, 1, 5, 3, 1, 0, 0, 0, 0, 9, 10, 0)  # F at dropoff (1,1), carrying block
    print(f"F at {get_agent_position(state, 'F')}, carrying: {get_agent_carrying(state, 'F')}")
    print(f"Dropoff (1,1) has {get_block_counts(state)['dropoff_1_1']} blocks")

    applicable = aplop(state, 'F')
    print(f"Applicable actions: {applicable}")

    if 'dropoff' in applicable:
        new_state, reward = apply(state, 'dropoff', 'F')
        print(f"After DROPOFF: carrying: {get_agent_carrying(new_state, 'F')}")
        print(f"Dropoff (1,1) now has {get_block_counts(new_state)['dropoff_1_1']} blocks")
        print(f"Reward: {reward}")

    # Test collision avoidance
    print(f"\n--- Testing COLLISION AVOIDANCE ---")
    state = (3, 4, 3, 5, 0, 0, 0, 0, 0, 10, 10, 0)  # F at (3,4), M at (3,5)
    print(f"F at {get_agent_position(state, 'F')}")
    print(f"M at {get_agent_position(state, 'M')}")

    applicable = aplop(state, 'F')
    print(f"F can do: {applicable}")
    print(f"'east' in applicable? {'east' in applicable} (should be False - M is blocking!)")
