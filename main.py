from policies import PExploit, PGreedy, PRandom
from world import *
from q_learning import *

# its just tests for stuff I work on
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TESTING POLICIES")
    print("=" * 50)

    # Import what we need for testing
    from q_learning import create_q_table, set_q_values

    # Setup test scenario
    state = (3, 5, 0, -2, 2)
    q_table = create_q_table()

    # Set some Q-values
    set_q_values(q_table, state, 'north', 5.0)
    set_q_values(q_table, state, 'south', 3.0)
    set_q_values(q_table, state, 'east', 8.0)
    set_q_values(q_table, state, 'west', 2.0)

    print(f"\nTest state: {state}")
    print(f"Q-values: north=5.0, south=3.0, east=8.0, west=2.0")

    # Test 1: PRandom (without pickup/dropoff)
    print("\n" + "-" * 50)
    print("TEST 1: PRandom (no pickup/dropoff available)")
    print("-" * 50)

    applicable = {'north', 'south', 'east', 'west'}

    print(f"Applicable actions: {applicable}")
    print("Running PRandom 10 times (should be random distribution):")

    random_results = {}
    for i in range(10):
        action = PRandom(state, applicable, q_table)
        random_results[action] = random_results.get(action, 0) + 1

    print(f"Results: {random_results}")
    print("(Should see roughly equal distribution)")

    # Test 2: PRandom (with pickup)
    print("\n" + "-" * 50)
    print("TEST 2: PRandom (with pickup available)")
    print("-" * 50)

    applicable_with_pickup = {'north', 'south', 'pickup', 'west'}

    print(f"Applicable actions: {applicable_with_pickup}")
    print("Running PRandom 5 times:")

    for i in range(5):
        action = PGreedy(state, applicable_with_pickup, q_table)
        print(f"  Attempt {i + 1}: {action}")

    print("(Should ALWAYS be 'pickup')")

    # Test 3: PGreedy
    print("\n" + "-" * 50)
    print("TEST 3: PGreedy")
    print("-" * 50)

    applicable = {'north', 'south', 'east', 'west'}

    print(f"Applicable actions: {applicable}")
    print("Running PGreedy 5 times:")

    for i in range(5):
        action = PGreedy(state, applicable, q_table)
        print(f"  Attempt {i + 1}: {action}")

    print("(Should ALWAYS be 'east' with Q=8.0)")

    # Test 4: PExploit
    print("\n" + "-" * 50)
    print("TEST 4: PExploit (epsilon=0.2)")
    print("-" * 50)

    applicable = {'north', 'south', 'east', 'west'}

    print(f"Applicable actions: {applicable}")
    print("Running PExploit 20 times:")

    exploit_results = {}
    for i in range(20):
        action = PExploit(state, applicable, q_table, epsilon=0.15)
        exploit_results[action] = exploit_results.get(action, 0) + 1

    print(f"Results: {exploit_results}")
    print("(Should see 'east' ~16 times (85%), others ~4 times total (15%))")

    # Test 5: PExploit with pickup
    print("\n" + "-" * 50)
    print("TEST 5: PExploit (with pickup available)")
    print("-" * 50)

    applicable_with_pickup = {'north', 'pickup', 'east', 'west'}

    print(f"Applicable actions: {applicable_with_pickup}")
    print("Running PExploit 5 times:")

    for i in range(5):
        action = PExploit(state, applicable_with_pickup, q_table)
        print(f"  Attempt {i + 1}: {action}")

    print("(Should ALWAYS be 'pickup' - overrides epsilon-greedy)")

    print("\n" + "=" * 50)
    print("✅ All policy tests complete!")
    print("=" * 50)