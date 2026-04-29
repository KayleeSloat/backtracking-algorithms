import random
import time
import sys
from functools import lru_cache

# ------------------------------------------------------------
# Dynamic Programming (0/1 Knapsack) - O(n * W) time, O(W) space
# ------------------------------------------------------------
def knapsack_dp(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # iterate backwards to ensure 0/1 property
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# ------------------------------------------------------------
# Backtracking with pruning (branch and bound)
# ------------------------------------------------------------
def knapsack_bt(weights, values, capacity):
    n = len(weights)
    # sort by value/weight ratio for better pruning (optional but helps)
    items = sorted(zip(weights, values), key=lambda x: x[1]/x[0], reverse=True)
    weights_sorted, values_sorted = zip(*items) if items else ([], [])
    
    best_value = 0
    
    def bound(i, curr_weight, curr_value):
        """upper bound of value achievable from item i onward (fractional knapsack)"""
        if curr_weight > capacity:
            return 0
        remaining = capacity - curr_weight
        bound_val = curr_value
        j = i
        while j < n and weights_sorted[j] <= remaining:
            remaining -= weights_sorted[j]
            bound_val += values_sorted[j]
            j += 1
        if j < n:  # take fraction of next item
            bound_val += (remaining / weights_sorted[j]) * values_sorted[j]
        return bound_val
    
    def backtrack(i, curr_weight, curr_value):
        nonlocal best_value
        if curr_weight > capacity:
            return
        if i == n:
            if curr_value > best_value:
                best_value = curr_value
            return
        
        # prune if bound cannot beat best
        if bound(i, curr_weight, curr_value) <= best_value:
            return
        
        # take item i
        backtrack(i + 1, curr_weight + weights_sorted[i], curr_value + values_sorted[i])
        # skip item i
        backtrack(i + 1, curr_weight, curr_value)
    
    backtrack(0, 0, 0)
    return best_value

# ------------------------------------------------------------
# Fixture to test both algorithms on same dataset multiple times
# ------------------------------------------------------------
def generate_knapsack_instance(n, max_weight, max_value, capacity_factor=0.5):
    """Generate random weights (1..max_weight), values (1..max_value)"""
    weights = [random.randint(1, max_weight) for _ in range(n)]
    values = [random.randint(1, max_value) for _ in range(n)]
    capacity = int(sum(weights) * capacity_factor)
    return weights, values, capacity

def time_algorithm(algorithm, weights, values, capacity, runs=5):
    """Measure average runtime over multiple runs"""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = algorithm(weights, values, capacity)
        end = time.perf_counter()
        times.append(end - start)
    avg_time = sum(times) / runs
    return avg_time, result

def run_comparison(n, max_weight, max_value, capacity_factor, runs_per_alg=3):
    print(f"\n{'='*60}")
    print(f"Test: n={n}, weight range=1..{max_weight}, value range=1..{max_value}")
    print(f"Capacity factor = {capacity_factor}")
    
    weights, values, capacity = generate_knapsack_instance(n, max_weight, max_value, capacity_factor)
    print(f"Total weight sum = {sum(weights)}, capacity = {capacity}")
    
    # DP timing
    dp_time, dp_val = time_algorithm(knapsack_dp, weights, values, capacity, runs=runs_per_alg)
    
    # BT timing (careful: only run if n is not too large)
    if n > 35:
        print("BT skipped: n too large (>35) would take astronomical time")
        bt_time, bt_val = float('inf'), None
    else:
        bt_time, bt_val = time_algorithm(knapsack_bt, weights, values, capacity, runs=runs_per_alg)
    
    print(f"\nResults (avg over {runs_per_alg} runs):")
    print(f"DP: {dp_time:.6f} sec, best value = {dp_val}")
    if bt_val is not None:
        print(f"BT: {bt_time:.6f} sec, best value = {bt_val}")
        print(f"Speedup (DP/BT): {bt_time/dp_time:.2f}x")
    else:
        print(f"BT: skipped")
    
    return dp_time, bt_time

# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)  # reproducible results
    
    print("KNAPSACK ALGORITHM COMPARISON")
    print("DP = Dynamic Programming (O(nW)), BT = Backtracking (O(2^n))")
    
    # Small instance (n=20) – BT still feasible
    run_comparison(n=20, max_weight=20, max_value=100, capacity_factor=0.5, runs_per_alg=5)
    
    # Medium instance (n=30) – BT starting to struggle
    run_comparison(n=30, max_weight=30, max_value=100, capacity_factor=0.5, runs_per_alg=5)
    
    # Large instance (n=50) – BT impossible
    run_comparison(n=50, max_weight=50, max_value=100, capacity_factor=0.5, runs_per_alg=3)
    
    # Very large n, small W – DP still fast
    run_comparison(n=2000, max_weight=10, max_value=100, capacity_factor=0.5, runs_per_alg=3)
    
    print("\n" + "="*60)
    print("CONCLUSION: DP wins for all but trivially small n or huge W.")
    print("Backtracking is only viable when n <= 30 due to exponential blowup.")
