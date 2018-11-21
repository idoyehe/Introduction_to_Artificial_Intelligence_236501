from framework import *
from deliveries import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union

# Load the map
roads = load_map_from_csv(Consts.get_data_file_path("tlv.csv"))

# Make `np.random` behave deterministic.
Consts.set_seed()


# --------------------------------------------------------------------
# -------------------------- Map Problem -----------------------------
# --------------------------------------------------------------------

def plot_distance_and_expanded_wrt_weight_figure(
        weights: Union[np.ndarray, List[float]],
        total_distance: Union[np.ndarray, List[float]],
        total_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    """
    assert len(weights) == len(total_distance) == len(total_expanded)

    fig, ax1 = plt.subplots()

    ax1.plot(weights, total_distance, 'b')

    # See documentation here:
    # https://matplotlib.org/2.0.0/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also search google for additional examples.
    # raise NotImplemented()

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('distance traveled', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    ax2.plot(weights, total_expanded, 'r')
    ax2.set_ylabel('states expanded', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_xlabel('weight')

    fig.tight_layout()
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem):
    # 1. Create an array of 20 numbers equally spreaded in [0.5, 1]
    #    (including the edges). You can use `np.linspace()` for that.
    # 2. For each weight in that array run the A* algorithm, with the
    #    given `heuristic_type` over the map problem. For each such run,
    #    store the cost of the solution (res.final_search_node.cost)
    #    and the number of expanded states (res.nr_expanded_states).
    #    Store these in 2 lists (array for the costs and array for
    #    the #expanded.
    # Call the function `plot_distance_and_expanded_by_weight_figure()`
    #  with that data.
    weights = np.linspace(0.5, 1, 20)
    costs = []
    expanded = []
    for w in weights:
        astr_w = AStar(heuristic_type, w)
        res_w = astr_w.solve_problem(problem)
        costs.append(res_w.final_search_node.cost)
        expanded.append(res_w.nr_expanded_states)
    plot_distance_and_expanded_wrt_weight_figure(weights, costs, expanded)


def map_problem():
    print()
    print('Solve the map problem.')

    # Ex.8
    map_prob = MapProblem(roads, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(map_prob)
    print(res)

    # Ex.10
    #       solve the same `map_prob` with it and print the results (as before).
    # Notice: AStar constructor receives the heuristic *type* (ex: `MyHeuristicClass`),
    #         and not an instance of the heuristic (eg: not `MyHeuristicClass()`).
    astr10 = AStar(NullHeuristic)
    res10 = astr10.solve_problem(map_prob)
    print(res10)

    # Ex.11
    #       solve the same `map_prob` with it and print the results (as before).
    astr11 = AStar(AirDistHeuristic)
    res11 = astr11.solve_problem(map_prob)
    print(res11)

    # Ex.12
    # 1. Complete the implementation of the function
    #    `run_astar_for_weights_in_range()` (upper in this file).
    # 2. Complete the implementation of the function
    #    `plot_distance_and_expanded_by_weight_figure()`
    #    (upper in this file).
    # 3. Call here the function `run_astar_for_weights_in_range()`
    #    with `AirDistHeuristic` and `map_prob`.
    run_astar_for_weights_in_range(AirDistHeuristic, map_prob)


# --------------------------------------------------------------------
# ----------------------- Deliveries Problem -------------------------
# --------------------------------------------------------------------

def relaxed_deliveries_problem():
    print()
    print('Solve the relaxed deliveries problem.')

    big_delivery = DeliveriesProblemInput.load_from_file('big_delivery.in', roads)
    big_deliveries_prob = RelaxedDeliveriesProblem(big_delivery)

    # Ex.16
    # solve the `big_deliveries_prob` with it and print the results (as before).
    astr16 = AStar(MaxAirDistHeuristic)
    res16 = astr16.solve_problem(big_deliveries_prob)
    print(res16)

    # Ex.17
    astr17 = AStar(MSTAirDistHeuristic)
    res17 = astr17.solve_problem(big_deliveries_prob)
    print(res17)

    # Ex.18
    run_astar_for_weights_in_range(MSTAirDistHeuristic, big_deliveries_prob)

    # Ex.24
    # 1. Run the stochastic greedy algorithm for 100 times.
    #    For each run, store the cost of the found solution.
    #    Store these costs in a list.
    # 2. The "Anytime Greedy Stochastic Algorithm" runs the greedy
    #    greedy stochastic for N times, and after each iteration
    #    stores the best solution found so far. It means that after
    #    iteration #i, the cost of the solution found by the anytime
    #    algorithm is the MINIMUM among the costs of the solutions
    #    found in iterations {1,...,i}. Calculate the costs of the
    #    anytime algorithm wrt the #iteration and store them in a list.
    # 3. Calculate and store the cost of the solution received by
    #    the A* algorithm (with w=0.5).
    # 4. Calculate and store the cost of the solution received by
    #    the deterministic greedy algorithm (A* with w=1).
    # 5. Plot a figure with the costs (y-axis) wrt the #iteration
    #    (x-axis). Of course that the costs of A*, and deterministic
    #    greedy are not dependent with the iteration number, so
    #    these two should be represented by horizontal lines.
    iter_num = 100
    costs_list = []
    anytime_cost_list = np.zeros(iter_num)
    for i in range(iter_num):
        # greedy stochastic iterations
        grd_stchstc = GreedyStochastic(MSTAirDistHeuristic)
        res24 = grd_stchstc.solve_problem(big_deliveries_prob)
        costs_list.append(res24.final_search_node.cost)
        anytime_cost_list[i] = min(costs_list)

    # A* solution
    astr24 = AStar(MSTAirDistHeuristic, heuristic_weight=0.5)
    resAstar24 = astr24.solve_problem(big_deliveries_prob)
    aStar_cost_list = np.ones(iter_num) * resAstar24.final_search_node.cost
    # greedy best 1 solution
    greedyBest1 = AStar(MSTAirDistHeuristic, heuristic_weight=1)
    resGreedyBest1 = greedyBest1.solve_problem(big_deliveries_prob)
    greedyBest1_cost_list = np.ones(iter_num) * resGreedyBest1.final_search_node.cost
    fig, ax1 = plt.subplots()

    iterations = np.linspace(1, iter_num, iter_num)
    plt.plot(iterations, costs_list, label="Greedy stochastic")
    plt.plot(iterations, anytime_cost_list, label="Anytime algorithm")
    plt.plot(iterations, aStar_cost_list, label="Astar")
    plt.plot(iterations, greedyBest1_cost_list, label="Greedy Best First")

    ax1.set_ylabel('Cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('Iteration #')

    plt.title("Costs Vs. Iteration #")
    plt.legend()
    plt.grid()
    plt.show()



def strict_deliveries_problem():
    print()
    print('Solve the strict deliveries problem.')

    small_delivery = DeliveriesProblemInput.load_from_file('small_delivery.in', roads)
    small_deliveries_strict_problem = StrictDeliveriesProblem(
        small_delivery, roads, inner_problem_solver=AStar(AirDistHeuristic), use_cache=True)

    # Ex.26
    # Call here the function `run_astar_for_weights_in_range()`
    # with `MSTAirDistHeuristic` and `small_deliveries_prob`.
    run_astar_for_weights_in_range(MSTAirDistHeuristic, small_deliveries_strict_problem)

    # Ex.28
    # an instance of `AStar` with the `RelaxedDeliveriesHeuristic`,
    # solve the `small_deliveries_strict_problem` with it and print the results (as before).
    astr28 = AStar(RelaxedDeliveriesHeuristic)
    res28 = astr28.solve_problem(small_deliveries_strict_problem)
    print(res28)

def main():
    map_problem()
    relaxed_deliveries_problem()
    strict_deliveries_problem()


if __name__ == '__main__':
    main()
