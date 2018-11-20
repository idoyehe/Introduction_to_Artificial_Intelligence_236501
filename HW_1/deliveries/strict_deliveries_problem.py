from framework.graph_search import *
from framework.ways import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem

from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.
    """
    pass


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel)
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)

        # Get the junction (in the map) that is represented by the state to expand.
        state_junction = state_to_expand.current_location

        # Iterate over the orders that haven't been dropped.
        for next_stop in self.possible_stop_points:

            # trying to get from cache the cost from current junction to the next junction
            cache_key = (state_junction.index, next_stop.index)
            operator_cost = self._get_from_cache(cache_key)

            if operator_cost is None:
                map_problem = MapProblem(self.roads, state_junction.index, next_stop.index)
                map_problem_sol = self.inner_problem_solver.solve_problem(map_problem)
                operator_cost = map_problem_sol.final_search_node.cost
                self._insert_to_cache(cache_key, operator_cost)

            # Check fuel subject to air distance
            if operator_cost > state_to_expand.fuel:
                continue

            if next_stop in self.gas_stations:
                # initialize new state of gas station with same dropped_so_far and gas_tank_capacity
                successor_state = StrictDeliveriesState(next_stop,
                                                        state_to_expand.dropped_so_far,
                                                        self.gas_tank_capacity)
            else:
                assert next_stop in self.drop_points
                if next_stop not in state_to_expand.dropped_so_far:
                    # creating new successor dropped_so_far by union of state_to_expand.dropped_so_far and {next_stop}
                    succ_dropped_so_far = state_to_expand.dropped_so_far.union(frozenset([next_stop]))
                    # initialize new state of drop point with succ_dropped_so_far and state_to_expand.fuel - operator_cost
                    successor_state = StrictDeliveriesState(next_stop,
                                                            succ_dropped_so_far,
                                                            state_to_expand.fuel - operator_cost)
                else:
                    continue
            # Yield the successor state and the cost of the operator we used to get this successor.
            yield successor_state, operator_cost

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        """
        assert isinstance(state, StrictDeliveriesState)

        return state.current_location in self.drop_points and self.drop_points == state.dropped_so_far
