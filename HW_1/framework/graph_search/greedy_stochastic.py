from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np


class GreedyStochastic(BestFirstSearch):
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor
        self.solver_name = 'GreedyStochastic (h={heuristic_name})'.format(
            heuristic_name=heuristic_function_type.heuristic_name)

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        if not self.open.has_state(successor_node.state) and not self.close.has_state(successor_node.state):
            self.open.push_node(successor_node)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        Remember: `GreedyStochastic` is greedy.
        """
        return self.heuristic_function.estimate(search_node.state)

    def _extract_next_search_node_to_expand(self) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         using the stochastic method to choose out of the N
         best items from open.
        Use `np.random.choice(...)` whenever you need to randomly choose
         an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
         see usage examples by searching it in Google.
        Notice: You might want to pop min(N, len(open) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """
        if self.open.is_empty():
            return None

        window_size = min(self.N, len(self.open))
        node_window = []
        for i in range(window_size):
            curr_node = self.open.pop_next_node()
            if curr_node.expanding_priority < 0.00001:
                return curr_node
            node_window.append(curr_node)

        pw = -float(1.0 / self.T)
        alpha_min = min([curr_node.expanding_priority for curr_node in node_window])

        nodes_prb = [(float(node.expanding_priority / alpha_min) ** pw) for node in node_window]
        sum_total = sum(nodes_prb)
        nodes_prb = [float(p / sum_total) for p in nodes_prb]

        chosen_node = np.random.choice(node_window, 1, nodes_prb)[0]
        node_window.remove(chosen_node)

        assert len(node_window) == window_size - 1
        for curr_node in node_window:
            # reinsert not chosen to open
            self.open.push_node(curr_node)

        # Update T
        self.T *= self.T_scale_factor
        assert not self.open.has_state(chosen_node)
        return chosen_node
