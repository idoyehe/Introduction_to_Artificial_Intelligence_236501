import random
import util
from game import Actions
from game import Agent


#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def __init__(self):
        super().__init__(index=0)
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
      getAction chooses among the best options according to the evaluation function.

      getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
      ------------------------------------------------------------------------------
      """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
      The evaluation function takes in the current GameState (pacman.py) and the proposed action
      and returns a number, where higher numbers are better.
      """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # return scoreEvaluationFunction(successorGameState) # old evaluation function call
        return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
    return gameState.getScore()


######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """
  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.
  """
    better_evaluated_score = gameState.getScore()

    successors_total_scores = 0
    legal_actions = 0
    for action in gameState.getLegalPacmanActions():
        legal_actions += 1
        pacman_state = gameState.generatePacmanSuccessor(action)
        successors_total_scores += pacman_state.getScore()

    if legal_actions > 0:
        better_evaluated_score = successors_total_scores / legal_actions

    pacman_pos = gameState.getPacmanPosition()

    ghosts_score = 0
    for ghost in gameState.getGhostStates():
        calc_manhattan = util.manhattanDistance(pacman_pos, ghost.configuration.pos)
        if calc_manhattan < 4:
            if ghost.scaredTimer > 2:  # Ghost Scared
                ghosts_score += calc_manhattan
            else:
                ghosts_score -= calc_manhattan

    better_evaluated_score += ghosts_score

    food = gameState.getFood().asList()

    current_food = pacman_pos
    while len(food) > 0:
        # return the nearest_food using the below line.
        nearest_food = min(food, key=lambda y: util.manhattanDistance(y, current_food))
        better_evaluated_score -= util.manhattanDistance(nearest_food, current_food)
        food.remove(nearest_food)
        current_food = nearest_food

    capsules = gameState.getCapsules()

    if 0 < len(capsules):
        nearest_capsules = min(capsules, key=lambda y: util.manhattanDistance(y, pacman_pos))
        better_evaluated_score -= util.manhattanDistance(nearest_capsules, pacman_pos)

    better_evaluated_score -= 2 * (
            gameState.hasWall(pacman_pos[0] + 1, pacman_pos[1]) and gameState.hasWall(pacman_pos[0] - 1, pacman_pos[1]))
    better_evaluated_score -= 2 * (
            gameState.hasWall(pacman_pos[0], pacman_pos[1] + 1) and gameState.hasWall(pacman_pos[0], pacman_pos[1] - 1))

    return better_evaluated_score


#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        super().__init__(index=0)  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.layers_developed = None  # number of layers to developed when calculating next action
        self.total_game_agents = None  # number of played agents
        self.next_action = None  # save the next action to preform


##############################################################################################################################

# c: implementing minimax
class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent
  """

    def getAction(self, gameState):
        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.
    """
        self.total_game_agents = gameState.getNumAgents()
        self.layers_developed = self.depth * self.total_game_agents
        self._rb_minimax_(gameState, self.index, self.layers_developed)
        return self.next_action

    def _rb_minimax_(self, game_state, agent_index, layers_number):
        """
        :param game_state: the current game state
        :param agent_index: the agent that play now
        :param layers_number: layers of number to develop
        :return: the action to preform
        """
        if game_state.isLose() or game_state.isWin() or layers_number == 0:
            return self.evaluationFunction(game_state)

        next_agent_index = (agent_index + 1) % self.total_game_agents

        if agent_index == self.index:  # Pacman agent
            current_max = float("-inf")  # initialized with -inf
            chosen_action = None
            for action in game_state.getLegalPacmanActions():
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_minimax_(successor_state, next_agent_index, layers_number - 1)
                if current_max < next_agent_value:
                    current_max = next_agent_value
                    chosen_action = action

            if layers_number == self.layers_developed:  # to return the action to the caller
                self.next_action = chosen_action

            return current_max  # return the max value to the recursive calls

        else:  # not Pacman turn -> other agents means ghosts
            current_min = float("inf")  # initialized with inf
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_minimax_(successor_state, next_agent_index, layers_number - 1)
                current_min = min(current_min, next_agent_value)
            return current_min


##############################################################################################################################

# d: implementing alpha-beta
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
        self.total_game_agents = gameState.getNumAgents()
        self.layers_developed = self.depth * self.total_game_agents
        self._rb_alpha_beta_(gameState, self.index, self.layers_developed, alpha=float('-inf'), beta=float('inf'))
        return self.next_action

    def _rb_alpha_beta_(self, game_state, agent_index, layers_number, alpha, beta):
        """
        :param game_state: the current game state
        :param agent_index: the agent that play now
        :param layers_number: layers of number to develop
        :param alpha: current alpha value
        :param beta: current beta value
        :return: the action to preform
        """
        if game_state.isLose() or game_state.isWin() or layers_number == 0:
            return self.evaluationFunction(game_state)

        next_agent_index = (agent_index + 1) % self.total_game_agents

        if agent_index == self.index:  # pacman agent
            current_max = float("-inf")
            chosen_action = None
            for action in game_state.getLegalPacmanActions():
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_alpha_beta_(successor_state, next_agent_index, layers_number - 1,
                                                        alpha=alpha, beta=beta)

                if current_max < next_agent_value:
                    current_max = next_agent_value
                    chosen_action = action
                alpha = max(current_max, alpha)
                if current_max >= beta:
                    return float("inf")

            if layers_number == self.layers_developed:  # to return the action to the caller
                self.next_action = chosen_action

            return current_max  # return the max value to the recursive calls

        else:  # not pacman turn -> other agents means ghosts
            current_min = float("inf")
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_alpha_beta_(successor_state, next_agent_index, layers_number - 1,
                                                        alpha=alpha, beta=beta)
                current_min = min(current_min, next_agent_value)
                beta = min(current_min, beta)
                if current_min <= alpha:
                    return float("-inf")
            return current_min


##############################################################################################################################

# e: implementing random expectimaxw
class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """
        self.total_game_agents = gameState.getNumAgents()
        self.layers_developed = self.depth * self.total_game_agents
        self._rb_random_expectimax_(gameState, self.index, self.layers_developed)
        return self.next_action

    def _rb_random_expectimax_(self, game_state, agent_index, layers_number):
        if game_state.isLose() or game_state.isWin() or layers_number == 0:
            return self.evaluationFunction(game_state)

        next_agent_index = (agent_index + 1) % self.total_game_agents

        if agent_index == self.index:  # pacman agent
            current_max = float("-inf")
            chosen_action = None
            for action in game_state.getLegalPacmanActions():
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_random_expectimax_(successor_state, next_agent_index, layers_number - 1)
                if current_max < next_agent_value:
                    current_max = next_agent_value
                    chosen_action = action

            if layers_number == self.layers_developed:  # to return the action to the caller
                self.next_action = chosen_action

            return current_max  # return the max value to the recursive calls

        else:  # not pacman turn -> other agents means ghosts
            sum_of_next_values = 0
            count_get_legal_actions = 0
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_random_expectimax_(successor_state, next_agent_index, layers_number - 1)
                sum_of_next_values += next_agent_value
                count_get_legal_actions += 1
            expectimax_value = sum_of_next_values / float(count_get_legal_actions)  # Expected value calculation
            return expectimax_value


##############################################################################################################################

# f: implementing directional expectimax
class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """
        self.total_game_agents = gameState.getNumAgents()
        self.layers_developed = self.depth * self.total_game_agents
        self._rb_directional_expectimax_(gameState, self.index, self.layers_developed)
        return self.next_action

    def _rb_directional_expectimax_(self, game_state, agent_index, layers_number):
        if game_state.isLose() or game_state.isWin() or layers_number == 0:
            return self.evaluationFunction(game_state)

        next_agent_index = (agent_index + 1) % self.total_game_agents

        if agent_index == self.index:  # pacman agent
            current_max = float("-inf")
            chosen_action = None
            for action in game_state.getLegalPacmanActions():
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_directional_expectimax_(successor_state, next_agent_index, layers_number - 1)
                if current_max < next_agent_value:
                    current_max = next_agent_value
                    chosen_action = action

            if layers_number == self.layers_developed:  # to return the action to the caller
                self.next_action = chosen_action

            return current_max  # return the max value to the recursive calls

        else:  # not pacman turn -> other agents means ghosts
            expectimax_value = 0
            dist = DirectionalExpectimaxAgent._get_directional_ghost_dist_(game_state, agent_index, 0.8, 0.8)
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_directional_expectimax_(successor_state, next_agent_index, layers_number - 1)
                expectimax_value += dist[action] * float(next_agent_value)
            return expectimax_value

    @staticmethod
    def _get_directional_ghost_dist_(game_state, ghost_index, prob_scaredFlee, prob_attack):
        ghostState = game_state.getGhostState(ghost_index)
        legalActions = game_state.getLegalActions(ghost_index)
        pos = game_state.getGhostPosition(ghost_index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = game_state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [util.manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = prob_attack

        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist


##############################################################################################################################
# implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent): # TODO: implement
    """
    Your competition agent
  """

    def getAction(self, gameState):
        """
      Returns the action using self.depth and self.evaluationFunction

    """

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


# # Dictionary for getting the layer name from layer sizes
# # Key: tuple of width and height (width, height)
# # Value: name of the layout - str
# layout_by_size = {(19, 7): 'capsuleClassic.lay',
#                   (20, 9): 'contestClassic.lay',
#                   (20, 11): 'mediumClassic.lay',
#                   (9, 5): 'minimaxClassic.lay',
#                   (25, 9): 'openClassic.lay',
#                   (28, 27): 'originalClassic.lay',
#                   (20, 7): 'smallClassic.lay',
#                   (5, 10): 'testClassic.lay',
#                   (8, 5): 'trappedClassic.lay',
#                   (20, 13): 'trickyClassic.lay'}
#
#
# def get_layout_name(layout):
#     return layout_by_size[(layout.width, layout.height)]
#
#
# # Dictionary to get the *real* shortest maze distance
# # between 2 positions
# # Key: layout name
# # Value: matrix with distance values - in the size of (layout width x layout height)
# # maze_distances =
#
#
# def generate_distance_matrix_by_layout(layout):
#     n_positions = layout.height * layout.width
#     dist_mat = [[None] * n_positions] * n_positions
#     graph = layout_to_graph(layout)
#     for x in range(layout.width):
#         for y in range(layout.height):
#             index = y*layout.width + x
#             # if (x,y) is a wall
#             if layout.layoutText[y][x] == '%':
#                 dist_mat[index] = [-1]*n_positions
#             else:                    # else calc distance to all other positions
#                 dist_mat[index] = calc_distance(graph, layout.width, layout.height, x, y)
#     return dist_mat
#
# # BFS function - calc distance from starting point (x,y) to all other positions
# def calc_distance(graph, width, height, x, y):
#     distance = [-1] * width * height
#     distance[y*width + x] = 0
#     queue = [y*width + x]
#     visited = []
#     while len(queue) != 0:
#         next = queue.pop(0)
#         visited.append(next)
#         for neighbor in graph[next]:
#             if distance[neighbor] == -1 or distance[neighbor] > distance[next]+1:
#                 distance[neighbor] = distance[next]+1
#                 if neighbor not in visited:
#                     queue.append(neighbor)
#     return distance
#
#
# def layout_to_graph(layout):
#     graph={}
#     for x in range(layout.width):
#         for y in range(layout.height):
#             connected = []
#             index = y*layout.width + x
#             if layout.layoutText[y][x] != '%':
#                 if layout.layoutText[y][x+1] != '%':
#                     connected.append(y * layout.width + x+1)
#                 if layout.layoutText[y][x-1] != '%':
#                     connected.append(y * layout.width + x-1)
#                 if layout.layoutText[y+1][x] != '%':
#                     connected.append((y+1) * layout.width + x)
#                 if layout.layoutText[y-1][x] != '%':
#                     connected.append((y-1) * layout.width + x)
#             graph[index] = connected
#     return graph
#
#
# all_layouts_path = ['HW_2\\layouts\\capsuleClassic.lay', 'HW_2\\layouts\\contestClassic.lay', 'HW_2\\layouts\\mediumClassic.lay', 'HW_2\\layouts\\minimaxClassic.lay', 'HW_2\\layouts\\openClassic.lay', 'HW_2\\layouts\\originalClassic.lay', 'HW_2\\layouts\\smallClassic.lay', 'HW_2\\layouts\\testClassic.lay', 'HW_2\\layouts\\trappedClassic.lay', 'HW_2\\layouts\\trickyClassic.lay']
#
# all_layouts = []
# for lay in all_layouts_path:
#     from layout import tryToLoad
#     all_layouts.append(tryToLoad(lay))
#
# for i,lay in enumerate(all_layouts):
#     print(i, get_layout_name(lay))
#     print(generate_distance_matrix_by_layout(lay))
#
