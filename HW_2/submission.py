import util
from game import Actions
from game import Agent
from random import choice, shuffle


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
        chosenIndex = choice(bestIndices)  # Pick randomly among the best
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

    def nextTurnFunction(self, agent_index):
        return (agent_index + 1) % self.total_game_agents

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

        next_agent_index = self.nextTurnFunction(agent_index)

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

        next_agent_index = self.nextTurnFunction(agent_index)

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

# e: implementing random expectimax
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

        next_agent_index = self.nextTurnFunction(agent_index)

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

        next_agent_index = self.nextTurnFunction(agent_index)

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


##############################################################################################################################
# implementing competition agent

layout_height = -1
realDistDict = {}


class CompetitionAgent(MultiAgentSearchAgent):
    """
        Our competition agent use different heuristic than all other agent
    """

    def __init__(self, evalFn='competitionAgentHeuristic', depth='4'):
        super().__init__(evalFn=evalFn, depth=depth)
        self.distanceCalculationFunction = layoutRealDist
        self.capsules_total = -1
        self.ghost_factor = 5  # default value
        self.game_layout = "Unknown -> Generic"
        self.preprocessing = True

    def knownLayoutsRecognizer(self, layout_map):
        layout_map_dict = {
            "capsuleClassic": ["%%%%%%%%%%%%%%%%%%%",
                               "%G.       G   ....%",
                               "%.% % %%%%%% %.%%.%",
                               "%.%o% %   o% %.o%.%",
                               "%.%%%.%  %%% %..%.%",
                               "%.....  P    %..%G%",
                               "%%%%%%%%%%%%%%%%%%%%"],

            "contestClassic": ["%%%%%%%%%%%%%%%%%%%%",
                               "%o...%........%...o%",
                               "%.%%.%.%%..%%.%.%%.%",
                               "%...... G GG%......%",
                               "%.%.%%.%% %%%.%%.%.%",
                               "%.%....% ooo%.%..%.%",
                               "%.%.%%.% %% %.%.%%.%",
                               "%o%......P....%....%",
                               "%%%%%%%%%%%%%%%%%%%%"],

            "mediumClassic": ["%%%%%%%%%%%%%%%%%%%%",
                              "%o...%........%....%",
                              "%.%%.%.%%%%%%.%.%%.%",
                              "%.%..............%.%",
                              "%.%.%%.%%  %%.%%.%.%",
                              "%......%G  G%......%",
                              "%.%.%%.%%%%%%.%%.%.%",
                              "%.%..............%.%",
                              "%.%%.%.%%%%%%.%.%%.%",
                              "%....%...P....%...o%",
                              "%%%%%%%%%%%%%%%%%%%%"],

            "minimaxClassic": ["%%%%%%%%%",
                               "%.P    G%",
                               "% %.%G%%%",
                               "%G    %%%",
                               "%%%%%%%%%"],

            "openClassic": ["%%%%%%%%%%%%%%%%%%%%%%%%%",
                            "%.. P  ....      ....   %",
                            "%..  ...  ...  ...  ... %",
                            "%..  ...  ...  ...  ... %",
                            "%..    ....      .... G %",
                            "%..  ...  ...  ...  ... %",
                            "%..  ...  ...  ...  ... %",
                            "%..    ....      ....  o%",
                            "%%%%%%%%%%%%%%%%%%%%%%%%%"],

            "originalClassic": ["%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
                                "%............%%............%",
                                "%.%%%%.%%%%%.%%.%%%%%.%%%%.%",
                                "%o%%%%.%%%%%.%%.%%%%%.%%%%o%",
                                "%.%%%%.%%%%%.%%.%%%%%.%%%%.%",
                                "%..........................%",
                                "%.%%%%.%%.%%%%%%%%.%%.%%%%.%",
                                "%.%%%%.%%.%%%%%%%%.%%.%%%%.%",
                                "%......%%....%%....%%......%",
                                "%%%%%%.%%%%% %% %%%%%.%%%%%%",
                                "%%%%%%.%%%%% %% %%%%%.%%%%%%",
                                "%%%%%%.%            %.%%%%%%",
                                "%%%%%%.% %%%%  %%%% %.%%%%%%",
                                "%     .  %G  GG  G%  .     %",
                                "%%%%%%.% %%%%%%%%%% %.%%%%%%",
                                "%%%%%%.%            %.%%%%%%",
                                "%%%%%%.% %%%%%%%%%% %.%%%%%%",
                                "%............%%............%",
                                "%.%%%%.%%%%%.%%.%%%%%.%%%%.%",
                                "%.%%%%.%%%%%.%%.%%%%%.%%%%.%",
                                "%o..%%.......  .......%%..o%",
                                "%%%.%%.%%.%%%%%%%%.%%.%%.%%%",
                                "%%%.%%.%%.%%%%%%%%.%%.%%.%%%",
                                "%......%%....%%....%%......%",
                                "%.%%%%%%%%%%.%%.%%%%%%%%%%.%",
                                "%.............P............%",
                                "%%%%%%%%%%%%%%%%%%%%%%%%%%%%"],

            "smallClassic": ["%%%%%%%%%%%%%%%%%%%%",
                             "%......%G  G%......%",
                             "%.%%...%%  %%...%%.%",
                             "%.%o.%........%.o%.%",
                             "%.%%.%.%%%%%%.%.%%.%",
                             "%........P.........%",
                             "%%%%%%%%%%%%%%%%%%%%"],

            "testClassic": ["%%%%%",
                            "% . %",
                            "%.G.%",
                            "% . %",
                            "%. .%",
                            "%   %",
                            "%  .%",
                            "%   %",
                            "%P .%",
                            "%%%%%"],

            "trappedClassic": ["%%%%%%%%",
                               "%   P G%",
                               "%G%%%%%%",
                               "%....  %",
                               "%%%%%%%%"],

            "trickyClassic": ["%%%%%%%%%%%%%%%%%%%%",
                              "%o...%........%...o%",
                              "%.%%.%.%%..%%.%.%%.%",
                              "%.%.....%..%.....%.%",
                              "%.%.%%.%%  %%.%%.%.%",
                              "%...... GGGG%.%....%",
                              "%.%....%%%%%%.%..%.%",
                              "%.%....%  oo%.%..%.%",
                              "%.%....% %%%%.%..%.%",
                              "%.%...........%..%.%",
                              "%.%%.%.%%%%%%.%.%%.%",
                              "%o...%...P....%...o%",
                              "%%%%%%%%%%%%%%%%%%%%"]
        }

        layout_tunning = {
            "capsuleClassic": {"ghostFactor": 7, "depth": 4},
            "contestClassic": {"ghostFactor": 7, "depth": 4},
            "mediumClassic": {"ghostFactor": 5, "depth": 4},
            "minimaxClassic": {"ghostFactor": 2, "depth": 4},
            "openClassic": {"ghostFactor": 1, "depth": 3},
            "originalClassic": {"ghostFactor": 5, "depth": 3},
            "smallClassic": {"ghostFactor": 4, "depth": 3},
            "testClassic": {"ghostFactor": 1, "depth": 4},
            "trappedClassic": {"ghostFactor": 1, "depth": 4},
            "trickyClassic": {"ghostFactor": 6, "depth": 3},
        }

        for iter_name, iter_map in layout_map_dict.items():
            if iter_map == layout_map:
                self.game_layout = iter_name
                self.ghost_factor = layout_tunning[self.game_layout]["ghostFactor"]
                self.depth = layout_tunning[self.game_layout]["depth"]
                return True
        return False

    def getAction(self, gameState):
        """
        Returns the action using self.depth and self.evaluationFunction
        """
        if self.preprocessing:  # initialize in first action ONLY
            walls = gameState.getWalls()
            generateDistanceDictByLayout(walls, walls.width, walls.height)
            self.knownLayoutsRecognizer(gameState.data.layout.layoutText)
            self.capsules_total = len(gameState.getCapsules())
            self.total_game_agents = gameState.getNumAgents()
            self.layers_developed = self.depth * self.total_game_agents
            self.preprocessing = False

        action_ordered_list = list()
        legal_moves = gameState.getLegalPacmanActions()

        pacman_dir = gameState.getPacmanState().configuration.direction
        if pacman_dir in legal_moves:
            action_ordered_list.append(pacman_dir)
            legal_moves.remove(pacman_dir)
            action_ordered_list += legal_moves
        else:
            action_ordered_list = legal_moves

        next_agent_index = self.nextTurnFunction(self.index)
        current_max = float("-inf")
        alpha = float('-inf')
        for action in action_ordered_list:
            successor_state = gameState.generateSuccessor(self.index, action)
            next_agent_value = self._rb_alpha_beta_(successor_state, next_agent_index, self.layers_developed - 1,
                                                    alpha=alpha, beta=float('inf'))
            if current_max < next_agent_value:
                alpha = next_agent_value
                current_max = next_agent_value
                self.next_action = action

        if alpha <= gameState.getScore() - 500:  # when pacman decided to commit suicide
            plan_b = [action for action in gameState.getLegalPacmanActions() if action != self.next_action]
            if len(plan_b) > 0:  # there is plan B
                return choice(plan_b)

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
            return self.evaluationFunction(game_state, self.distanceCalculationFunction, self.capsules_total, self.ghost_factor)

        next_agent_index = self.nextTurnFunction(agent_index)

        if agent_index == self.index:  # pacman agent
            current_max = float("-inf")
            for action in game_state.getLegalPacmanActions():
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_alpha_beta_(successor_state, next_agent_index, layers_number - 1, alpha=alpha, beta=beta)
                current_max = max(current_max, next_agent_value)
                alpha = max(current_max, alpha)
                if current_max >= beta:
                    return float("inf")

            return current_max  # return the max value to the recursive calls

        else:  # not pacman turn -> other agents means ghosts
            current_min = float("inf")
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent_value = self._rb_alpha_beta_(successor_state, next_agent_index, layers_number - 1, alpha=alpha, beta=beta)

                current_min = min(current_min, next_agent_value)
                beta = min(current_min, beta)
                if current_min <= alpha:
                    return float("-inf")
            return current_min


def competitionAgentHeuristic(gameState, distCalculationFunction, capsules_total, ghost_factor):
    """
    The competitionAgentHeuristic takes in a GameState (pacman.py) and pacman current direction should return a number to evalute the given game state,
    where higher numbers are better.
    """
    current_state_score = 100 * gameState.getScore()  # give more weight to score than other parameters

    pacman_pos = gameState.getPacmanPosition()

    # First Parameter Ghost:
    ghosts_evaluation = 0
    for ghost in gameState.getGhostStates():
        pacman_ghost_dist = distCalculationFunction(pacman_pos, ghost.configuration.pos)
        if ghost.scaredTimer >= pacman_ghost_dist:  # Ghost Scared
            ghosts_evaluation -= 50 * pacman_ghost_dist

        elif pacman_ghost_dist > ghost_factor:
            ghosts_evaluation += 50

    # Second Parameter Food:
    food_list = gameState.getFood().asList() + gameState.getCapsules()
    food_collection_trace = 0

    for food_pos in food_list:
        food_collection_trace += distCalculationFunction(food_pos, pacman_pos)

    food_evaluation = 0
    if len(food_list) > 0:
        food_evaluation = - (food_collection_trace / len(food_list))

    # Third Parameter Capsules:
    capsules = gameState.getCapsules()
    capsules_evaluation = 10000 * (capsules_total - len(capsules))

    return current_state_score + ghosts_evaluation + food_evaluation + capsules_evaluation


def parseLayout2Graph(walls, width, height):
    graph_dict = {}
    for x in range(width):
        for y in range(height):
            connected = list()
            index = x * height + y
            if not walls[x][y]:
                if not walls[x + 1][y]:
                    connected.append((x + 1) * height + y)
                if not walls[x - 1][y]:
                    connected.append((x - 1) * height + y)
                if not walls[x][y + 1]:
                    connected.append(x * height + y + 1)
                if not walls[x][y - 1]:
                    connected.append(x * height + y - 1)
            graph_dict[index] = connected
    return graph_dict


def xyBFS(graph, width, height, x, y):
    distance = [-1] * width * height
    distance[x * height + y] = 0
    queue = [x * height + y]
    while len(queue) != 0:
        next_vertex = queue.pop(0)
        for neighbor in graph[next_vertex]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[next_vertex] + 1
                queue.append(neighbor)
    return distance


def generateDistanceDictByLayout(walls, width, height):
    global layout_height, realDistDict
    layout_height = height
    n_positions = height * width
    dist_mat = [[None] * n_positions] * n_positions
    graph = parseLayout2Graph(walls, width, height)
    for x in range(width):
        for y in range(height):
            index = x * height + y
            # if (x,y) is a wall
            if walls[x][y]:
                dist_mat[index] = [-1] * n_positions
            else:  # else calc distance to all other positions
                dist_mat[index] = xyBFS(graph, width, height, x, y)
    realDistDict = {}
    for u in range(len(dist_mat)):
        for v in range(len(dist_mat)):
            if dist_mat[u][v] != -1 and u <= v:
                realDistDict[(u, v)] = dist_mat[u][v]


def layoutRealDist(xy1, xy2):
    index1 = int(xy1[1]) + int(xy1[0]) * layout_height
    index2 = int(xy2[1]) + int(xy2[0]) * layout_height
    key = (index1, index2) if (index1 < index2) else (index2, index1)
    default = util.manhattanDistance(xy1, xy2)
    return realDistDict.get(key, default)  # shouldn't return -1 just for edge case
