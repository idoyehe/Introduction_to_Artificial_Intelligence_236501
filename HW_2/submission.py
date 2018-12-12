import random, util
import numpy as np
from game import Agent


#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def __init__(self):
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
        return betterEvaluationFunction(successorGameState)
        # return scoreEvaluationFunction(successorGameState) # old evaluation function call


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

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """
    better_evaluated_score = gameState.getScore()

    pacman_pos = gameState.getPacmanPosition()
    list_ghost_pos = [ghost.configuration.pos for ghost in gameState.getGhostStates()]
    worst_manhattan_dist_to_ghost = min([util.manhattanDistance(pacman_pos, ghost_pos) for ghost_pos in list_ghost_pos])
    better_evaluated_score += worst_manhattan_dist_to_ghost

    all_food = gameState.getFood().asList()
    min_food_dist = 0
    for current_food in all_food:
        calc_manhattan = util.manhattanDistance(pacman_pos, current_food)
        if min_food_dist == 0 or calc_manhattan < min_food_dist:
            min_food_dist = calc_manhattan

    better_evaluated_score -= min_food_dist

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
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.layers_2_dev = None


######################################################################################
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

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
        self.layers_2_dev = self.depth * gameState.getNumAgents()
        return self._rb_minimax(gameState, 0, self.layers_2_dev)

    def _rb_minimax(self, game_state, agent_index, layers_number):
        """

        :param game_state: the current game state
        :param agent_index: the agent that play now
        :param layers_number: layers of number to develop
        :return: the action to preform
        """
        if game_state.isLose() or game_state.isWin() or layers_number == 0:
            return self.evaluationFunction(game_state)

        if agent_index == self.index:  # pacman agent
            current_max = float("-inf")
            chosen_action = None
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent = self._rb_minimax(successor_state, (agent_index + 1) % game_state.getNumAgents(), layers_number - 1)
                if current_max < next_agent:
                    current_max = next_agent
                    chosen_action = action

            if layers_number == self.layers_2_dev:  # to return the action to the caller
                return chosen_action
            else:
                return current_max  # return the max value to the recursive calls

        else:  # not pacman turn -> other agents means ghosts
            current_min = float("inf")
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent = self._rb_minimax(successor_state, (agent_index + 1) % game_state.getNumAgents(), layers_number - 1)
                current_min = min(current_min, next_agent)
            return current_min


######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
        self.layers_2_dev = self.depth * gameState.getNumAgents()
        return self._rb_alpha_beta(gameState, 0, self.layers_2_dev,
                                   alpha=float('-inf'), beta=float('inf'))

    def _rb_alpha_beta(self, game_state, agent_index, layers_number, alpha, beta):
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

        if agent_index == self.index:  # pacman agent
            current_max = float("-inf")
            chosen_action = None
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent = self._rb_alpha_beta(successor_state, (agent_index + 1) % game_state.getNumAgents(),
                                                 layers_number - 1, alpha=alpha, beta=beta)

                if current_max < next_agent:
                    current_max = next_agent
                    chosen_action = action
                alpha = max(current_max, alpha)
                if current_max >= beta:
                    return float("inf")

            if layers_number == self.layers_2_dev:  # to return the action to the caller
                return chosen_action
            else:
                return current_max  # return the max value to the recursive calls

        else:  # not pacman turn -> other agents means ghosts
            current_min = float("inf")
            for action in game_state.getLegalActions(agent_index):
                successor_state = game_state.generateSuccessor(agent_index, action)
                next_agent = self._rb_alpha_beta(successor_state, (agent_index + 1) % game_state.getNumAgents(),
                                                 layers_number - 1, alpha=alpha, beta=beta)
                current_min = min(current_min, next_agent)
                beta = min(current_min, beta)
                if current_min <= alpha:
                    return float("-inf")
            return current_min


######################################################################################
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

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


######################################################################################
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

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
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