from pacman import runGames, readCommand
from submission import *
from layout import getLayout

layout_list = [
    "originalClassic",
    "testClassic",
    "trappedClassic",
    "trickyClassic",
    "smallClassic",
    "minimaxClassic",
    "openClassic",
    "mediumClassic",
    "contestClassic",
    "capsuleClassic"
]
d = "4"
currentAgent = DirectionalExpectimaxAgent(depth=d)
agent_str = "DirectionalExpectimaxAgent"

curr_layout = layout_list[3]

ghost_agents = ["DirectionalGhost", "RandomGhost"]

for curr_ghost in ghost_agents:
    args = readCommand(["-n", "5", "-k", "2", "-l", curr_layout, "-p", agent_str, "-q", "-g", curr_ghost])  # Get game components based on input
    args['pacman'] = currentAgent
    avg = runGames(**args)
    print(agent_str + "," + str(d) + "," + curr_layout + "," + str(avg) + "," + str(float(currentAgent.sum_turn_time) / float(currentAgent.played)) + "," + curr_ghost)
