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
currentAgent = RandomExpectimaxAgent(depth=d)
agent_str = "RandomExpectimaxAgent"


for curr_layout in layout_list:
    args = readCommand(["-n", "7", "-k", "2", "-l", curr_layout, "-p", agent_str, "-q"])  # Get game components based on input
    args['pacman'] = currentAgent
    avg = runGames(**args)
    print(agent_str + "," + str(d) + "," + curr_layout + "," + str(avg) + "," + str(float(currentAgent.sum_turn_time) / float(currentAgent.played)))
