from pacman import runGames, readCommand
from submission import *
from time import time
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

# curr_layout = layout_list[3]

ghost_agents = ["DirectionalGhost", "RandomGhost"]

for curr_layout in layout_list:
    print("Current layout: ", curr_layout)
    currentAgent = CompetitionAgent()
    agent_str = "CompetitionAgent"
    n = 10
    args = readCommand(["-n", str(n), "-k", "2", "-l", curr_layout, "-p", agent_str, "-q", "-g", ghost_agents[1]])  # Get game components based on input
    args['pacman'] = currentAgent
    start = time()
    avg = runGames(**args)
    print("\n")
    end = time()
    print("Current layout AVG completion time: ", float(end - start) / float(n))
    print("\n")
