from pacman import runGames, readCommand
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

ghost_agents = ["DirectionalGhost", "RandomGhost"]

for curr_layout in layout_list:
    print("Current layout: ", curr_layout)
    games_counter = 10
    args = readCommand(["-n", str(games_counter), "-k", "2", "-l", curr_layout, "-p", "CompetitionAgent", "-q", "-g", ghost_agents[1]])  # Get game components based on input
    start = time()
    avg = runGames(**args)
    end = time()
    print("\n")
    print("Current layout AVG completion time: ", float(end - start) / float(games_counter))
    print("\n")
