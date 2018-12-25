# Dictionary for getting the layer name from layer sizes
# Key: tuple of width and height (width, height)
# Value: name of the layout - str
layout_by_size = {(19, 7): 'capsuleClassic.lay',
                  (20, 9): 'contestClassic.lay',
                  (20, 11): 'mediumClassic.lay',
                  (9, 5): 'minimaxClassic.lay',
                  (25, 9): 'openClassic.lay',
                  (28, 27): 'originalClassic.lay',
                  (20, 7): 'smallClassic.lay',
                  (5, 10): 'testClassic.lay',
                  (8, 5): 'trappedClassic.lay',
                  (20, 13): 'trickyClassic.lay'}


def get_layout_name(layout):
    return layout_by_size[(layout.width, layout.height)]


# Dictionary to get the *real* shortest maze distance
# between 2 positions
# Key: layout name
# Value: matrix with distance values - in the size of (layout width x layout height)
def generate_distance_matrix_by_layout(layout):
    n_positions = layout.height * layout.width
    dist_mat = [[None] * n_positions] * n_positions
    graph = layout_to_graph(layout)
    for x in range(layout.width):
        for y in range(layout.height):
            index = x * layout.height + y
            # if (x,y) is a wall
            if layout.walls[x][y] == True:
                dist_mat[index] = [-1]*n_positions
            else:                    # else calc distance to all other positions
                dist_mat[index] = calc_distance(graph, layout.width, layout.height, x, y)
    return dist_mat

# BFS function - calc distance from starting point (x,y) to all other positions
def calc_distance(graph, width, height, x, y):
    distance = [-1] * width * height
    distance[x*height + y] = 0
    queue = [x*height + y]
    visited = [x*height + y]
    while len(queue) != 0:
        next = queue.pop(0)
        for neighbor in graph[next]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[next] + 1
                visited.append(neighbor)
                queue.append(neighbor)
    return distance


def layout_to_graph(layout):
    graph={}
    for x in range(layout.width):
        for y in range(layout.height):
            connected = []
            index = x * layout.height + y
            if layout.walls[x][y] == False:
                if layout.walls[ x + 1][y] == False:
                    connected.append((x + 1) * layout.height + y)
                if layout.walls[ x - 1][y] == False:
                    connected.append((x - 1) * layout.height + y)
                if layout.walls[x][y + 1 ] == False:
                    connected.append( x * layout.height + y + 1)
                if layout.walls[x][y - 1 ] == False:
                    connected.append( x * layout.height + y - 1)
            graph[index] = connected
    return graph


all_layouts_path = ['.\\layouts\\capsuleClassic.lay', '.\\layouts\\contestClassic.lay', '.\\layouts\\mediumClassic.lay', '.\\layouts\\minimaxClassic.lay', '.\\layouts\\openClassic.lay', '.\\layouts\\originalClassic.lay', '.\\layouts\\smallClassic.lay', '.\\layouts\\testClassic.lay', '.\\layouts\\trappedClassic.lay', '.\\layouts\\trickyClassic.lay']

all_layouts = []
for lay in all_layouts_path:
    from layout import tryToLoad
    all_layouts.append(tryToLoad(lay))


for i, lay in enumerate(all_layouts):
    print(i, get_layout_name(lay))
    print(generate_distance_matrix_by_layout(lay))

f = open(".\\docs\\BFS_layouts_results.txt", 'a')

for i, lay in enumerate(all_layouts):
    f.write(get_layout_name(lay)+'\n')
    m = generate_distance_matrix_by_layout(lay)
    d = {}
    for u in range(len(m)):
        for v in range(len(m)):
            if m[u][v] != -1 and u <= v:
                d[(u,v)] = m[u][v]
    f.write(str(d)+'\n')

f.close()
