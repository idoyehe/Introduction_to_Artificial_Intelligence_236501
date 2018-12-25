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
            index = y*layout.width + x
            # if (x,y) is a wall
            if layout.layoutText[y][x] == '%':
                dist_mat[index] = [-1]*n_positions
            else:                    # else calc distance to all other positions
                dist_mat[index] = calc_distance(graph, layout.width, layout.height, x, y)
    return dist_mat

# BFS function - calc distance from starting point (x,y) to all other positions
def calc_distance(graph, width, height, x, y):
    distance = [-1] * width * height
    distance[y*width + x] = 0
    queue = [y*width + x]
    visited = []
    while len(queue) != 0:
        next = queue.pop(0)
        visited.append(next)
        for neighbor in graph[next]:
            if distance[neighbor] == -1 or distance[neighbor] > distance[next]+1:
                distance[neighbor] = distance[next]+1
                if neighbor not in visited:
                    queue.append(neighbor)
    return distance


def layout_to_graph(layout):
    graph={}
    for x in range(layout.width):
        for y in range(layout.height):
            connected = []
            index = y*layout.width + x
            if layout.layoutText[y][x] != '%':
                if layout.layoutText[y][x+1] != '%':
                    connected.append(y * layout.width + x+1)
                if layout.layoutText[y][x-1] != '%':
                    connected.append(y * layout.width + x-1)
                if layout.layoutText[y+1][x] != '%':
                    connected.append((y+1) * layout.width + x)
                if layout.layoutText[y-1][x] != '%':
                    connected.append((y-1) * layout.width + x)
            graph[index] = connected
    return graph


all_layouts_path = ['HW_2\\layouts\\capsuleClassic.lay', 'HW_2\\layouts\\contestClassic.lay', 'HW_2\\layouts\\mediumClassic.lay', 'HW_2\\layouts\\minimaxClassic.lay', 'HW_2\\layouts\\openClassic.lay', 'HW_2\\layouts\\originalClassic.lay', 'HW_2\\layouts\\smallClassic.lay', 'HW_2\\layouts\\testClassic.lay', 'HW_2\\layouts\\trappedClassic.lay', 'HW_2\\layouts\\trickyClassic.lay']

all_layouts = []
for lay in all_layouts_path:
    from layout import tryToLoad
    all_layouts.append(tryToLoad(lay))


for i,lay in enumerate(all_layouts):
    print(i, get_layout_name(lay))
    print(generate_distance_matrix_by_layout(lay))
