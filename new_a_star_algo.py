import heapq
import numpy as np
import cv2
from collections import defaultdict, namedtuple

# Define a simple class for nodes to make priority queue operations clearer
Node = namedtuple('Node', ['position', 'g', 'h', 'f', 'parent'])

def heuristic(a, b):
    # Using Euclidean distance as the heuristic function
    return np.linalg.norm(np.array(a) - np.array(b))

def create_map(map_width, map_height, clearance):
    
    # Create a binary grid for pathfinding
    grid_binary = np.zeros((map_height, map_width), dtype=np.uint8)

    # Create a visual map with colors
    grid_visual = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255

    #defining border clearance 
    cv2.rectangle(grid_binary, (0, 0), (map_width, map_height), 1, clearance)

    # clear inner part of the border to get free space
    cv2.rectangle(grid_binary, (clearance, clearance), (map_width-clearance, map_height-clearance), 0, -1)

    # Visual representation of borders with clearance
    cv2.rectangle(grid_visual, (0, 0), (map_width, map_height), (255, 0, 0), clearance)

    #rectangular obstacles into the binary grid and visual grid
    cv2.rectangle(grid_binary, (100, 100), (175, 500), 1, -1)
    cv2.rectangle(grid_visual, (100, 100), (175, 500), (255, 150, 0), -1)

    cv2.rectangle(grid_binary, (275, 0), (350, 400), 1, -1)
    cv2.rectangle(grid_visual, (275, 0), (350, 400), (150, 255, 0), -1)

    cv2.rectangle(grid_binary, (900, 50), (1100, 125), 1, -1)
    cv2.rectangle(grid_visual, (900, 50), (1100, 125), (150, 80, 0), -1)

    cv2.rectangle(grid_binary, (900, 375), (1100, 450), 1, -1)
    cv2.rectangle(grid_visual, (900, 375), (1100, 450), (150, 80, 0), -1)

    cv2.rectangle(grid_binary, (1020, 125), (1100, 375), 1, -1)
    cv2.rectangle(grid_visual, (1020, 125), (1100, 375), (150, 80, 0), -1)

    #hexagonal obstacle into the binary grid and visual grid 
    hexagon = np.array([[650, 120], [537, 185], [537, 315], [650, 380], [763, 315], [763, 185]], np.int32)
    cv2.fillPoly(grid_binary, [hexagon], 1)
    cv2.fillPoly(grid_visual, [hexagon], (0, 0, 200))


    kernel = np.ones((2*clearance+1, 2*clearance+1), np.uint8)
    grid_binary = cv2.dilate(grid_binary, kernel, iterations=1)
    
    return grid_binary, grid_visual

def get_neighbors(node, grid, L, clearance):
    neighbors = []
    for dtheta in [0, 30, 60, -30, -60]:
        new_theta = (node.position[2] + dtheta) % 360
        theta_rad = np.radians(new_theta)
        new_x = node.position[0] + L * np.cos(theta_rad)
        new_y = node.position[1] + L * np.sin(theta_rad)
        if not is_in_obstacle_or_clearance((new_x, new_y), grid, clearance):
            neighbors.append((new_x, new_y, new_theta))
    return neighbors

def is_in_obstacle_or_clearance(point, grid, clearance):
    # Checking if the point is within the obstacle map considering clearance
    x, y = int(point[0]), int(point[1])
    return grid[y-clearance:y+clearance+1, x-clearance:x+clearance+1].any()

def a_star(start, goal, grid, L, clearance):
    # Initialize both open and closed set
    open_set = []
    h_score = heuristic(start[:2], goal[:2])
    start_node = Node(position=start, g=0, h=h_score, f=0 + h_score, parent=None)
    heapq.heappush(open_set, (start_node.f, start_node))
    
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    
    closed_set = set()

    while open_set:
        _, current_node = heapq.heappop(open_set)
        
        if current_node.position in closed_set:
            continue

        if np.linalg.norm(np.array(current_node.position[:2]) - np.array(goal[:2])) <= clearance:
            # Reconstruct path
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        closed_set.add(current_node.position)

        for neighbor_pos in get_neighbors(current_node, grid, L, clearance):
            if neighbor_pos in closed_set:
                continue
    
            tentative_g_score = g_score[current_node.position] + L

            if tentative_g_score < g_score[neighbor_pos]:
                h_score = heuristic(neighbor_pos[:2], goal[:2])
                f_score = tentative_g_score + h_score
                neighbor_node = Node(position=neighbor_pos, g=tentative_g_score, h=h_score, f=f_score, parent=current_node)
                heapq.heappush(open_set, (f_score, neighbor_node))
                came_from[neighbor_pos] = current_node.position
                g_score[neighbor_pos] = tentative_g_score
             
    return None

# Assume the `create_map` function is defined elsewhere or replace this with your own map creation logic
grid_binary, grid_visual = create_map(600, 250, 5) # This needs to be defined

# Start and goal positions
start = (25, 25, 30)  # Example start position
goal = (575, 25, 30)  # Example goal position
L = 10  # Movement increment

# Call the a_star function
path = a_star(start, goal, grid_binary, L, 5)

# If a path is found, print the path
if path:
    print("Path found:")
    for step in path:
        print(step)
else:
    print("No path found.")
