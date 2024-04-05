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
    cv2.rectangle(grid_visual, (0, 0), (map_width, map_height), (255, 255, 255), clearance)

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
grid_binary, grid_visual = create_map(1200, 500, 5) # This needs to be defined

# Start and goal positions
start = (50, 50, 30)  # Example start position
goal = (1150, 50, 30)  # Example goal position
L = 10  # Movement increment

# Call the a_star function
# Call the a_star function
path = a_star(start, goal, grid_binary, L, 5)

# If a path is found, visualize the path
if path:
    print("Path found:")
    for step in path:
        # Since we flip the image later, invert the y-coordinate here
        print((step[0], grid_visual.shape[0] - step[1], step[2]))

    # Flip the visual grid along the horizontal axis
    flipped_visual = cv2.flip(grid_visual, 0)

    for i in range(len(path)-1):
        # Since we flip the image later, invert the y-coordinate here
        start_point = (int(path[i][0]), grid_visual.shape[0] - int(path[i][1]))
        end_point = (int(path[i+1][0]), grid_visual.shape[0] - int(path[i+1][1]))
        # Draw lines between each pair of points
        cv2.line(flipped_visual, start_point, end_point, (0, 255, 0), 2)  # Green line for the path

    # Invert the y-coordinate of the start and goal for visualization due to flipping
    start_vis = (int(start[0]), grid_visual.shape[0] - int(start[1]))
    goal_vis = (int(goal[0]), grid_visual.shape[0] - int(goal[1]))

    # Mark the start and goal positions
    cv2.circle(flipped_visual, start_vis, 10, (255, 0, 0), -1)  # Start position in blue
    cv2.circle(flipped_visual, goal_vis, 10, (0, 0, 255), -1)  # Goal position in red
    cv2.imshow("Path", flipped_visual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No path found.")



