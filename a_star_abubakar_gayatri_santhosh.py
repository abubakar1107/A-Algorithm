import heapq
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict  

def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy  

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

def validate_goal(grid_binary, goal):
    if grid_binary[goal[1], goal[0]] == 1:
        raise ValueError("Goal position not reachable: It is within an obstacle space.")

def get_neighbors(node, L):
    x, y, theta = node
    neighbors = []
    for dtheta in [0, 30, 60, -30, -60]:  # Given action set in degrees
        new_theta = (theta + dtheta) % 360  # Ensure theta stays within [0, 360)
        # Convert theta to radians for calculation
        theta_rad = np.radians(new_theta)
        # Calculate new position based on heading and step size
        new_x = x + L * np.cos(theta_rad)
        new_y = y + L * np.sin(theta_rad)
        neighbors.append((new_x, new_y, new_theta))
    return neighbors

def a_star(start, goal, grid, grid_visual, L):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start[:2], goal[:2]), 0, start))
    came_from = {}
    g_score = {start: 0}
    height = grid_visual.shape[0]
    closed_set = defaultdict(bool)  

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        x, y, _ = current

        if heuristic(current[:2], goal[:2]) <= 10: 
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path = path[::-1]

            # Draw the final path 
            for i in range(1, len(path)):
                cv2.line(grid_visual, 
                        (int(path[i-1][0]), int(path[i-1][1])), 
                        (int(path[i][0]), int(path[i][1])), 
                        (0, 255, 0), 2)
            return path, grid_visual

        closed_set[current] = True  # Mark as explored

        for neighbor in get_neighbors(current, L):
            x, y, _ = neighbor
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[int(y), int(x)] == 0:
                if neighbor in closed_set:
                    continue  # Skip already explored nodes

                tentative_g_score = current_g + L
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor[:2], goal[:2])
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

                    # Real-time visualization 
                    cv2.line(grid_visual,
                            (int(current[0]), int(current[1])),
                            (int(neighbor[0]), int(neighbor[1])),
                            (255, 0, 0), 1)
                    cv2.imshow("Explored Nodes and Path", cv2.flip(grid_visual, 0)) 

                    if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit
                        return None, grid_visual  

    return None, grid_visual  # If no path is found

# Define the grid size and create the map
map_width, map_height = 1200, 500
grid_binary, grid_visual = create_map(map_width, map_height, 5)

start = (10, 10, 10)  # Starting position with orientation
goal = (700, 400, 0)  # Goal position with orientation
L = 50  # Step size

# Visualize the exploration on a map
path, grid_visual = a_star(start, goal, grid_binary, grid_visual, L)
if path:
    print("Path found:", path)
else:
    print("No path found.")

cv2.imshow("Explored Nodes and Path", cv2.flip(grid_visual, 0))
cv2.waitKey(0)
cv2.destroyAllWindows()
