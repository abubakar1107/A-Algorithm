import heapq
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import time
import os 

def heuristic(a, b):
    """Heuristic function that returns the manhattan distance between two nodes or points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_within_obstacle(x, y, vertices):
    """Function to check if a point is within an obstacle defined by its vertices."""
    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def generate_hexagon(centre, side_length):
    angle_start = np.pi / 6  # Starting angle for hexagon vertices calculation
    return [
        (int(centre[0] + np.cos(angle_start + np.pi / 3 * i) * side_length),
         int(centre[1] + np.sin(angle_start + np.pi / 3 * i) * side_length))
        for i in range(6)
    ]

def create_map(map_width, map_height, clearance):

    """Function to create the map with specified obstacle space"""

    grid_binary = np.zeros((map_height, map_width), dtype=np.uint8)
    grid_visual = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255

    hex_center = (650//2, 250//2)
    hex_side_length = 150//2

    obstacles = [
        {'type': 'rectangle', 'vertices': [(100//2, 100//2), (175//2, 500//2)], 'color': [50, 50, 200]},
        {'type': 'rectangle', 'vertices': [(275//2, 0//2), (350//2, 400//2)], 'color': [0, 100, 100]},
        {'type': 'rectangle', 'vertices': [(900//2, 50//2), (1100//2, 125//2)], 'color': [80, 200, 100]},
        {'type': 'rectangle', 'vertices': [(900//2, 375//2), (1100//2, 450//2)], 'color': [80, 200, 100]},
        {'type': 'rectangle', 'vertices': [(1020//2, 125//2), (1100//2, 375//2)], 'color': [80, 200, 100]},
        {'type': 'polygon', 'vertices': generate_hexagon(hex_center, hex_side_length), 'color': [200, 200, 0]}
    ]

    for obstacle in obstacles:
        if obstacle['type'] == 'rectangle':
            x_min, y_min = obstacle['vertices'][0]
            x_max, y_max = obstacle['vertices'][1]
            grid_binary[y_min:y_max+1, x_min:x_max+1] = 1
            grid_visual[y_min:y_max+1, x_min:x_max+1] = obstacle['color']
        elif obstacle['type'] == 'polygon':
            for y in range(map_height):
                for x in range(map_width):
                    if is_within_obstacle(x, y, obstacle['vertices']):
                        grid_binary[y, x] = 1
                        grid_visual[y, x] = obstacle['color']

    # Apply clearance using dilation
    kernel = np.ones((2 * clearance + 1, 2 * clearance + 1), np.uint8)
    grid_binary = cv2.dilate(grid_binary, kernel, iterations=1)
    # Re-apply border after dilation
    cv2.rectangle(grid_visual, (0, 0), (map_width - 1, map_height - 1), (255, 0, 0), clearance)

    return grid_binary, grid_visual

neighbor_cache = {}  # Cache to store neighbors

def get_neighbors(node, L):

    """Function that computes the child nodes or neighbour nodes"""

    # Check if the node's neighbors are in the cache
    if node in neighbor_cache:
        return neighbor_cache[node]
    
    neighbors = []

    #calculating new nodes for the entire action set
    for dtheta in [0, 30, 60, -30, -60]:
        new_theta = (node[2] + dtheta) % 360
        theta_rad = np.radians(new_theta)
        new_x = node[0] + L * np.cos(theta_rad) 
        new_y = node[1] + L * np.sin(theta_rad)
        neighbors.append((new_x, new_y, new_theta))
    
    # Store the calculated neighbors in the cache
    neighbor_cache[node] = neighbors
    return neighbors

def orientation_difference(theta1, theta2):
    # Calculate the smallest difference between two angles
    return min(abs(theta1 - theta2), 360 - abs(theta1 - theta2))

def is_in_obstacle_or_clearance(node, grid, clearance):

    """Function to check if the start and goal node are in obstacle or clearance space"""

    x, y = int(node[0]), int(node[1])
    obstacle_space = cv2.dilate(grid, np.ones((clearance*2+1, clearance*2+1), np.uint8), iterations=1)
    return obstacle_space[y, x] == 1

def a_star(start, goal, grid, grid_visual, L, clearance=5):

    """A* logic"""

    start_time = time.time()

    # Check if start node or goal node is in an obstacle or within the clearance space
    if is_in_obstacle_or_clearance(start, grid, clearance):
        print("\n-------------------\nStart Co-ordinates and orientation are invalid\n-------------------")
        raise ValueError("Start Node is within an obstacle or the specified clearance space.")
    if is_in_obstacle_or_clearance(goal, grid, clearance):
        print("\n-------------------\nGoal Co-ordinates and orientation are invalid\n-------------------")
        raise ValueError("Goal Node is within an obstacle or the specified clearance space.")
    
    open_set = []
    heapq.heappush(open_set, (heuristic(start[:2], goal[:2]), 0, start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    orientation_threshold = 10  # Orientation threshold in Degrees
    current_batch_size = 1  
    max_batch_size = 150  # Maximum batch size for faster visualization
    nodes_explored = 0  
    updates_to_visualize = []  # Stores the updates (lines) to visualize in batches

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        # Check proximity to goal location, not orientation yet
        if heuristic(current[:2], goal[:2]) <= 10:
            
            if orientation_difference(current[2], goal[2]) <= orientation_threshold:
                # Finalize the path with orientation adjusted
                path = []
                while current in came_from:
                    rounded_node = tuple(round(coord, 2) for coord in current) 
                    path.append(rounded_node)
                    current = came_from[current]
                start_rounded = tuple(round(coord, 2) for coord in start)
                path.append(start_rounded)
                path.reverse()
                end_time = time.time()
                time_elapsed = end_time-start_time

                for update in updates_to_visualize:
                    cv2.line(grid_visual, *update, (255, 0, 0), 1)
                updates_to_visualize.clear()  # Clear the updates after visualizing

                # Plotting the final path
                for i in range(1, len(path)):
                    cv2.circle(grid_visual, start[:2], 5, (0,0,255),-1)
                    cv2.circle(grid_visual, goal[:2], 5, (0,0,0),-1)
                    cv2.line(grid_visual, (int(path[i-1][0]), int(path[i-1][1])), (int(path[i][0]), int(path[i][1])), (0, 255, 0), 2)

                cv2.imshow("Final Path", cv2.flip(grid_visual, 0))
                cv2.waitKey(0)
                
                return path, grid_visual, time_elapsed

        closed_set.add(current)

        for neighbor in get_neighbors(current, L):
            x, y, _ = neighbor
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[int(y), int(x)] == 0:
                if neighbor in closed_set:
                    continue

                tentative_g_score = current_g + L
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor[:2], goal[:2])
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

                    updates_to_visualize.append(((int(current[0]), int(current[1])), (int(neighbor[0]), int(neighbor[1]))))

        nodes_explored += 1

        
        # Batch wise visualization logic
        if len(updates_to_visualize) >= current_batch_size:
            for update in updates_to_visualize:
                cv2.circle(grid_visual, start[:2], 5, (0,0,255),-1)
                cv2.circle(grid_visual, goal[:2], 5, (0,0,0),-1)
                cv2.line(grid_visual, *update, (255, 0, 0), 1)
            cv2.imshow("Exploration in Progress", cv2.flip(grid_visual, 0))
            cv2.waitKey(1)  # Short delay for the visualization to update
            updates_to_visualize.clear()  # Clear the updates after visualizing

        
        # Adjust the batch size for visualization dynamically (linear increase of batch size)
        current_batch_size = min(max_batch_size, 1 + nodes_explored // 100)

    cv2.destroyAllWindows()
    return None, grid_visual 

if __name__ == "__main__":

    #canvas size of 600x250 units
    map_width, map_height = 600, 250
    grid_binary, grid_visual = create_map(map_width, map_height, 5)

    x_start = int(input("Enter the Start Node X coordinate:"))
    y_start = int(input("Enter the Start Node Y coordinate:"))
    theta_start = int(input("Enter the Start Node Orientation:"))
    print("\n------------------------------------\n")
    x_goal = int(input("Enter the Goal Node X coordinate:"))
    y_goal = int(input("Enter the Goal Node Y coordinate:"))
    theta_goal = int(input("Enter the Goal Node Orientation:"))

    # start = (10, 10, 10)  
    # goal = (350, 200, 45)

    start = (x_start,y_start,theta_start)
    goal = (x_goal,y_goal,theta_goal)
    L = 10  # Step size
    
    print("\n----Path calculation started----\n")

    path, grid_visual, time_elapsed = a_star(start, goal, grid_binary, grid_visual, L)
    

    if path:
        print("\n\nPath found in",time_elapsed," seconds","\n\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n\n", 
               path,"\n\n<><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n")
    else:
        print("No path found!!!")

