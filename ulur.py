import numpy as np
import cv2
import heapq
from collections import defaultdict

# Robot parameters and map dimensions
R = 0.033  # Radius of wheels in meters
L = 0.160  # Distance between wheels in meters
robot_diameter_mm = 306  # Approximation to the larger dimension in mm for clearance calculation
user_clearance_mm = 50  # Additional clearance
total_clearance_mm = int((robot_diameter_mm / 2) + user_clearance_mm)
map_width, map_height = 6000, 2000  # Map dimensions in pixels (assuming 1 pixel = 1 mm for simplicity)
R1 = 50
R2 = 150

# Action set based on RPM inputs (simplified for demonstration)
actions = [[0, R1], [R1, 0], [R1, R1], [0, R2], [R2, 0], [R2, R2], [R1, R2], [R2, R1]]

def create_map(width, height, clearance):
    obstacle_map = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    cv2.rectangle(obstacle_map, (1500 - clearance, 0 - clearance), (1750 + clearance, 1000 + clearance), (255,0,0), -1)
    cv2.rectangle(obstacle_map, (2500 - clearance, 1000 - clearance), (2750 + clearance, 2000 + clearance), (0,255,0), -1)
    cv2.circle(obstacle_map, (4200, 800), 600 + clearance, (0, 0, 255), -1)
    return obstacle_map

def calculate_new_position(x, y, theta, UL, UR, dt=1):
    Vl = UL * (2 * np.pi * R) / 60
    Vr = UR * (2 * np.pi * R) / 60
    Dx = (Vl + Vr) / 2 * np.cos(np.radians(theta)) * dt
    Dy = (Vl + Vr) / 2 * np.sin(np.radians(theta)) * dt
    Dtheta = (Vr - Vl) / L * dt
    return x + Dx * 1000, y + Dy * 1000, (theta + np.degrees(Dtheta)) % 360

def is_collision_free(x, y, obstacle_map):
    if 0 <= x < obstacle_map.shape[1] and 0 <= y < obstacle_map.shape[0]:
        # Check if the pixel is white, indicating free space.
        return np.all(obstacle_map[int(y), int(x)] == [255, 255, 255])
    return False

def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def resize_map_for_display(map_img, scale_percent=20):
    width = int(map_img.shape[1] * scale_percent / 100)
    height = int(map_img.shape[0] * scale_percent / 100)
    return cv2.resize(map_img, (width, height), interpolation=cv2.INTER_AREA)

def a_star(start, goal, actions, base_obstacle_map, visualization=False):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if np.linalg.norm(np.array(current[:2]) - np.array(goal[:2])) < 100:
            path, actions = reconstruct_path(came_from, current)
            if visualization:
                visualize_path(base_obstacle_map, path, start, goal)
            return path, actions  # Now also returns actions

        for action in actions:
            neighbor = calculate_new_position(current[0], current[1], current[2], action[0], action[1])
            if not is_collision_free(neighbor[0], neighbor[1], base_obstacle_map):
                continue  # Skip this neighbor if it leads to a collision
            tentative_g_score = g_score[current] + np.linalg.norm(np.array(neighbor[:2]) - np.array(current[:2]))
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] =  (current, action)
                g_score[neighbor] = tentative_g_score
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor, goal), neighbor))


        if visualization and len(came_from) % 50 == 0:  # Conditional visualization to reduce overhead
            visualize_exploration(base_obstacle_map, came_from, current)

    return []

def reconstruct_path(came_from, current):
    path = []
    actions = []
    while current in came_from:
        current, action = came_from[current]
        path.append(current)
        actions.append(action)
    return path[::-1], actions[::-1]


def visualize_exploration(base_obstacle_map, came_from, current):
    vis_map = base_obstacle_map.copy()
    for node, parent_action_tuple in came_from.items():
        parent, _action = parent_action_tuple  # Unpack the tuple to get the parent node and ignore the action
        cv2.line(vis_map, (int(parent[0]), int(parent[1])), (int(node[0]), int(node[1])), (0, 0, 0), 1)
    cv2.circle(vis_map, (int(current[0]), int(current[1])), 10, color=(0, 0, 0), thickness=-1)

    resized_vis_map = resize_map_for_display(vis_map)
    cv2.imshow('Exploration', resized_vis_map)
    cv2.waitKey(1)



def visualize_path(base_obstacle_map, path, start, goal):
    # Visualize the final path
    vis_map = resize_map_for_display(base_obstacle_map)
    for i in range(len(path) - 1):
        cv2.line(vis_map, (int(path[i][0]), int(path[i][1])), (int(path[i+1][0]), int(path[i+1][1])), (255, 0, 0), 5)
    cv2.imshow('Final Path', vis_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    obstacle_map = create_map(map_width, map_height, total_clearance_mm)
    start = (500, 1000, 0)
    goal = (2250, 700, 0)
    path, actions = a_star(start, goal, actions, obstacle_map, visualization=True)
    visualize_path(obstacle_map, path, start, goal)
    if path:
        print("Path found:", path)
        print("\n<><><><><><><><><><><><><><><><><><><><><><>\n")
        print(actions)
    else:
        print("No path found.")
