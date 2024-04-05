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

# Action set based on RPM inputs (simplified for demonstration)
actions = [[0, 30], [30, 0], [30, 30], [0, 60], [60, 0], [60, 60], [30, 60], [60, 30]]

# Helper functions
def create_map(width, height, clearance):
    obstacle_map = np.ones((height, width), dtype=np.uint8) * 255  # White background
    # Expanded obstacles to consider the robot's dimensions and clearance
    cv2.rectangle(obstacle_map, (1500 - clearance, 0 - clearance), (1750 + clearance, 1000 + clearance), (0,0,0), -1)
    cv2.rectangle(obstacle_map, (2500 - clearance, 1000 - clearance), (2750 + clearance, 2000 + clearance), (0,0,0), -1)
    cv2.circle(obstacle_map, (4200, 800), 600 + clearance, (0,0,0), -1)
    return obstacle_map

def calculate_new_position(x, y, theta, UL, UR, dt=1):
    Vl = UL * (2 * np.pi * R) / 60  # Linear velocity of left wheel in m/s
    Vr = UR * (2 * np.pi * R) / 60  # Linear velocity of right wheel in m/s
    Dx = (Vl + Vr) / 2 * np.cos(np.radians(theta)) * dt
    Dy = (Vl + Vr) / 2 * np.sin(np.radians(theta)) * dt
    Dtheta = (Vr - Vl) / L * dt
    return x + Dx * 1000, y + Dy * 1000, (theta + np.degrees(Dtheta)) % 360  # Convert meters to mm

def is_collision_free(x, y, obstacle_map):
    """Checks if the position (x, y) is free of collisions."""
    if 0 <= x < obstacle_map.shape[1] and 0 <= y < obstacle_map.shape[0]:
        return obstacle_map[int(y), int(x)] == 255  # Check if the pixel is white (255)
    return False


def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def resize_map_for_display(map_img, scale_percent=20):
    """
    Resize the map image based on a scaling percentage.
    :param map_img: The original map image to be resized.
    :param scale_percent: The percentage of the original size.
    :return: Resized map image.
    """
    width = int(map_img.shape[1] * scale_percent / 100)
    height = int(map_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(map_img, dim, interpolation=cv2.INTER_AREA)
    return resized

# A* Algorithm
# Modify the A* Algorithm to incorporate resized visualization
def a_star(start, goal, actions, base_obstacle_map, visualization=False):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))  # Initialize with start node
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if visualization:
            vis_map = base_obstacle_map.copy()  # Work on a copy for visualization
            cv2.circle(vis_map, (int(current[0]), int(current[1])), radius=20, color=(0, 0, 255), thickness=-1)  # Adjusted radius for visibility
            resized_vis_map = resize_map_for_display(vis_map)  # Resize for display
            cv2.imshow('Exploration', resized_vis_map)
            cv2.waitKey(1)

        if heuristic(current, goal) < 100:  # Adjusted for scaled visualization
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            if visualization:
                for point in path:
                    cv2.circle(vis_map, (int(point[0]), int(point[1])), radius=20, color=(0, 255, 0), thickness=-1)  # Adjusted radius for visibility
                resized_vis_map = resize_map_for_display(vis_map)  # Resize for display
                cv2.imshow('Final Path', resized_vis_map)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return path[::-1]

        for action in actions:
            neighbor = calculate_new_position(current[0], current[1], current[2], action[0], action[1])
            if not is_collision_free(neighbor[0], neighbor[1], base_obstacle_map):  # Use base map for collision checks
                continue

            tentative_g_score = current_g + heuristic(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

    return []

# Visualization
def draw_path(path, obstacle_map):

    rs_map = resize_map_for_display(obstacle_map, scale_percent=20)
    for i in range(len(path) - 1):
        cv2.line(obstacle_map, (int(path[i][0]), int(path[i][1])), (int(path[i+1][0]), int(path[i+1][1])), (255, 0, 0), 2)
    cv2.imshow('Path', rs_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    start = (500, 1000, 0)  # Start position (x, y, theta)
    goal = (1290, 300, 0)  # Goal position (x, y, not considering theta for goal)
    obstacle_map = create_map(map_width, map_height, total_clearance_mm)  # Create the obstacle map with adjusted clearance
    path = a_star(start, goal, actions, obstacle_map, visualization=True)

    if path:
        print("Path found. Drawing path...")
        draw_path(path, obstacle_map)  # Ensure path drawing accounts for map scaling if used
    else:
        print("No path found.")
