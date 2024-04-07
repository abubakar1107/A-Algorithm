#!/usr/bin/env python3

import numpy as np
import cv2
import heapq

# Robot parameters and map dimensions
R = 33  # Radius of wheels in meters
L = 160  # Distance between wheels in meters
robot_diameter_mm = 306  # Approximation to the larger dimension in mm for clearance calculation
user_clearance_mm = 50  # Additional clearance
total_clearance_mm = int((robot_diameter_mm / 2) + user_clearance_mm)
map_width, map_height = 6000, 2000  # Map dimensions in pixels (assuming 1 pixel = 1 mm for simplicity)
R1 = 50
R2 = 50
actions = [[0, R1], [R1, 0], [R1, R1], [0, R2], [R2, 0], [R2, R2], [R1, R2], [R2, R1]]
# Weight for the heuristic function
HEURISTIC_WEIGHT = 2 # Typical values might range from 1 to 2

def create_map(width, height, clearance):
    obstacle_map = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Add obstacles with clearance
    cv2.rectangle(obstacle_map, (1500 - clearance, 0 - clearance), (1750 + clearance, 1000 + clearance), (255,0,0), -1)
    cv2.rectangle(obstacle_map, (2500 - clearance, 1000 - clearance), (2750 + clearance, 2000 + clearance), (0,255,0), -1)
    cv2.circle(obstacle_map, (4200, 800), 600 + clearance, (0, 0, 255), -1)
    
    # Add clearance to the walls of the map
    # Top border
    cv2.rectangle(obstacle_map, (0, 0), (width, clearance//10), (0,0,255), -1)
    # Bottom border
    cv2.rectangle(obstacle_map, (0, height - (clearance//10)), (width, height), (0,0,255), -1)
    # Left border
    cv2.rectangle(obstacle_map, (0, 0), (clearance//10, height), (0,0,255), -1)
    # Right border
    cv2.rectangle(obstacle_map, (width - (clearance//10), 0), (width, height), (0,0,255), -1)

    return obstacle_map

def calculate_new_position(x, y, theta, UL, UR, dt=0.1):
    t= 0

    while t < 1:
        Vl = UL * (2 * np.pi * R) / 60
        Vr = UR * (2 * np.pi * R) / 60
        Dx = (Vl + Vr) / 2 * np.cos(np.radians(theta)) * dt
        Dy = (Vl + Vr) / 2 * np.sin(np.radians(theta)) * dt
        Dtheta = (Vr - Vl) / L * dt
        x = x + Dx
        y = y + Dy
        theta = (theta + np.degrees(Dtheta)) % 360
        t = t + dt
  
    return x + Dx, y + Dy , (theta + np.degrees(Dtheta)) % 360

def is_trajectory_collision_free(xi, yi, thetai, UL, UR, obstacle_map, dt=0.1):
    """
    Checks if the trajectory from the current position (xi, yi, thetai) using the given actions (UL, UR)
    is free from collisions, focusing on start, end, and midpoint of the trajectory.

    Parameters remain the same.
    """
    # Calculate end position of the trajectory
    xn, yn, _ = calculate_new_position(xi, yi, thetai, UL, UR, dt)

    # Midpoint of the trajectory
    xm, ym, _ = calculate_new_position(xi, yi, thetai, UL, UR, dt/2)

    # Check start, midpoint, and end positions for collision
    if not is_collision_free(xi, yi, obstacle_map) or \
       not is_collision_free(xn, yn, obstacle_map) or \
       not is_collision_free(xm, ym, obstacle_map):
        return False

    return True



def is_collision_free(x, y, obstacle_map):
    """
    Checks if the given position (x, y) is free from collisions, considering the obstacle map.

    Parameters:
    - x, y: Position coordinates to check.
    - obstacle_map: The map with obstacles.

    Returns:
    - True if the position is collision-free, False otherwise.
    """
    if 0 <= x < obstacle_map.shape[1] and 0 <= y < obstacle_map.shape[0]:
        return np.all(obstacle_map[int(y), int(x)] == [255, 255, 255])
    return False

def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def rpm_to_velocity(actions, R, L):
    """
    Convert RPM actions to [linear_x, angular_z] format.

    Parameters:
    - actions: List of [RPM_l, RPM_r] actions.
    - R: Radius of the wheels in meters.
    - L: Distance between the wheels in meters.

    Returns:
    - List of [linear_x, angular_z] actions.
    """
    converted_actions = []
    for action in actions:
        RPM_l, RPM_r = action
        # Convert RPM to linear velocity in meters per second
        V_l = RPM_l * (2 * np.pi * R) / 60
        V_r = RPM_r * (2 * np.pi * R) / 60
        # Calculate linear and angular velocities
        linear_x = (V_r + V_l) / 2
        angular_z = (V_r - V_l) / L
        converted_actions.append([linear_x, angular_z])
    
    return converted_actions

def validate_position(x, y, obstacle_map):
    """
    Check if a given position is within the allowed space.
    """
    if x < 0 or y < 0 or x >= obstacle_map.shape[1] or y >= obstacle_map.shape[0]:
        return False  # Position is out of map bounds
    return np.all(obstacle_map[int(y), int(x)] == [255, 255, 255])

def a_star(start, goal, actions, base_obstacle_map, visualization=True):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0}
    visited = set()  # Set to store visited nodes
    
    THRESHOLD_DISTANCE = 100  # Set a threshold distance to the goal point
    
    # Inside the while loop of the a_star function
    while open_set:
        current_f, current = heapq.heappop(open_set)
    
        # Check if the current node is close enough to the goal
        if np.linalg.norm(np.array(current[:2]) - np.array(goal[:2])) < THRESHOLD_DISTANCE:
            # Check if the current node is within the threshold distance of the goal
            if not is_collision_free(current[0], current[1], base_obstacle_map):
                continue  # Skip this node if it's not collision-free
            path, actions = reconstruct_path(came_from, current)
            if visualization:
                visualize_path(base_obstacle_map, path, start, goal)
            return path, actions
        
        visited.add(current)  # Mark current node as visited
        
        for action in actions:
            neighbor = calculate_new_position(current[0], current[1], current[2], action[0], action[1])
            
            # Skip this neighbor if it's already visited
            if neighbor in visited or not is_trajectory_collision_free(current[0], current[1], current[2], action[0], action[1], base_obstacle_map):
                continue
            
            if not is_collision_free(neighbor[0], neighbor[1], base_obstacle_map):
                continue
            
            tentative_g_score = g_score[current] + np.linalg.norm(np.array(neighbor[:2]) - np.array(current[:2]))
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = (current, action)
                g_score[neighbor] = tentative_g_score
                # Apply the heuristic weight
                f_score = tentative_g_score + heuristic(neighbor, goal) * HEURISTIC_WEIGHT
                heapq.heappush(open_set, (f_score, neighbor))

        
        if visualization and len(came_from) % 50 == 0:
            visualize_exploration(base_obstacle_map, came_from, current)
            

    return [], []

def plot_actual_curve_on_map(canvas, Xi, Yi, Thetai, UL, UR, color=(0, 0, 0), thickness=2):
    """
    Simulates and draws the actual curve taken by the robot from a parent node to a child node,
    based on differential drive kinematics.

    Parameters:
    - canvas: The map canvas as a numpy array.
    - Xi, Yi: Initial position coordinates in pixels.
    - Thetai: Initial orientation angle in degrees.
    - UL, UR: Left and right wheel speeds (RPM).
    - color: The color of the curve.
    - thickness: The thickness of the curve.
    """
    dt = 0.1  # Time step for simulation
    t = 0  # Current simulation time
    duration = 1.0  # Total duration of the action

    # Convert initial orientation to radians for calculations
    Thetai_rad = np.radians(Thetai)

    while t < duration:
        t += dt
        # Calculate velocities
        Vl = UL * (2 * np.pi * R) / 60
        Vr = UR * (2 * np.pi * R) / 60
        # Calculate change in position
        Vx = (Vr + Vl) / 2.0
        Vy = 0  # No lateral movement in differential drive
        # Calculate change in orientation
        omega = (Vr - Vl) / L

        # Update position and orientation
        Dx = Vx * np.cos(Thetai_rad) * dt
        Dy = Vx * np.sin(Thetai_rad) * dt
        Dtheta = omega * dt

        Xn = Xi + Dx
        
        Yn = Yi + Dy
        Thetan = Thetai_rad + Dtheta

        # Draw segment
        cv2.line(canvas, (int(Xi), int(Yi)), (int(Xn), int(Yn)), color, thickness)

        # Update for next iteration
        Xi, Yi, Thetai_rad = Xn, Yn, Thetan

# Example usage within the visualization function
def visualize_exploration(base_obstacle_map, came_from, current):
    vis_map = base_obstacle_map.copy()
    for node, (parent, action) in came_from.items():
        plot_actual_curve_on_map(vis_map, parent[0], parent[1], parent[2], action[0], action[1], color=(255, 0,255 ), thickness=2)

    # Highlight the current node
    cv2.circle(vis_map, (int(current[0]), int(current[1])), 10, color=(0, 255, 0), thickness=-1)

    resized_vis_map = resize_map_for_display(vis_map)
    cv2.imshow('Exploration with Actual Curves', resized_vis_map)
    cv2.waitKey(1)

def visualize_path(base_obstacle_map, path, start, goal):
    # Visualize the final path
    vis_map = base_obstacle_map.copy()
    scale_percent = 20
    resized_vis_map = resize_map_for_display(vis_map, scale_percent)
    for i in range(len(path) - 1):
        # Scale path coordinates according to the resized map
        scaled_start = (int(path[i][0] * scale_percent / 100), int(path[i][1] * scale_percent / 100))
        scaled_end = (int(path[i+1][0] * scale_percent / 100), int(path[i+1][1] * scale_percent / 100))
        cv2.line(resized_vis_map, scaled_start, scaled_end, (0, 0, 0), 2)
    cv2.imshow('Final Path', resized_vis_map)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def resize_map_for_display(map_img, scale_percent=20):  # Change scale_percent to 10
    width = int(map_img.shape[1] * scale_percent / 100)
    height = int(map_img.shape[0] * scale_percent / 100)
    return cv2.resize(map_img, (width, height), interpolation=cv2.INTER_AREA)


def reconstruct_path(came_from, current):
    path = []
    actions = []
    while current in came_from:
        current, action = came_from[current]
        path.append(current)
        actions.append(action)
    return path[::-1], actions[::-1]


def execute_path( path, rpm_actions,obstacle_map, visualization=True):
    velocities = []
    # ang_z = []
    if visualization:
        # map = obstacle_map.copy()
        vis_map = resize_map_for_display(obstacle_map, 20)
        for idx, (x, y, _) in enumerate(path):
            cv2.circle(vis_map, (int(x), int(y)), 5, (0, 255, 0), -1)
            if idx > 0:
                cv2.line(vis_map, (int(path[idx-1][0]/5), int(path[idx-1][1]/5)), (int(x/5), int(y/5)), (255, 0, 0), 2)
                # cv2.imshow("Path Planning Visualization", obstacle_map)
                cv2.waitKey(0)

    
    for i, action in enumerate(rpm_actions):
        # Convert RPM actions to linear and angular velocities
        linear_x, angular_z = rpm_to_velocity([action], R, L)[0]  # Assuming rpm_to_velocity returns a list of [linear_x, angular_z]
        velocities.append((linear_x/1000,-angular_z))
    
    print(velocities)

def get_valid_position(prompt, obstacle_map):
    while True:
        position_str = input(prompt)  # Example format input: "x,y,theta"
        x, y, theta = map(int, position_str.split(','))
        if validate_position(x, y, obstacle_map):
            return x, y, theta
        else:
            print("Invalid position. The position is either out of bounds or within an obstacle.")



def main(args=None):
    obstacle_map = create_map(map_width, map_height, total_clearance_mm)
    start = (500, 1000, 0)  # Start position
    goal = (5750, 1000, 0)  # Goal position, ensure to have a consistent format with start
    # start = (100,1900, 0)
    # goal = (5000, 1500, 0)
    path, rpm_actions = a_star(start, goal, actions, obstacle_map, visualization=True)  # Ensure visualization matches your intent
    if path:
        execute_path(path, rpm_actions, obstacle_map, visualization=True)  # Corrected order and added visualization argument
    else:
        print('No path found.')
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
