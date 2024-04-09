#!/usr/bin/env python3

import numpy as np
import cv2
import heapq

# Robot parameters and map dimensions
R = 33  # Radius of wheels in mm
L = 160  # Distance between wheels in mm
map_width, map_height = 6000, 2000  # Map dimensions in pixels 
HEURISTIC_WEIGHT = 10 # Weight for the heuristic A*
def create_map(width, height, clearance):
    """
    Function to create a map with bloated obstacles and clearance with robot radius
    """
    obstacle_map = np.ones((height, width, 3), dtype=np.uint8) * 255  

    # actual obsactales
    cv2.rectangle(obstacle_map, (1500 - clearance, 0 - clearance), (1750 + clearance, 1000 + clearance), (255,0,0), -1)
    cv2.rectangle(obstacle_map, (2500 - clearance, 1000 - clearance), (2750 + clearance, 2000 + clearance), (0,255,0), -1)
    cv2.circle(obstacle_map, (4200, 800), 600 + clearance, (0, 0, 255), -1)
    
    #map walls as obstacles
    cv2.rectangle(obstacle_map, (0, 0), (width, clearance//10), (0,0,255), -1)
    cv2.rectangle(obstacle_map, (0, height - (clearance//10)), (width, height), (0,0,255), -1)
    cv2.rectangle(obstacle_map, (0, 0), (clearance//10, height), (0,0,255), -1)
    cv2.rectangle(obstacle_map, (width - (clearance//10), 0), (width, height), (0,0,255), -1)

    return obstacle_map

def create_map_vis(width, height, clearance=10):
    
    """
    Similar Function to create a map with obstacles and clearance for visualization without bloating
    """

    obstacle_map = np.ones((height, width, 3), dtype=np.uint8) * 255  

    cv2.rectangle(obstacle_map, (1500 - clearance, 0 - clearance), (1750 + clearance, 1000 + clearance), (255,0,0), -1)
    cv2.rectangle(obstacle_map, (2500 - clearance, 1000 - clearance), (2750 + clearance, 2000 + clearance), (0,255,0), -1)
    cv2.circle(obstacle_map, (4200, 800), 600 + clearance, (0, 0, 255), -1)
    

    cv2.rectangle(obstacle_map, (0, 0), (width, clearance//10), (0,0,255), -1)
    cv2.rectangle(obstacle_map, (0, height - (clearance//10)), (width, height), (0,0,255), -1)
    cv2.rectangle(obstacle_map, (0, 0), (clearance//10, height), (0,0,255), -1)
    cv2.rectangle(obstacle_map, (width - (clearance//10), 0), (width, height), (0,0,255), -1)

    return obstacle_map

def calculate_new_position(x, y, theta, UL, UR, dt=0.075):

    """
    Calculates the new position of the robot based on the current position and the current velocity.
    """

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

def is_trajectory_collision_free(xi, yi, thetai, UL, UR, obstacle_map, dt=0.075):
    """
    Checks if the trajectory from the current position to the next position is free from collisions
    """
    t = 0
    while t < 1:
        Vl = UL * (2 * np.pi * R) / 60
        Vr = UR * (2 * np.pi * R) / 60
        Dx = (Vl + Vr) / 2 * np.cos(np.radians(thetai)) * dt
        Dy = (Vl + Vr) / 2 * np.sin(np.radians(thetai)) * dt
        Dtheta = (Vr - Vl) / L * dt
        xn = xi + Dx
        yn = yi + Dy

        if not is_collision_free(xn, yn, obstacle_map):
            return False

        xi, yi, thetai = xn, yn, (thetai + np.degrees(Dtheta)) % 360
        t += dt

    return True


def is_collision_free(x, y, obstacle_map):
    """
    Checks if the given position (x, y) is free from collisions.
    """
    if 0 <= x < obstacle_map.shape[1] and 0 <= y < obstacle_map.shape[0]:
        return np.all(obstacle_map[int(y), int(x)] == [255, 255, 255])
    return False

def heuristic(a, b):
    """
    Calculates the heuristic distance (euclidian) between two points.
    """
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def rpm_to_velocity(actions, R, L):
    """
    Convert RPM velocities to [linear_x, angular_z] format.
    """
    converted_actions = []
    for action in actions:
        RPM_l, RPM_r = action
        V_l = RPM_l * (2 * np.pi * R) / 60
        V_r = RPM_r * (2 * np.pi * R) / 60
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

def a_star(start, goal, actions, base_obstacle_map,vis_map, visualization=True):

    """Actual A* algorithm logic, tuned for Phase 2 from Phase 1"""

    open_set = [] # initialize open set
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0}
    visited = set()  
    
    threshold_distance = 100  
    
    
    while open_set: # While there are still open nodes
        current_f, current = heapq.heappop(open_set)

        if visualization and len(came_from) % 50 == 0:
            #visualize the exploration
            visualize_exploration(vis_map, came_from, current)
        # Check if the current node is close enough to the goal dnf if the current node is within the threshold distance of the goal
        if np.linalg.norm(np.array(current[:2]) - np.array(goal[:2])) < threshold_distance:

            if not is_collision_free(current[0], current[1], base_obstacle_map):

                continue  # Skip this node if it's not collision-free
            path, actions = reconstruct_path(came_from, current)

            if visualization:
                visualize_path(vis_map, path, actions)
                visualize_exploration(vis_map, came_from, current)
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
                # Apply the heuristic weight and check if the new node is closer to the goal
                f_score = tentative_g_score + heuristic(neighbor, goal) * HEURISTIC_WEIGHT 
                heapq.heappush(open_set, (f_score, neighbor))

        
        # if visualization and len(came_from) % 50 == 0:
        #     #visualize the exploration
        #     visualize_exploration(vis_map, came_from, current)
            

    return [], []

def plot_actual_curve_on_map(canvas, Xi, Yi, Thetai, UL, UR, color=(0, 0, 0), thickness=2):
    """
    Draws the actual curve taken by the robot from a parent node to a child node,
    based on differential drive kinematics.
    """
    dt = 0.1  # Time step 
    t = 0  # Current time
    duration = 1.0  # Total duration

    Thetai_rad = np.radians(Thetai)

    while t < duration:
        t += dt
        # Calculate velocities
        Vl = UL * (2 * np.pi * R) / 60
        Vr = UR * (2 * np.pi * R) / 60
        # Calculate change in position
        Vx = (Vr + Vl) / 2.0
        Vy = 0  

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
        Xi, Yi, Thetai_rad = Xn, Yn, Thetan


def visualize_exploration(base_obstacle_map, came_from, current):

    """
    Function to visualize the exploration searched by the algorithm.
    """
    vis_map = base_obstacle_map.copy()
    for node, (parent, action) in came_from.items():
        plot_actual_curve_on_map(vis_map, parent[0], parent[1], parent[2], action[0], action[1], color=(255, 0,255 ), thickness=2)

    cv2.circle(vis_map, (int(current[0]), int(current[1])), 10, color=(0, 255, 0), thickness=-1)

    resized_vis_map = resize_map_for_display(vis_map)
    cv2.imshow('Exploration with Actual Curves', resized_vis_map)
    cv2.waitKey(1)

def visualize_path(base_obstacle_map, path, rpm_actions, scale_percent=20):

    """
    Function to visualize the path taken by the robot to reach the goal.
    """
    vis_map = base_obstacle_map.copy()
    for i in range(len(path) - 1):
        Xi, Yi, Thetai = path[i]
        UL, UR = rpm_actions[i]  

        plot_actual_curve_on_map(vis_map, Xi, Yi, Thetai, UL, UR, color=(0, 0, 0), thickness=10)

    resized_vis_map = resize_map_for_display(vis_map, scale_percent)
    cv2.imshow('Final Path with Curves', resized_vis_map)
    cv2.waitKey(0)

def resize_map_for_display(map_img, scale_percent=20): 
    """
    Function to resize a map image for display.
    """
    width = int(map_img.shape[1] * scale_percent / 100)
    height = int(map_img.shape[0] * scale_percent / 100)
    return cv2.resize(map_img, (width, height), interpolation=cv2.INTER_AREA)


def reconstruct_path(came_from, current):
    """
    Reconstructs the path from the current node back to the start node.
    """
    path = []
    actions = []
    while current in came_from:
        current, action = came_from[current]
        path.append(current)
        actions.append(action)
    return path[::-1], actions[::-1]


def execute_path( path, rpm_actions,obstacle_map, visualization=True):

    """
    Function to execute the path taken by the robot and print the velocities.
    """
    velocities = []
    if visualization:
        vis_map = resize_map_for_display(obstacle_map, 20)
        for idx, (x, y, _) in enumerate(path):
            cv2.circle(vis_map, (int(x), int(y)), 5, (0, 255, 0), -1)
            if idx > 0:
                cv2.line(vis_map, (int(path[idx-1][0]/5), int(path[idx-1][1]/5)), (int(x/5), int(y/5)), (255, 0, 0), 2)
                # cv2.imshow("Path Planning Visualization", obstacle_map)
                cv2.waitKey(0)
    
    for i, action in enumerate(rpm_actions):
        # Convert RPM actions to linear and angular velocities
        linear_x, angular_z = rpm_to_velocity([action], R, L)[0] 
        velocities.append((linear_x/1000,-angular_z))
    print("\nVelocities to publish to cmd_vel:\n")
    print(velocities)

def get_valid_position(prompt, obstacle_map):

    """
    Function to get a valid position from the user.
    """
    while True:
        position_str = input(prompt)  
        x, y, theta = map(int, position_str.split(','))
        y = map_height - y 
        if validate_position(x, y, obstacle_map):
            return (x, y, theta)
        else:
            print("Invalid position. The position is either out of bounds or within an obstacle.")



def main(args=None):

    """
    Main function with User Inputs
    """
    
    robot_diameter_mm = 440 # Robot outer diameter including wheels from documentation in mm
    user_clearance_mm = int(input("Enter user clearance: "))  # Additional clearance in mm  
    total_clearance_mm = int((robot_diameter_mm / 2) + user_clearance_mm)

    obstacle_map = create_map(map_width, map_height, total_clearance_mm)
    vis_map = create_map_vis(map_width, map_height) 

    start = get_valid_position("Enter start position (x, y, theta): ", obstacle_map)
    goal = get_valid_position("Enter goal position (x, y, theta): ", obstacle_map)

    R1 = int(input("Enter R1 (1st possible rpm): "))
    R2 = int(input("Enter R2 (2nd possible rpm): "))
    actions = [[0, R1], [R1, 0], [R1, R1], [0, R2], [R2, 0], [R2, R2], [R1, R2], [R2, R1]]


    # start = (500, 1000, 0)  # Start position
    # goal = (5750, 1000, 0)  # Goal position
    
    path, rpm_actions = a_star(start, goal, actions, obstacle_map,vis_map, visualization=True)  
    if path:
        print("Path: \n------------------",path, "\n------------------")
        execute_path(path, rpm_actions, obstacle_map, visualization=True)  
    else:
        print('No path found.')
    


if __name__ == '__main__':
    main()
