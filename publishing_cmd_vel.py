#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import heapq

# Robot parameters and map dimensions
R = 0.033  # Radius of wheels in meters
L = 0.160  # Distance between wheels in meters
robot_diameter_mm = 306  # Approximation to the larger dimension in mm for clearance calculation
user_clearance_mm = 50  # Additional clearance
total_clearance_mm = int((robot_diameter_mm / 2) + user_clearance_mm)
map_width, map_height = 6000, 2000  # Map dimensions in pixels (assuming 1 pixel = 1 mm for simplicity)
R1 = 50
R2 = 150
actions = [[0, R1], [R1, 0], [R1, R1], [0, R2], [R2, 0], [R2, R2], [R1, R2], [R2, R1]]
# Weight for the heuristic function
HEURISTIC_WEIGHT = 1.5  # Typical values might range from 1 to 2

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

def a_star(start, goal, actions, base_obstacle_map, visualization=False):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0}
    visited = set()  # Set to store visited nodes
    
    THRESHOLD_DISTANCE = 10  # Set a threshold distance to the goal point
    
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
            if neighbor in visited:
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

    return []

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
    vis_map = base_obstacle_map.copy()
    scale_percent = 20
    resized_vis_map = resize_map_for_display(vis_map, scale_percent)
    for i in range(len(path) - 1):
        # Scale path coordinates according to the resized map
        scaled_start = (int(path[i][0] * scale_percent / 100), int(path[i][1] * scale_percent / 100))
        scaled_end = (int(path[i+1][0] * scale_percent / 100), int(path[i+1][1] * scale_percent / 100))
        cv2.line(resized_vis_map, scaled_start, scaled_end, (0, 0, 0), 1)
    cv2.imshow('Final Path', resized_vis_map)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def resize_map_for_display(map_img, scale_percent=10):  # Change scale_percent to 10
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


class TurtlebotPathPlanner(Node):
    def __init__(self):
        super().__init__('turtlebot_path_planner')
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.obstacle_map = create_map(map_width, map_height, total_clearance_mm)
        # Visualization window setup
        cv2.namedWindow("1Path Planning Visualization", cv2.WINDOW_NORMAL)
        
    def publish_velocity(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.vel_pub.publish(msg)


    def visualize_map(self, resize_map_for_display):
        cv2.imshow("Path Planning Visualization", resize_map_for_display)
        cv2.waitKey(0)  # A short wait, so the window is responsive

    def execute_path(self, path, rpm_actions, visualization=True):
        if visualization:
            vis_map = self.obstacle_map.copy()
            for idx, (x, y, _) in enumerate(path):
                cv2.circle(vis_map, (int(x), int(y)), 5, (0, 255, 0), -1)
                if idx > 0:
                    cv2.line(vis_map, (int(path[idx-1][0]), int(path[idx-1][1])), (int(x), int(y)), (255, 0, 0), 2)
                self.visualize_map(vis_map)

        self.get_logger().info('Executing path...')
        for i, action in enumerate(rpm_actions):
            # Convert RPM actions to linear and angular velocities
            linear_x, angular_z = rpm_to_velocity([action], R, L)[0]  # Assuming rpm_to_velocity returns a list of [linear_x, angular_z]

            self.publish_velocity(linear_x, angular_z)
            print(linear_x,'x')
            print(angular_z)
            rclpy.spin_once(self, timeout_sec=1)  # Adjust time based on simulation speed and accuracy

        self.publish_velocity(0.0, 0.0)  # Stop the robot at the end of the path


def main(args=None):
    rclpy.init(args=args)
    turtlebot_path_planner = TurtlebotPathPlanner()

    start = (100, 100, 0)  # Start position
    goal = (5000, 1500)  # Goal position
    path, rpm_actions = a_star(start, goal, actions, turtlebot_path_planner.obstacle_map)
    path_actions = [(x, y, theta) for x, y, theta in path]  # Assuming path is a list of tuples (x, y, theta)

    if path:
        turtlebot_path_planner.execute_path(path_actions,rpm_actions)
        turtlebot_path_planner.get_logger().info('Path execution completed.')
    else:
        turtlebot_path_planner.get_logger().info('No path found.')

    turtlebot_path_planner.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
