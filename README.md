# ENPM661 Project 3 phase 2

Implementation of the A* algorithm on a Differential Drive (non-holonomic) TurtleBot3 robot

This README provides instructions on how to run the code for path planning (Part 01) and launching the ROS Node for path planning  (Part 02) using TurtleBot in a simulated environment.


## Dependencies

The following libraries and ROS packages have to be installed before running the code:

- Python 3
- NumPy
- OpenCV
- rclpy
- geometry_msgs
- nav_msgs
- time


## Team Members

- Abubakar Siddiq Palli | DirectoryID: absiddiq | UID: 120403422
- Gayatri Davuluri | DirectoryID: gayatrid | UID: 120304866
- Dhana Santhosh Reddy Jangama Reddy | DirectoryID: js5162 | UID: 120405570

## GitHub Repository Link

Access the code at: https://github.com/abubakar1107/A-Algorithm/tree/main


## Running the Code

### Part 01 - Path Planning - A* Visualization

Navigate to Part01 folder and run the script "proj3_ph2_p1_abubakar_gayatri_santhosh.py" using Python 3 in VSCode or use the following command
```bash
"python3 proj3_ph2_p1_abubakar_gayatri_santhosh.py"
```
User input: 
    start = (450, 1000, 0)  # Start position
    goal = (5800, 1000, 0)  # Goal position
    user clearance = 20
    RPM1 = 40
    RPM2 = 70

### Part 02 - Gazebo Simulation - Turtlebot3 waffle

Navigate to Part02 folder and copy the ROS2 package named "turtlebot3_project3" into your Workspace and run the following command.

Terminal 1:

    build the workspace with command "colcon build"
    launch the launch file named "competition_world.launch.py" using below command
```bash
            "ros2 launch turtlebot3_project3 competition_world.launch.py"
```
Terminal 2:

    run the script "proj3p2_abubakar_gayatri_santhosh.py" using below command
    
```bash
        "ros2 run turtlebot3_project3 proj3p2_abubakar_gayatri_santhosh.py"
```
User input for part2: 
    start = (500, 1000, 0)  # Start position
    goal = (5750, 1000, 0)  # Goal position
    user clearance = 30
    RPM1 = 50
    RPM2 = 50

Please find videos in below links:

    Part 1: https://drive.google.com/file/d/1jGHCvUWjVw_HrfJmx6Yasz2ceRDpbrNB/view?usp=sharing

    Part 2: https://drive.google.com/file/d/1vTdOrw4zyXc3kmnC72ljj2v12FfjLlvF/view?usp=sharing



