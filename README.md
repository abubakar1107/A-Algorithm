# Team Details:

1. Abubakar Siddiq Palli | DirectoryID: absiddiq | UID: 120403422
2. Gayatri Davuluri | DirectoryID: gayatrid | UID: 120304866
3. Dhana Santhosh Reddy Jangama Reddy | DirectoryID: js5162 | UID: 120405570

# A* Pathfinding Algorithm

A Python script that implements the A* algorithm for pathfinding in a 2D environment, considering obstacles, clearance, and orientation.

## Features

- Efficient pathfinding with the A* algorithm.
- Obstacle avoidance with specified clearance.
- Orientation-aware goal reaching.
- Visualization of pathfinding process using OpenCV.

## Setup

**Dependencies**: Python 3, NumPy, OpenCV-Python.

**Configuration**:
- **Canvas Size**: 600x250 units.
- **Start Node used in Video**: (10, 10, 10) (x, y, orientation).
- **Goal Node used in Video**: (350, 200, 45) (x, y, orientation).
- **Step Size (L) used**: 10 units.
- **Clearance**: 5 units from obstacles.


## Running the Script

```bash
python a_star_abubakar_gayatri_santhosh.py
```

## Output

- Validates start/goal positions for obstacle clearance.
- Finds and visualizes the optimal path.
- Prints path coordinates and execution time.


