# Autonomous TurtleBot with YOLOv5 and Deep Reinforcement Learning

A ROS 1 robotics project that combines TurtleBot3, Gazebo simulation, YOLOv5 object detection, and Deep Reinforcement Learning for vision-guided autonomous navigation.

## Project Goal

The goal of this project is to develop an autonomous TurtleBot that can:

- Detect objects using YOLOv5
- Locate an object in the camera image
- Center itself toward the object
- Move toward the target
- Reach the target
- Avoid obstacles using LiDAR
- Work in randomized environments
- Learn through Reinforcement Learning
- Be evaluated using robotics and RL metrics
- Eventually support a complete MLOps pipeline

## System Architecture

```text
Gazebo Simulation
       |
       v
TurtleBot RGB Camera
       |
       v
     YOLOv5
       |
       v
   /yolo_state
       |
       v
    DQN Agent
       |
       v
    /cmd_vel
       |
       v
   TurtleBot3
```

The camera provides images to YOLOv5.

YOLOv5 detects an object and converts the detection into a compact numerical state.

The DQN agent receives this state and selects an action.

The selected action is converted into a ROS `/cmd_vel` command that controls the TurtleBot.

## YOLOv5 State

The current YOLO node publishes:

```text
[center_x, center_y, width, height]
```

Where:

- `center_x` = horizontal center of the detected bounding box
- `center_y` = vertical center of the detected bounding box
- `width` = normalized width of the bounding box
- `height` = normalized height of the bounding box

The values are normalized relative to the camera image dimensions.

Using a compact numerical state instead of feeding full camera images directly into DQN makes training significantly lighter and faster.

## Reinforcement Learning

The current reinforcement learning algorithm is a Deep Q-Network (DQN).

The project currently includes:

- Neural-network Q-function
- Epsilon-greedy exploration
- Experience replay
- Replay buffer
- Target network
- PyTorch optimization
- Model checkpointing
- Training reward logging

### Current DQN Architecture

```text
Input State
     |
     v
Linear Layer
     |
   ReLU
     |
     v
Linear Layer
     |
   ReLU
     |
     v
Output Layer
     |
     v
5 Actions
```

The current network uses:

```text
Input: 2 values
Hidden Layer 1: 64 neurons
Hidden Layer 2: 64 neurons
Output: 5 actions
```

## Current Action Space

```text
0 = Move Forward
1 = Move Backward
2 = Turn Left
3 = Turn Right
4 = Stop
```

These actions are converted into ROS `geometry_msgs/Twist` commands and published to:

```text
/cmd_vel
```

## Main ROS Topics

Important ROS topics currently used by the project include:

```text
/camera/rgb/image_raw
/yolo_state
/cmd_vel
/scan
/odom
/imu
/joint_states
/tf
/tf_static
/clock
/gazebo/model_states
/gazebo/link_states
```

## Main Technologies

This project uses:

- Python
- ROS 1
- Gazebo
- TurtleBot3
- YOLOv5
- PyTorch
- OpenCV
- NumPy
- Deep Q-Networks
- Experience Replay
- Reinforcement Learning
- Computer Vision

## Repository Structure

```text
first/
├── CMakeLists.txt
├── package.xml
├── requirements.txt
├── README.md
├── launch/
│   └── project.launch
└── src/
    ├── train.py
    ├── dqn.py
    ├── dqn_agent.py
    ├── replay_buffer.py
    ├── rl_env.py
    ├── set_env.world
    ├── YOLO integration scripts
    └── yolov5/
```

Additional experimental computer-vision and ROS scripts are currently retained while the project is being developed.

## Current Project Status

The basic ROS + Gazebo + YOLOv5 + DQN pipeline is operational.

The current system can:

- Launch Gazebo
- Spawn TurtleBot3
- Receive RGB camera images
- Run YOLOv5 object detection
- Publish object information through `/yolo_state`
- Receive the YOLO state in the RL environment
- Select actions using DQN
- Publish commands through `/cmd_vel`
- Store transitions in experience replay
- Train the DQN
- Save model checkpoints
- Save replay-buffer data

The current baseline is being preserved before major improvements are made.

## Known Baseline Issues

The current implementation is functional but still contains reinforcement-learning and robotics issues that will be fixed incrementally.

Current known issues include:

- RL state mapping needs correction
- YOLO publishes `[center_x, center_y, width, height]`, while the current training code uses only part of this state
- Reward logic needs improvement
- Episode termination needs improvement
- Maximum episode duration is not currently enforced
- Epsilon decay occurs too quickly during long episodes
- Target network synchronization needs improvement
- Environment reset does not yet physically reset the robot and world
- Multiple `/cmd_vel` publishers can interfere with RL training
- YOLO currently uses the first detected object instead of a selected target
- YOLO visualization adds unnecessary training overhead
- Gazebo GUI adds training overhead
- Training logging is currently too frequent
- Training throughput needs optimization

These issues will be fixed individually and documented through separate Git commits.

## Development Roadmap

### Stage 1 — Baseline Architecture Audit

Completed.

The complete system architecture, ROS topics, YOLO state, DQN architecture, actions, reward function, replay buffer, and training loop were inspected.

### Stage 2 — Object Centering

Build a simplified RL task where the TurtleBot learns to center a detected target.

Initial state:

```text
target horizontal offset
```

Initial actions:

```text
Turn Left
Turn Right
Stop
```

### Stage 3 — Target Approach

Add target size as an approximation of distance.

The robot will learn to move toward the object.

### Stage 4 — Target Reaching

Create a reliable success condition and proper episode termination.

### Stage 5 — LiDAR Obstacle Avoidance

Use compressed LiDAR observations such as:

```text
left distance
front distance
right distance
```

### Stage 6 — Randomized Training

Randomize:

- Robot starting position
- Target position
- Obstacles
- Environment configurations

### Stage 7 — Multiple Object Targets

Use YOLO class information so the robot can locate selected target objects.

### Stage 8 — RL Algorithm Comparison

Potential algorithms include:

- DQN
- Double DQN
- PPO

### Stage 9 — Performance Optimization

Training will be optimized using techniques such as:

- Headless Gazebo
- Reduced camera processing
- Reduced YOLO inference frequency
- Disabled visualization during training
- Efficient environment reset
- Reduced terminal logging
- GPU utilization monitoring
- Environment steps-per-second measurement

### Stage 10 — Evaluation

Planned evaluation metrics include:

- Success rate
- Collision rate
- Average episode reward
- Episode duration
- Time to target
- Distance travelled
- Target-loss count
- Centering error
- Environment steps per second

### Stage 11 — MLOps

After the robotics system becomes stable, the project can include:

- MLflow experiment tracking
- Hyperparameter logging
- Model Registry
- Candidate/champion models
- Automated evaluation
- CI/CD
- Testing
- Monitoring
- Retraining workflow

## ROS Dependencies

The main ROS dependencies are declared in `package.xml`.

They currently include:

```text
rospy
std_msgs
geometry_msgs
sensor_msgs
cv_bridge
gazebo_ros
xacro
turtlebot3_description
turtlebot3_teleop
```

## Python Dependencies

Python dependencies are listed in:

```text
requirements.txt
```

Current dependencies include:

```text
numpy
torch
torchvision
opencv-python
```

## YOLOv5 Dependency

This project uses the official Ultralytics YOLOv5 repository.

YOLOv5 is maintained separately from the project-specific robotics code so that third-party source code is clearly distinguished from the code developed for this project.

The current YOLOv5 source is based on the official repository:

```text
https://github.com/ultralytics/yolov5
```

The project currently references YOLOv5 commit:

```text
eef637c
```

## Build the ROS Workspace

From the Catkin workspace:

```bash
cd ~/my_ws
catkin_make
source devel/setup.bash
```

## TurtleBot3 Model

Before launching TurtleBot3, set the appropriate model if required.

For example:

```bash
export TURTLEBOT3_MODEL=burger
```

## Launch the Simulation

Launch the project using:

```bash
roslaunch first project.launch
```

The launch file:

- Loads the custom Gazebo world
- Loads the TurtleBot3 robot description
- Spawns TurtleBot3
- Starts TurtleBot3 teleoperation

## Training Pipeline

The current training pipeline is approximately:

```text
Start Gazebo
     |
     v
Start TurtleBot
     |
     v
Camera Image
     |
     v
YOLOv5 Detection
     |
     v
/yolo_state
     |
     v
RL Environment
     |
     v
DQN Action Selection
     |
     v
/cmd_vel
     |
     v
TurtleBot Movement
     |
     v
Reward
     |
     v
Replay Buffer
     |
     v
DQN Training
```

## Project Philosophy

This repository preserves the original working baseline and improves the project through small engineering changes.

Each major bug fix, optimization, RL improvement, and MLOps stage will be documented through Git so that the development history demonstrates how the complete autonomous robotics system evolved.

## Status

**Active Development**

Current focus:

```text
Baseline preservation
        ↓
Correct RL implementation
        ↓
Object centering
        ↓
Target approach
        ↓
Obstacle avoidance
        ↓
Robust autonomous behavior
        ↓
Evaluation
        ↓
MLOps
```