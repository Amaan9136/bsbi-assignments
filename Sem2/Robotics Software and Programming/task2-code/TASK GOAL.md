The robot’s main goal is to act as a **warehouse inspection patrol robot**. It must travel through inspection checkpoints, avoid pallet obstacles, and return to the charging dock while reporting its behavior.

## Mission route

it patrols a rectangular warehouse area:

The route covers the sides of the inspection area. It is not a full systematic coverage mission in which every square meter is cleaned or scanned. It is a **waypoint-based patrol mission**.

## What the robot should do

The robot should:

- Start in the `IDLE` state.
- Wait for a human command such as `start`.
- Navigate between the checkpoints.
- Use odometry to know its current position.
- Use its LiDAR scan to detect obstacles.
- Stop or turn when a pallet is too close.
- Move around the obstacle.
- Replan and continue toward the current checkpoint.
- Visit all checkpoints.
- Finish at the charging dock.
- Change to `MISSIONCOMPLETE`.
- Publish monitoring data such as distance traveled, mission time, state transitions, and obstacle encounters.

## State sequence

The intended behavior is:

```text
IDLE
  ↓ start command
NAVIGATE
  ↓ obstacle detected
AVOIDOBSTACLE
  ↓
REPLAN
  ↓
NAVIGATE
  ↓ all checkpoints reached
MISSIONCOMPLETE
```

The important point is that the goal is not simply “move forward.” The robot must demonstrate **goal-directed navigation, obstacle avoidance, replanning, and monitoring**.

## What does “cover” mean?

In this project, “cover” means the robot should patrol the route connecting the checkpoints

The pallet obstacles are deliberately placed near these patrol legs so that the robot is forced to exercise its avoidance and replanning states.

## One-sentence project goal

> The goal of the robot is to autonomously patrol warehouse inspection checkpoints, avoid obstacles using LiDAR, replan its movement when necessary, return to the charging dock, and report the mission’s performance through ROS 2 monitoring data.

___

## What the robot's goal is, and what to watch for after s:

1
Launch and idle
ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py spawns the TurtleBot3 burger_cam at (0,0), on the green charging dock. It sits still in IDLE — this is the screenshot you shared.
2
Send 's' to start
In Shell 2, run keyboard_hri_node and press 's'. State goes IDLE → NAVIGATE. Watch the terminal log 'Mission started - heading to Checkpoint 1 - East Aisle.'
3
Watch it drive to Checkpoint 1 (3,0)
The robot turns to face the waypoint and drives forward. In Gazebo, the blue LIDAR rays now move with it, clearly detached from the green dock.
4
Watch it dodge a pallet
Near a pallet stack the state flips to AVOID_OBSTACLE (backs off, turns to a clear heading) then REPLAN (drives that heading for ~0.5m) before returning to NAVIGATE — logged each time.
5
Checkpoints 2 and 3
It repeats this to (3,3) then (0,3), logging 'Reached Checkpoint N...' and 'Next target: ...' at each one.
6
Return to the dock
After (0,3) it heads back to (0,0). State becomes MISSION_COMPLETE and the log prints 'All checkpoints visited - back at the charging dock.'
7
Read the monitor summary
monitor_node (Shell 1) prints total time, distance travelled, obstacle-encounter count, and time spent per state