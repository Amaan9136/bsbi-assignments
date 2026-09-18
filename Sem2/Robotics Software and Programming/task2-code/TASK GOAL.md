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