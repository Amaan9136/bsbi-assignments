The robot’s main goal is to act as a **warehouse inspection patrol robot**. It must travel through four predefined inspection checkpoints, avoid pallet obstacles, and return to the charging dock while reporting its behavior.

## Mission route

The robot starts at the green charging dock at approximately **(0.0, 0.0)** and follows this closed-loop route:

1. Move to checkpoint 1: **(3.0, 0.0)**
2. Move to checkpoint 2: **(3.0, 3.0)**
3. Move to checkpoint 3: **(0.0, 3.0)**
4. Return to checkpoint 4/dock: **(0.0, 0.0)**

So, conceptually, it patrols a rectangular warehouse area:

```text
(0.0, 3.0)  ───────────  (3.0, 3.0)
     ▲                         │
     │                         ▼
(0.0, 0.0)  ◄──────────  (3.0, 0.0)
 charging dock
```

The route covers the four sides of the inspection area. It is not a full systematic coverage mission in which every square meter is cleaned or scanned. It is a **waypoint-based patrol mission**.

## What the robot should do

The robot should:

- Start in the `IDLE` state.
- Wait for a human command such as `start`.
- Navigate between the four checkpoints.
- Use odometry to know its current position.
- Use its LiDAR scan to detect obstacles.
- Stop or turn when a pallet is too close.
- Move around the obstacle.
- Replan and continue toward the current checkpoint.
- Visit all four checkpoints.
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

In this project, “cover” means the robot should patrol the route connecting the four checkpoints:

- Bottom side: `(0.0, 0.0)` to `(3.0, 0.0)`.
- Right side: `(3.0, 0.0)` to `(3.0, 3.0)`.
- Top side: `(3.0, 3.0)` to `(0.0, 3.0)`.
- Left side: `(0.0, 3.0)` to `(0.0, 0.0)`.

The pallet obstacles are deliberately placed near these patrol legs so that the robot is forced to exercise its avoidance and replanning states.

## One-sentence project goal

> The goal of the robot is to autonomously patrol four warehouse inspection checkpoints, avoid obstacles using LiDAR, replan its movement when necessary, return to the charging dock, and report the mission’s performance through ROS 2 monitoring data.