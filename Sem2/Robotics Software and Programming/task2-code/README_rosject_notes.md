# Rosject notes: semantic_nav_monitor

**Use case:** Warehouse Inspection Patrol Robot - the TurtleBot3 patrols
four checkpoints in a custom warehouse bay (`worlds/warehouse_inspection.sdf`),
avoiding eight pallet obstacles placed along its route, while a monitor node logs
its state transitions and performance metrics.

---

## My local WSL Jazzy setup (read this first if not on TheConstruct.ai)

This package was originally written for **TheConstruct.ai Rosjects**, which run
**ROS2 Humble + Gazebo Classic** (`gazebo_ros`). I am running it locally
instead, on **WSL2, Ubuntu 24.04 "noble", ROS2 Jazzy**, confirmed with:

```bash
echo $ROS_DISTRO        # -> jazzy
which gzserver          # -> (empty, Classic not installed)
which gz                # -> /opt/ros/jazzy/opt/gz_tools_vendor/bin/gz
ros2 pkg list | grep -E "gazebo_ros|ros_gz_sim"   # -> ros_gz_sim, ros_gz_sim_demos (no gazebo_ros)
lsb_release -cs          # -> noble
```

Gazebo Classic does not exist for Jazzy — Jazzy pairs with **new Gazebo
"Harmonic"** via the `ros_gz_sim` / `ros_gz_bridge` / `ros_gz_image` packages
instead. Launching the original `gazebo_ros`-based launch file on Jazzy fails
with:

```
PackageNotFoundError: "package 'gazebo_ros' not found ..."
```

### Status: changes already applied on my machine

1. **`launch/semantic_nav_monitor.launch.py` — already Jazzy-ported, no
   action needed.** The version I copied into `~/ros2_ws/src/semantic_nav_monitor/`
   already uses `ros_gz_sim`'s `gz_sim.launch.py` (invoked twice — once with
   `-r -s -v2 <world>` for the server, once with `-g -v2` for the GUI client)
   instead of the Classic `gzserver.launch.py` / `gzclient.launch.py`
   includes. This mirrors how TurtleBot3's own
   `turtlebot3_gazebo/launch/turtlebot3_world.launch.py` does it on Jazzy.
   `robot_state_publisher.launch.py` and `spawn_turtlebot3.launch.py` from
   `turtlebot3_gazebo` didn't need changes — on Jazzy they already internally
   use `ros_gz_sim`/`ros_gz_bridge` (confirmed by inspecting
   `/opt/ros/jazzy/share/turtlebot3_gazebo/launch/`).

2. **`worlds/warehouse_inspection.world` → renamed to `warehouse_inspection.sdf`
   and patched.** Two separate fixes, both confirmed against Open Robotics'
   own current gz-sim8 (Harmonic) example worlds
   (github.com/gazebosim/gz-sim, `gz-sim8` branch, `examples/worlds/`):

   - **Extension modernized.** Gazebo Harmonic's own shipped example worlds
     (`shapes.sdf`, `lights.sdf`, etc.) all use the `.sdf` extension, not
     `.world` — `.world` is a Gazebo-Classic-era convention. `gz sim` itself
     doesn't actually care about the extension (it parses SDF content
     regardless of filename), but `.sdf` is the current standard, so the
     file was renamed and the `<sdf version="...">` header bumped from
     `1.6` to `1.11` to match the version declared in the official Harmonic
     example worlds.
   - **`model://` includes replaced with inline SDF.** The file had the two
     Gazebo-Classic-style includes:
     ```xml
     <include><uri>model://sun</uri></include>
     <include><uri>model://ground_plane</uri></include>
     ```
     These failed to resolve on this WSL install (`gz sim` errored with
     `Unable to find uri[model://sun]` / `...[model://ground_plane]`,
     which made the server exit immediately, cascading into a black
     GUI window and every other launched process being killed). Replaced
     with an inline `<light type="directional" name="sun">` and an inline
     `<model name="ground_plane">` (plane geometry + collision + visual) —
     this is exactly the pattern Open Robotics uses in its own official
     Harmonic example worlds, and it removes the dependency on
     `GZ_SIM_RESOURCE_PATH`/Fuel-cache resolution entirely, so it will load
     the same way on any machine. Everything else in the world file (walls,
     eight pallet obstacles, three checkpoint markers, charging dock) is
     plain SDF geometry/materials with no Classic-only plugins, so it
     needed no changes.

   Both fixes were applied directly to the file at
   `~/ros2_ws/src/semantic_nav_monitor/worlds/`; the old
   `warehouse_inspection.world` no longer exists in this package — use
   `warehouse_inspection.sdf` everywhere from now on.

3. **`setup.cfg` and `resource/semantic_nav_monitor` — created.** These two
   files are required by `setup.py` (`script-dir`/`install-scripts` config,
   and the ament resource-index marker) but were not present in the copied
   package. `setup.cfg` was created with the standard
   `[develop]`/`[install]` ament_python content, and an empty
   `resource/semantic_nav_monitor` marker file was created alongside it, in
   `~/ros2_ws/src/semantic_nav_monitor/`.

4. **Jazzy/Harmonic + TurtleBot3 packages — installed.** In place of the
   Humble/Classic equivalents (`ros-humble-gazebo-ros-pkgs` etc.):
   ```bash
   sudo apt-get install -y curl lsb-release gnupg
   sudo curl https://packages.osrfoundation.org/gazebo.gpg \
     --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
     | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
   sudo apt-get update
   sudo apt-get install -y gz-harmonic ros-jazzy-ros-gz
   sudo apt-get install -y ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-msgs ros-jazzy-turtlebot3-simulations
   ```

5. **Known harmless rosdep warning** — safe to ignore if it appears during
   `colcon build`:
   ```
   semantic_nav_monitor: Cannot locate rosdep definition for [gazebo_ros]
   ```
   This is expected: `package.xml` still lists `gazebo_ros` as an
   `exec_depend` because that's correct for the Rosject/Humble target
   environment described in the assignment brief. It is not needed at
   runtime on this Jazzy/Harmonic local setup, since the launch file no
   longer calls into it.

### Verify before building

Confirm the world-file patch actually took before rebuilding, from
`~/ros2_ws/src/semantic_nav_monitor/worlds/`:

```bash
ls warehouse_inspection.sdf          # confirm the renamed file exists
grep -n "model://" warehouse_inspection.sdf
```

The `grep` should print **nothing**. If `warehouse_inspection.sdf` doesn't
exist (only the old `.world` name is present), or if `grep` still shows
`model://sun` or `model://ground_plane`, the update didn't apply and needs
to be re-copied before continuing.

Also confirm the two added package files are in place, from
`~/ros2_ws/src/semantic_nav_monitor/`:

```bash
ls setup.cfg resource/semantic_nav_monitor
```

Both paths should be listed with no "No such file" errors.

### Build and launch (once verification above is clean)

```bash
cd ~/ros2_ws
colcon build --packages-select semantic_nav_monitor
source install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py
```

If the Gazebo GUI window fails to open (common on WSL2 without WSLg / on
Windows 10), that's a separate display issue, not a package issue — the
simulation server (`gz sim -s`) and the ROS nodes will still be running
headless; confirm via `ros2 topic echo /mission_state` in another terminal,
or launch with the GUI disabled by removing the `gz_sim_client_cmd` action
from the launch file.

If this is later run **on TheConstruct.ai**, use the *original*
`gazebo_ros`-based launch file and the backed-up
`warehouse_inspection.world.classic.bak` world file (restored to
`warehouse_inspection.world`) instead — this Jazzy port is WSL-specific.
TheConstruct.ai Rosjects are Humble + Gazebo Classic and already have
`gazebo_ros` available, so no changes are needed there. Keep both versions
if demoing on both environments.

---

## Original instructions (TheConstruct.ai Rosject, Humble + Gazebo Classic)

**Note:** this section targets a *different* platform (Humble + Gazebo
Classic on TheConstruct.ai) from the WSL/Jazzy/Harmonic section above, and
deliberately still refers to `warehouse_inspection.world` — Gazebo Classic
never adopted the `.sdf` naming convention, so `.world` is correct here.
Only the Harmonic-targeted copy of the file (used locally on WSL) was
renamed to `.sdf`.

Copy the `semantic_nav_monitor` folder into the `~/ros2_ws/src/` directory of
your Rosject on TheConstruct.ai, then run the following commands from the
Rosject's shell (Shell #1).

## 1. Build the package

```bash
cd ~/ros2_ws
colcon build --packages-select semantic_nav_monitor
source install/setup.bash
```

## 2. Set the TurtleBot3 model

```bash
export TURTLEBOT3_MODEL=burger
```

## 3. Launch the custom world, mission controller and monitor node

```bash
ros2 launch semantic_nav_monitor semantic_nav_monitor.launch.py
```

This loads the custom `warehouse_inspection.world` (walls, eight pallet
obstacles, four coloured checkpoint markers, a green charging-dock marker),
spawns the TurtleBot3 at the dock (0,0), and starts the
`mission_controller` and `monitor_node` nodes. The robot starts in the
`IDLE` state and will not move until it receives a start command.

## 4. Give the robot commands (from a second shell, Shell #2)

Preferred (literal keyboard HRI, no Enter key needed):

```bash
source ~/ros2_ws/install/setup.bash
ros2 run semantic_nav_monitor keyboard_hri_node
```

Then press:
- `s` to start the mission
- `p` to pause it
- `x` to stop it (mission is marked complete)
- `q` to quit the keyboard node (does not stop the mission)

Alternative (single one-shot commands, no live keypresses):

```bash
source ~/ros2_ws/install/setup.bash
ros2 topic pub --once /hri_command std_msgs/String "data: 'start'"
ros2 topic pub --once /hri_command std_msgs/String "data: 'pause'"
ros2 topic pub --once /hri_command std_msgs/String "data: 'stop'"
```

## 5. Watch the mission state and monitoring output

```bash
ros2 topic echo /mission_state
```

The `monitor_node` also prints state transitions and periodic mission
summaries (total time, distance travelled, obstacle-encounter count, time
per state) directly to its own console output in Shell #1.

## 6. Inspect the camera feed (optional, if a camera is enabled on the model)

```bash
ros2 run rqt_image_view rqt_image_view
```

## 7. What to capture for the report

- A Gazebo screenshot showing the warehouse bay, checkpoints, obstacles and
  the robot mid-patrol.
- Terminal screenshots of `mission_controller`/`monitor_node` output showing
  at least one full IDLE -> NAVIGATE -> AVOID_OBSTACLE -> REPLAN ->
  NAVIGATE -> MISSION_COMPLETE cycle.
- The final `monitor_node` summary block (total time, distance, obstacle
  encounters, time-per-state).

## 8. Rosject naming requirement

Per the assignment brief, the Rosject itself must be renamed to a unique
10-character alphanumeric string (for example, generated via random.org) and
that name must be quoted in Section 4.1 of the accompanying report, replacing
the `PLACEHOLDER_ROSJECT_NAME` marker.

---

The `rosject_notes` string below is provided for convenience, in case the
same content is wanted directly inside a Python file or notebook comment
inside the Rosject.