tasks:

when it detects the obstacle i want to pass in the better way. also in the corners there are 4 obstacles in 4 corners i want it to smartly move from those corner obstacles. also in the world i want to place the small obstacles in the path of the robot world. where slightly left to the path. slightly right to the path and middle to the path. also it should automatically choose the better path to move. where it should auto right and auto left plan. currently when robot finds the obstacle it just trys to go back and fourth infront of those obstacles which is wrong. it should come back and choose the proper right or left and then again move towards that point (i dont want to mention everything i just want you to give a proper navigation to make my assignment work properly without any issues) "Goal-Oriented Autonomous Robot with Monitoring (ROS2)". make it a proper and good warehouse type of looking environment. where also make the brown boxes small and stack multiple boxes to make it look like a proper warehouse

logging is not properly working "i guess this is the command to check the logging: devcontainers@Amaan-Ideapad-3:~/ros2_ws$ ros2 topic echo /mission_state" but its just stuck. even if i run the robot by doing "s" it still stuck. fix all these issues and make it as needed for the assignment. do not change anything thats already working unless its needed

i want to see the lidar waves and sensor based signals in the gui of the sim itself with some parameter or an argument which will help to toggle to see or disable the lidar sensor sensing, also if i a m using any other sensors then i want to do that. and also instead of using the export TURTLEBOT3_MODEL=burger in all run give a default option in the code by making the burger there in the code itself. make all the needed changes and present the updated files

devcontainers@Amaan-Ideapad-3:~/ros2_ws$ ros2 topic list
/clock
/cmd_vel
/hri_command
/imu
/joint_states
/mission_state
/odom
/parameter_events
/robot_description
/rosout
/scan
/tf
/tf_static
devcontainers@Amaan-Ideapad-3:~/ros2_ws$ ros2 topic echo /mission_state
....(NOTHING WORKS, WAITING FROM LAST 20 Min)....
___________________________________

CONTEXT — read before doing anything

This is a ROS2 Jazzy + Gazebo Sim 8.11.0 (gz-sim) TurtleBot3 Burger project,
package `semantic_nav_monitor`, assignment: "Goal-Oriented Autonomous Robot
with Monitoring." Source of truth for the repo layout is a repomix export;
if one is attached, extract it with the repomix-extractor skill FIRST
(one call) and work from the reconstructed files — do not re-parse the
markdown. If no repomix export is attached, ask for one before proceeding;
do not attempt this from memory of a prior conversation.

Key files:
- semantic_nav_monitor/semantic_nav_monitor/mission_controller.py — the FSM
  (IDLE/NAVIGATE/AVOID_OBSTACLE/REPLAN/MISSION_COMPLETE), LIDAR-based
  obstacle detection, HRI command handling, /cmd_vel (TwistStamped) publishing
- semantic_nav_monitor/worlds/warehouse_inspection.sdf — the custom world
- semantic_nav_monitor/launch/semantic_nav_monitor.launch.py
- semantic_nav_monitor/semantic_nav_monitor/keyboard_hri_node.py
- semantic_nav_monitor/semantic_nav_monitor/monitor_node.py

Already fixed and confirmed working in this environment, do not re-diagnose
or revert unless a specific new task below requires touching that exact code:
- The world file was missing gz-sim system plugins (Physics, Sensors with
  ogre2, SceneBroadcaster, UserCommands) — this caused /scan to register as
  a topic but publish nothing. Fixed by adding <plugin> tags at the top of
  <world>. Confirmed: /scan now publishes at ~4.3-5Hz (sensor's declared
  <update_rate> is 5, from turtlebot3_burger/model.sdf — this rate is
  correct/expected, do not "fix" it).
- forward_obstacle_distance() originally used a fixed ±30° angular detection
  cone, which misses off-centerline obstacles because the cone's real-world
  width shrinks with range. Replaced with a Cartesian corridor check
  (ROBOT_HALF_WIDTH_M + OBSTACLE_LATERAL_MARGIN_M lateral threshold,
  forward>0 projection) in the same method. Keep this approach; extend it
  rather than reverting to angle-based detection.
- gz sim RTF confirmed ≈1.0, sim clock is not desynced — do not re-investigate
  timing/performance as a root cause for anything below unless new evidence
  specifically points there.
- /mission_state topic confirmed present in `ros2 topic list` and publishing
  correctly (node is alive, publisher works). An earlier "stuck echo" report
  was a user-side typo/timing issue, not a real bug — do not re-investigate
  this unless new, different evidence is presented.
- The reverse-then-turn AVOID_OBSTACLE behavior (REVERSE_DURATION_S=1.0 then
  rotate in place using avoid_turn_direction chosen from side_clearance())
  is the CURRENT behavior the user wants replaced — see Task 1.

TASKS (do in this order; each depends on groundwork from the previous one)

1. Replace the AVOID_OBSTACLE behavior. Current problem: robot detects an
   obstacle, reverses, turns in place, and can end up oscillating
   back-and-forth in front of the obstacle rather than committing to a
   direction and resuming toward the goal. Required behavior: on obstacle
   detection, pick a clear side using existing side_clearance() (or an
   improved version), back off only as far as needed to have a clear turn
   radius, rotate to face a heading that clears the obstacle with margin,
   then transition back to NAVIGATE (not idle in REPLAN oscillating) so the
   waypoint-seeking controller resumes and naturally curves around the
   obstacle rather than requiring a separate "go back to the line" state.
   Preserve the MissionState enum and pub/sub interfaces (/scan, /odom,
   /cmd_vel TwistStamped, /mission_state, /hri_command) exactly — only
   change the avoidance state's internal logic and transition targets.
   The four corner obstacles need specific handling: a corner obstacle can
   have poor clearance on one entire side (a wall nearby), so
   side_clearance() must weight wall proximity (available from the same
   /scan data, not just the two obstacle-adjacent sectors) when picking
   turn direction — don't turn toward a wall.
   Once REPLAN's role changes (or is removed) because NAVIGATE now resumes
   directly, check whether REPLAN becomes dead code entirely and, if so,
   remove it and the MissionState.REPLAN member rather than leaving an
   unused state — don't leave two mechanisms doing the same job.

2. World redesign (warehouse_inspection.sdf), only after Task 1 is working:
   - Add exactly 3 small new obstacles positioned relative to the direct
     line between two existing waypoints: one offset slightly left of the
     path centerline, one slightly right, one directly on the centerline —
     use small enough offsets that this is a genuine navigation test, not
     an easy pass. Pick which leg makes sense given existing pallet
     placement; don't collide with existing obstacles.
   - Existing 8 pallet obstacles stay as-is except: shrink the box size
     (currently 0.3x0.3x0.3) to something visually smaller, and replace
     single boxes with small 2-4 box stacks per position to look like
     stacked warehouse pallets — keep total footprint per obstacle cluster
     similar to what it replaces so existing waypoint geometry/corridor
     math doesn't need re-tuning, unless the new centerline obstacles
     already force a retune.
   - Identify which 4 of the existing pallet obstacles are the "corner"
     ones (near wall intersections) from their pose values already in the
     file, don't assume numbering — check actual (x,y) against the wall
     bounds (x in [-2,5], y in [-2,5] per the file's own header comment).

3. LIDAR visualization in the Gazebo GUI, toggleable via a launch argument
   (e.g. `show_lidar:=true`), not a code recompile. Research gz-sim 8.x's
   actual supported mechanism for this before writing anything — likely
   candidates are the sensor's own <visualize> tag (already true in the
   upstream turtlebot3_burger model, confirm what it currently renders as)
   combined with a GUI plugin (e.g. a LaserScan visualization plugin) added
   to the world's <gui> block, or gz-sim's built-in "Visualize Lidar" GUI
   plugin if this version ships one. Do not invent an SDF tag that doesn't
   exist in gz-sim 8.11 — check against real gz-sim GUI plugin
   documentation/source, not assumption.

4. Default TURTLEBOT3_MODEL=burger in code instead of requiring `export`
   before every run — set it via SetEnvironmentVariable in the launch
   file's LaunchDescription (or equivalent), not by hardcoding it in a
   way that breaks someone overriding it explicitly.

5. Dedup pass, after all above are done: search the full package for any
   remaining code that duplicates functionality now handled elsewhere
   (e.g. two things setting TURTLEBOT3_MODEL, two things computing
   obstacle clearance, dead states/constants left over from the Task 1
   rewrite, a leftover REPLAN-era constant like CLEAR_DETECTIONS_REQUIRED
   if REPLAN is removed). Remove the redundant one, keep the one that's
   actually wired into the current flow. List what was removed and why in
   your response text (not as code comments) so the user can sanity-check
   the removals.

CONSTRAINTS
- Don't touch anything not named above. In particular do not modify
  keyboard_hri_node.py, monitor_node.py's core logic, or the TwistStamped
  /cmd_vel publishing, unless a task explicitly requires it.
- No code comments, no unrelated reformatting, match existing style exactly,
  per this project's established editing convention.
- Present each changed file as an actual file via the file-creation/present
  flow, not just inline diffs in chat.
- No sandbox/simulation testing needed — I will test locally and report
  back with actual terminal/log output, which you should wait for before
  iterating further on a given task rather than guessing at a second fix.

TOOL-CALL BUDGET
- Batch reads: view the full directory tree once, then view all files you
  expect to need in as few calls as possible rather than one file per call
  when they're related.
- Batch edits: for a single file, make all needed str_replace edits back
  to back without re-viewing the file between every single edit unless a
  prior edit changed something a later edit depends on.
- Don't re-verify things already confirmed working in this session's
  context above (plugins, scan rate, RTF, /mission_state) — treat those as
  settled facts, not things to re-check.
- Prefer one comprehensive response per task over multiple small
  round-trips.