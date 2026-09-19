things to do:

even if it can move btwn the things. it is taking wrong path based on camera (maybe), but it should have done it properly . it should go btwn the things. and also i do not see the path. tell me if the things i am running in the cmds are correct in the right terminal
the lider is stuck in the green pad. even if it works the gazebo sim shows the lider in the charing station itself. i want you to check it all properly and fix it. find the issues /bugs and fix it and prestent the proper changed code zip
path isnt visble in the gazebo sim gui
robot keeps spinning, i want you to do make it properly work please. i beg u

robot properly avoids the obstacles but still i want you to fix it in the proper way in the gui of gazebo sim

techincally robot (looking at the image, since the space is bigger then the robot) can actaully pass btwn the things/objects but it takes bigger path bcz it takes the more size then its needed as a margin. i want it to consider fit on its size and move instead of taking too much space / margin on it. fix it and update the code
(previously ai said this, but it was fixed so i want you to explore more possibilities )
the robot was solving obstacle avoidance more cautiously than its own size needs. The cause was in compute_escape_angle() — it looks for the smallest turn angle that clears a required_clearance distance. The robot only actually needs about 0.23m of clearance (its own half-width) check for other causes too.

RN Robot is randomly moving in the warehouse environment. its just moving like stupid.

 a new PLANNING state (6 states total now, still ≥5 required). On s, mission_controller builds the checkpoint route as a nav_msgs/Path (current pose → each waypoint) and publishes it once on /planned_path, then auto-advances to NAVIGATE — logged as IDLE → PLANNING → NAVIGATE, exactly the "plan first, then move automatically" flow you wanted. AVOID_OBSTACLE/REPLAN are untouched — the plan is the checkpoint order, live obstacles are still handled reactively as before. gui need a toggle which shows the path that is planned on enabled. and can also disable the path that is being shown, but plan shouldnt change. robot first plans the path and then i want you to draw a line on the planned path and then move towards the checkpoints. it can just first plan then move automatically? also tell me if it go against my guidelines in the brief?
IF NOT THEN IMPLEMENT THIS. WHERE I WANT TO MAKE IT FOLLOW THAT PLANNED OPTIMAL PATH INSTEAD OF HALLUCINATING AND RANDOMLY MOVING. need the ui toggle button as i have for visualize lider i want same for visualize path thing. where i can see the path on what was planned i want a yellow colored path on complete movemnt what robot is going to do. which can be toggelld

_________________
i am using gazebo 8.11, dont run anything in the sandbox. u can test if you want for the did changes but dont overspend time/tool and token on it.
change the code files that needs to be edited and then present the changed files. do the proper analysis and fix all the things mentioned