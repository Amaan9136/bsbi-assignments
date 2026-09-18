tasks:

the robot is just rotating in the first corner blue since there is no way to pass by the first corner. so i want to fix it where regardless of corner i want to just make it properly navigate where also move the obstacle from the corners (if assignment's (task 2 brief) allows it to do that) move the corner pieces and make it to navigate in the world of warehouse to properly navigate. current the robot moves avoids obstacles but problem is when the corner is found it just stops and then does this below log:.  i want to avoid spinning in the corner in any better way (can change/move world elements but must be what we are using but satisfying the assignment brief, and only if its needed.) the problem is corners having the obstacles. so you can give a solution to that considering the given brief pdf without removing anything that it says needed.

also the Lider is shown in the green thing but the robot should carry it with it to fix it. i guess its just a visual bug. since robot properly avoids the obstacles but still i want you to fix it in the proper way in the gui. (shown in 1789730578639_image.png)

devcontainers@Amaan-Ideapad-3:~/ros2_ws$ ros2 topic echo /mission_state
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---
data: NAVIGATE
---
data: AVOID_OBSTACLE
---
data: REPLAN
---


also in the gui i want to give the option 
Entity tree, Component inspector and also select move and related things with some toggle button so that i can properly see them. since its not visible as its custom gui thing is there. (if already exists just mention how to find it)