# Pose waypoint
geometry_msgs/PoseStamped pose
# use cartesian motion instead of joint space motion
bool cart
# velocity scaling factor
float64 velocity_scaling_factor
# blending radius in case of smooth sequential motion (if 0.0, each way point is reached at zero velocity)
float64 blending_radius
# names of the motion planner profile (pre-set of planning_pipeline, planner_id, constraints ...) to use
# If no name is specified, a default motion planner will be used
string planner_profile
string planning_group
string end_effector_link
