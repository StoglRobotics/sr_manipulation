#####  Detailed Documentation  ########
#
# MANIP_REACH_PREGRASP - Reaching 'pre-grasp' pose from an arbitrary robot state *without* object in
# the gripper. More details in the note below. Default planner: 'ompl'.
#
# MANIP_MOVE_GRASP - Move manipulator to the grasp position *without* object attached.
# It is expeced that the robot is in 'pre-grasp' state and linear movement in cartesian space is
# posible to reach grasp position. Default planner: 'pilz_lin'.
#
# MANIP_MOVE_GRASP_ADJUST - Adjusting grasp clearance if any to avoid object movement when gripper
# is closing.Default planner: 'pilz_lin'.
#
# MANIP_GRASP - Grasping the object (closing gripper) and attaching object to the manipualtor for
# planning.
#
# MANIP_GRIPPER_CLOSE - equivalent to MANIP_GRASP.
#
# MANIP_MOVE_POSTGRASP - Move manipulator to the post grasp position *with* an object attached.
# It is expeced that the robot has graped (and attached) an object in the previous step.
# More details in the note below. Default planner: 'pilz_lin'.
#
# MANIP_REACH_PREPLACE - Reaching 'pre-place' pose poro an arbitrary robot state *with* object in
# the gripper. More details in the note below. Default planner: 'ompl_with_constraints'
#
# MANIP_MOVE_PLACE - Move manipulator to the place position *with* an object attached.
# It is expeced that the robot is in 'pre-place' state and linear movement in cartesian space is
# posible to reach grasp position. Default planner: 'pilz_lin'.
#
# MANIP_RELEASE - Releasing the object (opening gripper) and detaching object from the manipualtor
# for planning.
#
# MANIP_GRIPPER_OPEN - equivalent to MANIP_RELEASE.
#
# MANIP_MOVE_PLACE_ADJUST - Adjusting grasp clearance if any to avoid object movement when gripper
# is moving away from the object. Default planner: 'pilz_lin'.
#
# MANIP_MOVE_POSTPLACE - Move manipulator to the post place position *without* object attached.
# It is expeced that the robot has placed (and detached) an object in the previous step.
# More details in the note below. Default planner: 'pilz_lin'.
#
# MANIP_GRIPPER_ADJUST - Adjust Gripper Gauge. Gripper is moved to adjustment position and adjusting
# procedure is executed. The procedure needs synchronized behavior of the manipualtor and the gripper.
#
#
# NOTE on 'pre-' and 'post-' grasping and placing poses - The poses are defined from the object's
# perspecitve, i.e., in the target frame form where object is grasped or placed to. The relevant
# parameters are distance from the targt and direction vector from which target should be approached.
# One can also define planner with or without constraines for those actions. Per



# definition of various types of manipulation actions/sequence of actions
int8 type
int8 MANIP_UNKNOWN=0
# pick sub actions
int8 MANIP_REACH_PREGRASP=1
int8 MANIP_MOVE_GRASP=2
int8 MANIP_GRASP=3
int8 MANIP_MOVE_POSTGRASP=4
# leaving room for other types of pick
# place sub actions
int8 MANIP_MOVE_GRASP_ADJUST=31
int8 MANIP_REACH_PREPLACE=51
int8 MANIP_MOVE_PLACE=52
int8 MANIP_MOVE_PLACE_ADJUST=53
int8 MANIP_RELEASE=54
int8 MANIP_MOVE_POSTPLACE=55
# gripper special actions
int8 MANIP_GRIPPER_ADJUST=100
int8 MANIP_GRIPPER_OPEN=101
int8 MANIP_GRIPPER_CLOSE=102
