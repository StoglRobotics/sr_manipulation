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
