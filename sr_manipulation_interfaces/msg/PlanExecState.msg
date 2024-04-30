# planning state
int8 plan_state
int8 PLAN_RUNNING=2
int8 PLAN_SUCCESS=1
int8 PLAN_UNKNOWN=0
int8 PLAN_ERROR=-1
int8 PLAN_ERROR_GOAL_ACTIVE=-2
# start or final in collision
int8 PLAN_ERROR_STATE_IN_COLLISION=-3
int8 PLAN_ERROR_NO_SOLUTION=-4
int8 PLAN_ERROR_TIMEOUT=-5


# string to get more details on the error
string plan_message

# execution state
int8 exec_state
int8 EXEC_REACTIVE_MOVING=5
int8 EXEC_SCENE_UPDATE=4
int8 EXEC_MOVING=2
int8 EXEC_SUCCESS=1
int8 EXEC_UNKNOWN=0
int8 EXEC_ERROR=-1
int8 EXEC_ERROR_MOVING=-1
int8 EXEC_ERROR_SCENE_UPDATE=-2
# string to get more details on the error
string exec_message
