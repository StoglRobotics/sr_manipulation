# BSD 3-Clause License
#
# Copyright (c) 2023, Stogl Robotics Consulting
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# based on moveit_clients.py, scene_client 
#
# Author Dr Denis Stogl, Guillaume Walck


from copy import deepcopy

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse, ActionClient
from rclpy.action.server import ServerGoalHandle, GoalStatus
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sr_manipulation_interfaces.action import PlanMoveTo, Manip
from sr_manipulation_interfaces.msg import ManipType, PlanExecState, ServiceResult
from sr_manipulation_interfaces.srv import AttachObject, DetachObject

from control_msgs.action import GripperCommand

from geometry_msgs.msg import PoseStamped, Vector3Stamped, Pose
from moveit_msgs.msg import Grasp
import moveit_msgs.msg
from sr_ros2_python_utils.transforms import TCPTransforms
from sr_ros2_python_utils.visualization_publishers import VisualizatonPublisher
import numpy as np
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from rclpy.action import ActionClient
from moveit_msgs.msg import Constraints, OrientationConstraint, JointConstraint, PositionConstraint
from moveit_msgs.msg import MotionPlanRequest
from moveit_msgs.msg import PlanningOptions
from moveit_msgs.msg import RobotState, RobotTrajectory
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from sensor_msgs.msg import JointState
from sr_manipulation_interfaces.srv import MoveToPose, MoveToPoseSeq
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String

def wait_for_response(future, client):
    while rclpy.ok():
        rclpy.spin_once(client)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                client.get_logger().info(
                    'Service call to move_to_pose or move_to_pose_seq failed %r' % (e,))
                return None
            else:
                return response


class RobotClient(Node):
    def __init__(self):
        """
        Create a new client for managing manipulation requests.
        """
        super().__init__('robot_client_node')

        # defaults
        self.active_goal = None
        self.plan_move_to_feedback = PlanMoveTo.Feedback()
        self.manip_feedback = Manip.Feedback()
        self.SUPPORTED_MANIP_ACTIONS = [ManipType.MANIP_MOVE_GRASP,
                                        ManipType.MANIP_MOVE_POSTGRASP,
                                        ManipType.MANIP_MOVE_PLACE,
                                        ManipType.MANIP_MOVE_PLACE_ADJUST,
                                        ManipType.MANIP_MOVE_POSTPLACE,
                                        ManipType.MANIP_REACH_PREGRASP,
                                        ManipType.MANIP_REACH_PREPLACE,
                                        ManipType.MANIP_GRASP,
                                        ManipType.MANIP_MOVE_GRASP_ADJUST,
                                        ManipType.MANIP_RELEASE,
                                        ManipType.MANIP_GRIPPER_ADJUST,
                                        ManipType.MANIP_GRIPPER_OPEN,
                                        ManipType.MANIP_GRIPPER_CLOSE]
        self.MANIP_ACTIONS_TO_STR = {ManipType.MANIP_MOVE_GRASP: "MoveTo Grasp Pose",
                                     ManipType.MANIP_MOVE_POSTGRASP: "MoveTo Post Grasp Pose",
                                     ManipType.MANIP_MOVE_PLACE: "MoveTo Place Pose",
                                     ManipType.MANIP_MOVE_PLACE_ADJUST: "Adjust Gripper place position",
                                     ManipType.MANIP_MOVE_POSTPLACE: "MoveTo Post Place Pose",
                                     ManipType.MANIP_REACH_PREGRASP: "Reach Pre-Grasp Pose",
                                     ManipType.MANIP_REACH_PREPLACE: "Reach Pre-Place Pose",
                                     ManipType.MANIP_GRASP: "Grasp the Object",
                                     ManipType.MANIP_MOVE_GRASP_ADJUST: "Adjust Gripper pick position",
                                     ManipType.MANIP_RELEASE: "Release the Grasped Object",
                                     ManipType.MANIP_GRIPPER_ADJUST: "Adjust Gripper gauge",
                                     ManipType.MANIP_GRIPPER_OPEN: "Open Gripper",
                                     ManipType.MANIP_GRIPPER_CLOSE: "Close Gripper"}

        self.subscriber_callback_group = MutuallyExclusiveCallbackGroup()
        # Services to MoveItWrapper
        # Moveit Motion or Scene changes should not happen in parallel, so use same Mutually exclusive callback group
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self.stop_trajectory_srv = self.create_service(
            Trigger,
            "/stop_trajectory_execution",
            self.stop_trajectory_cb,
            callback_group=self.service_callback_group,
        )

        self.stop_trajectory_publisher = self.create_publisher(String, "/trajectory_execution_event", 10)

        self.action_callback_group = MutuallyExclusiveCallbackGroup()
        self.plan_move_to_server = ActionServer(self, PlanMoveTo, '/plan_move_to',
                                                goal_callback=self.plan_move_to_goal_cb,
                                                cancel_callback=self.plan_move_to_cancel_cb,
                                                #handle_accepted_callback=self.plan_move_to_accepted_cb, # when this is used, execution does not occur
                                                execute_callback=self.plan_move_to_execute_cb,
                                                callback_group=self.action_callback_group)
    
        self.manip_server = ActionServer(self, Manip, '/manipulate',
                                                goal_callback=self.manip_goal_cb,
                                                cancel_callback=self.manip_cancel_cb,
                                                #handle_accepted_callback=self.plan_move_to_accepted_cb, # when this is used, execution does not occur
                                                execute_callback=self.manip_execute_cb,
                                                callback_group=self.action_callback_group)
        # Gripper ActionClient
        self.action_client_callback_group = MutuallyExclusiveCallbackGroup()

        self.gripper_cli = ActionClient(
            self, GripperCommand, "/gripper_controller/gripper_cmd", callback_group = self.action_client_callback_group)
        # Everything related to planning
        self.move_group_cli = ActionClient(self, MoveGroup, '/move_action', callback_group = self.action_client_callback_group)
        self.execute_trajectory_cli = ActionClient(self, ExecuteTrajectory, '/execute_trajectory', callback_group = self.action_client_callback_group)
        self.robot_joint_state = JointState()
        self.default_velocity_scaling_factor = 0.5
        self.saved_plan = None

        self.attach_object_cli = self.create_client(AttachObject, '/attach_object', callback_group = self.service_callback_group)
        self.detach_object_cli = self.create_client(DetachObject, '/detach_object', callback_group = self.service_callback_group)
        self.declare_parameter("tf_prefix", "")
        self.tf_prefix = self.get_parameter("tf_prefix").value
        self.declare_parameter("chain_base_link", "base_link")
        self.chain_base_link = self.get_parameter("chain_base_link").value
        self.declare_parameter("chain_tip_link", "tcp_link")
        self.chain_tip_link = self.get_parameter("chain_tip_link").value
        self.declare_parameter("tool_link", "gripper_link")
        self.tool_link = self.get_parameter("tool_link").value
        self.declare_parameter("allowed_touch_links", ["tcp_link", "gripper_base"])
        self.allowed_touch_links = self.get_parameter("allowed_touch_links").value
        self.declare_parameter("fixed_frame", "world")
        self.fixed_frame = self.get_parameter("fixed_frame").value

        self.tcp_transforms = TCPTransforms(self)
        self.visualization_publisher = VisualizatonPublisher(self)

        self.plan_first = True
        
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.declare_all_parameters()
        self.planning_constraints = self.initialize_constraints()
        self.get_logger().info('Robot Client ready')
    def declare_all_parameters(self):
        self.declare_parameter('planning_group', rclpy.Parameter.Type.STRING)
        self.declare_parameter('trajectory_execution_timeout', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('default_allowed_planning_time', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('default_profile_name', rclpy.Parameter.Type.STRING)
        self.declare_parameter('planner_profiles.planner_ids', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('planner_profiles.num_planning_attempts', rclpy.Parameter.Type.INTEGER_ARRAY)
        self.declare_parameter('planner_profiles.is_cartonly_planners', rclpy.Parameter.Type.BOOL_ARRAY)
        self.declare_parameter('planner_profiles.use_constraints', rclpy.Parameter.Type.BOOL_ARRAY)
        self.declare_parameter('planner_profiles.allowed_planning_times', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('constraints.orientation.frame_id', rclpy.Parameter.Type.STRING)
        self.declare_parameter('constraints.orientation.link_name', rclpy.Parameter.Type.STRING)
        self.declare_parameter('constraints.orientation.orientation', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('constraints.orientation.absolute_tolerance', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('constraints.orientation.weight', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('constraints.box.frame_id', rclpy.Parameter.Type.STRING)
        self.declare_parameter('constraints.box.link_name', rclpy.Parameter.Type.STRING)
        self.declare_parameter('constraints.box.dimensions', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('constraints.box.position', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('constraints.box.orientation', rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('constraints.box.weight', rclpy.Parameter.Type.DOUBLE)

    def initialize_constraints(self):
        orientation_weight = self.get_parameter("constraints.orientation.weight").value
        constraints = Constraints()
        if orientation_weight>1e-3:
            orientation_constraint = OrientationConstraint()
            orientation_constraint.header.frame_id = self.chain_tip_link
            orientation_constraint.link_name = self.chain_tip_link
            orientation = self.get_parameter("constraints.orientation.orientation").value
            orientation_constraint.orientation.x = orientation[0]
            orientation_constraint.orientation.y = orientation[1]
            orientation_constraint.orientation.z = orientation[2]
            orientation_constraint.orientation.w = orientation[3]
            tolerance = self.get_parameter("constraints.orientation.absolute_tolerance").value
            orientation_constraint.absolute_x_axis_tolerance = tolerance[0]
            orientation_constraint.absolute_y_axis_tolerance = tolerance[1]
            orientation_constraint.absolute_z_axis_tolerance = tolerance[2]

            orientation_constraint.weight = orientation_weight
            constraints.orientation_constraints.append(orientation_constraint)
        
        position_weight = self.get_parameter("constraints.box.weight").value
        if position_weight>1e-3:
            position_frame = self.get_parameter("constraints.box.frame_id").value
            position_constraint = PositionConstraint()
            position_constraint.header.frame_id = position_frame
            position_constraint.link_name = self.chain_tip_link

            position_constraint.target_point_offset.x = 0.0
            position_constraint.target_point_offset.y = 0.0
            position_constraint.target_point_offset.z = 0.0

            box_bounding_volume = SolidPrimitive()
            box_bounding_volume.type = 1
            box_dimensions = self.get_parameter("constraints.box.dimensions").value
            box_bounding_volume.dimensions.append(box_dimensions[0])
            box_bounding_volume.dimensions.append(box_dimensions[1])
            box_bounding_volume.dimensions.append(box_dimensions[2])
            position_constraint.constraint_region.primitives.append(box_bounding_volume)

            box_pose = Pose()
            box_position = self.get_parameter("constraints.box.position").value
            box_pose.position.x = box_position[0]
            box_pose.position.y = box_position[1]
            box_pose.position.z = box_position[2]
            box_orientation = self.get_parameter("constraints.box.orientation").value
            box_pose.orientation.x = box_orientation[0]
            box_pose.orientation.y = box_orientation[1]
            box_pose.orientation.z = box_orientation[2]
            box_pose.orientation.w = box_orientation[3]
            position_constraint.constraint_region.primitive_poses.append(box_pose)
            position_constraint.weight = position_weight
            constraints.position_constraints.append(position_constraint)
        return constraints
        

            

    def stop_trajectory_cb(self, request: Trigger.Request, response: Trigger.Response):
        msg = String()
        msg.data = "stop"
        self.stop_trajectory_publisher.publish(msg)
        response.success = True
        return response
    def joint_state_cb(self, msg):
        self.robot_joint_state = msg
        
    def execute(self, plan: RobotTrajectory):
        self.get_logger().info("Executing planned trajectory")
        self.execute_trajectory_cli.wait_for_server()
        goal = ExecuteTrajectory.Goal()

        goal.trajectory = plan
        return self.execute_trajectory_cli.send_goal(goal).result

    def plan(self, pose:Pose, velocity_scaling_factor = 1.0):
        self.move_group_cli.wait_for_server()
        goal = MoveGroup.Goal()
        
        goal.request = MotionPlanRequest()
        goal_pos = PositionConstraint()
        goal_pos.header.frame_id = self.fixed_frame
        goal_pos.link_name = self.chain_tip_link
        goal_pos.target_point_offset.x = 0.0
        goal_pos.target_point_offset.y = 0.0
        goal_pos.target_point_offset.z = 0.0

        goal_bounding_volume = SolidPrimitive()
        goal_bounding_volume.type = 2
        goal_bounding_volume.dimensions.append(1e-4)
        goal_pos.constraint_region.primitives.append(goal_bounding_volume)
        goal_pos.constraint_region.primitive_poses.append(pose)
        goal_pos.weight = 1.0

        goal_ori = OrientationConstraint()
        goal_ori.header.frame_id = self.fixed_frame
        goal_ori.link_name = self.chain_tip_link
        goal_ori.orientation = pose.orientation
        goal_ori.absolute_x_axis_tolerance = 1e-4
        goal_ori.absolute_y_axis_tolerance = 1e-4
        goal_ori.absolute_z_axis_tolerance = 1e-4
        goal_ori.weight = 1.0
        goal_constraint = Constraints()
        goal_constraint.position_constraints.append(goal_pos)
        goal_constraint.orientation_constraints.append(goal_ori)

        goal.request.path_constraints = self.planning_constraints

        goal.request.max_velocity_scaling_factor = velocity_scaling_factor

        goal.request.allowed_planning_time = self.get_parameter("default_allowed_planning_time").value

        goal.request.start_state.joint_state = self.robot_joint_state
        
        goal.request.goal_constraints.append(goal_constraint)

        goal.request.group_name = self.get_parameter("planning_group").value
        goal.request.planner_id = self.get_parameter("default_profile_name").value

        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True

        return self.move_group_cli.send_goal(goal).result

    def send_move_request(self, pose:Pose, cartesian_trajectory:bool=True, planner_profile:str="", velocity_scaling_factor=None, allowed_planning_time:float=0.0, plan_only:bool=False):
        # use default velocity scaling if not defined
        if not velocity_scaling_factor:
            velocity_scaling_factor = self.default_velocity_scaling_factor
            
        if plan_only is True:
            plan_result = self.plan(pose, velocity_scaling_factor = velocity_scaling_factor)
            if plan_result.error_code.val == 1:
                self.saved_plan = plan_result.planned_trajectory
            else:
                self.get_logger().error(f"Planning failed with error code {plan_result.error_code}")
                return False
        else:
            if self.saved_plan is None:
                plan_result = self.plan(pose, velocity_scaling_factor = velocity_scaling_factor)
                if plan_result.error_code.val == 1:
                    self.saved_plan = plan_result.planned_trajectory
                else:
                    self.get_logger().error(f"Planning failed with error code {plan_result.error_code}")
                    return False
            
            execution_result = self.execute(self.saved_plan)
            self.saved_plan = None
            if execution_result.error_code.val == 1:
                return True
            else:
                self.get_logger().error(f"Execution failed with error code {plan_result.error_code}")
                return False
            
        return True
    
    def send_move_seq_request(self, poses:list[Pose], carts:list[bool]=[False], velocity_scaling_factors:list[float]=[0.1],
                                blending_radii:list[float] = [0.05], profiles:list[str]=[""], allowed_planning_time:float=0.0):
        if self.move_seq_cli is None:
            self.get_logger().error(f"Move to Sequence is not available.")
            resp = MoveToPoseSeq.Response()
            resp.success = False
            return resp

        req = MoveToPoseSeq.Request()
        req.poses = poses
        req.carts = carts
        req.planner_profiles = profiles
        req.velocity_scaling_factors = velocity_scaling_factors
        req.blending_radii = blending_radii
        req.allowed_planning_time = allowed_planning_time

        self.get_logger().info(f"Received request to move to pose seq with a list of {len(poses)} poses.")
        future = self.move_seq_cli.call_async(req)
        response = wait_for_response(future, self)
        if not response.success:
            self.get_logger().error(f"Move to {len(poses)} poses has failed.")
            return False
        else:
            self.get_logger().debug(f"Successfully moved to {len(poses)} poses.")
        return True

    def send_stop_request(self):
        self.req_stop =  Trigger.Request()
        response = self.stop_trajectory_exec_cli.call(self.req_stop)
        future = self.stop_trajectory_exec_cli.call_async(self.req_stop)
        response = wait_for_response(future, self)
        if not response.success:
            self.get_logger().fatal("Stopping the robot has failed!")
            return False
        self.get_logger().debug("Robot is successfully stopped.")
        return True

    def plan_move_to_goal_cb(self, goal: PlanMoveTo.Goal):
        self.get_logger().debug('Received new PlanMoveTo goal...')

        if self.active_goal is not None:
            self.get_logger().error('PlanMoveTo goal rejected because there is already an active goal.')
            return GoalResponse.REJECT
        else:
            # validate the goal
            #TODO(gwalck) check the target is a known frame ? otherwise converted it ?
            if len(goal.targets) == 0:
                self.get_logger().error('PlanMoveTo goal rejected because there is no target.')
                return GoalResponse.REJECT
            if len(goal.targets) > 1 and goal.targets[0].blending_radius > 0.0 and self.move_seq_cli is None:
                self.get_logger().error('PlanMoveTo goal rejected because there is no MoveToSeq service available.')
                return GoalResponse.REJECT

            self.active_goal = goal
            self.get_logger().debug('PlanMoveTo goal accepted.')
            return GoalResponse.ACCEPT
        
    def joint_state_cb(self, msg):
        joint_state_to_copy = JointState()
        joint_state_to_copy.name = msg.name[:6]
        joint_state_to_copy.position = msg.position[:6]
        joint_state_to_copy.velocity = msg.velocity[:6]
        joint_state_to_copy.effort = msg.effort[:6]
        self.robot_joint_state = joint_state_to_copy
    
    def plan_move_to_cancel_cb(self, request):
        #TODO handle stop trajectory if we where moving
        if self.send_stop_request():
            self.get_logger().debug(f"Current goal and motion cancelled.")
            return CancelResponse.ACCEPT
        else:
            self.get_logger().error(f"Cannot cancel the current motion.")
            return CancelResponse.REJECT

    def plan_move_to_execute_cb(self, goal_handle: ServerGoalHandle):
        self.get_logger().debug('PlanMoveTo execution started.')
        #self.active_goal = goal_handle
        request = self.active_goal #goal_handle.request
        result = PlanMoveTo.Result()
        result.state.plan_state = PlanExecState.PLAN_UNKNOWN
        result.state.exec_state = PlanExecState.EXEC_UNKNOWN
        
        self.plan_move_to_feedback.state.plan_state = PlanExecState.PLAN_RUNNING
        self.plan_move_to_feedback.state.exec_state = PlanExecState.EXEC_MOVING
        goal_handle.publish_feedback(self.plan_move_to_feedback)
        # do the actual planning and execution
        # only use move seq if blending radious are provided, otherwise use standard move_seq
        if len(request.targets) > 1 and request.targets[0].blending_radius > 0.0: # use MoveToSeq
            poses = []
            carts = []
            planner_profiles = []
            velocity_scaling_factors = []
            blending_radii = []
            for i, target in enumerate(request.targets):
                self.get_logger().warn(f'  Target {i} pose x {target.pose.pose.position.x}.')
                if (target.pose.header.frame_id != self.chain_base_link):
                    pose = self.compute_manip_pose(target.pose)
                else:
                    pose = target.pose.pose
                poses.append(pose)
                carts.append(target.cart)
                planner_profiles.append(target.planner_profile)
                velocity_scaling_factors.append(target.velocity_scaling_factor)
                blending_radii.append(target.blending_radius)
                self.visualization_publisher.publish_pose_as_transform(pose, self.chain_base_link, "mortar_pose_" + str(i), is_static=False)
                
            ret = self.send_move_seq_request(poses, carts, velocity_scaling_factors,
                                  blending_radii, planner_profiles, request.allowed_planning_time)
            self.plan_move_to_feedback.state.plan_state = PlanExecState.PLAN_UNKNOWN
            self.plan_move_to_feedback.state.exec_state = PlanExecState.EXEC_UNKNOWN
            goal_handle.publish_feedback(self.plan_move_to_feedback)

        else:
            for i, target in enumerate(request.targets):
                if (target.pose.header.frame_id != self.fixed_frame):
                    pose = self.compute_manip_pose(target.pose)
                else:
                    pose = target.pose.pose
                self.visualization_publisher.publish_pose_as_transform(pose, self.fixed_frame, "base_position", is_static=True)
                self.plan_move_to_feedback.state.plan_message = f"planning for target {i}/{len(request.targets)}"
                self.plan_move_to_feedback.state.exec_message = f"executing for target {i}/{len(request.targets)}"
                goal_handle.publish_feedback(self.plan_move_to_feedback)

                velocity_scaling_factor = None
                if target.velocity_scaling_factor != 0.0:
                    velocity_scaling_factor = target.velocity_scaling_factor
                ret = self.send_move_request(pose, cartesian_trajectory=target.cart,
                                    planner_profile=target.planner_profile,
                                    velocity_scaling_factor=velocity_scaling_factor,
                                    allowed_planning_time=request.allowed_planning_time)
                if not ret:
                    self.plan_move_to_feedback.state.plan_message = f" failed plan for target {i}/{len(request.targets)}"
                    self.plan_move_to_feedback.state.exec_message = f" failed execution for target {i}/{len(request.targets)}"
                    break

            self.plan_move_to_feedback.state.plan_state = PlanExecState.PLAN_UNKNOWN
            self.plan_move_to_feedback.state.exec_state = PlanExecState.EXEC_UNKNOWN
                
            self.get_logger().debug('PlanMoveTo execution done.')
        self.active_goal = None
        if not goal_handle.is_cancel_requested:
            if ret:
                result.state.plan_state = PlanExecState.PLAN_SUCCESS
                result.state.exec_state = PlanExecState.EXEC_SUCCESS
                result.success = True
                self.get_logger().debug('PlanMoveTo succeeded.')
                goal_handle.succeed()
            else:
                result.state.plan_state = PlanExecState.PLAN_ERROR
                result.state.exec_state = PlanExecState.EXEC_ERROR
                result.success = False
                self.get_logger().debug('PlanMoveTo failed.')
                goal_handle.abort()     
        return result

    def validate_manip_goal(self, goal: Manip.Goal) -> bool:
        if not goal.manipulation_sequence:
            self.get_logger().warn('Manip goal rejected. no manipulation sequence provided.')
            return False
        for manip in goal.manipulation_sequence:
            if manip not in self.SUPPORTED_MANIP_ACTIONS:
                self.get_logger().warn(f'Manip goal rejected. Type ({manip}) not in supported manip types.')
                return False
            #TODO(gwalck)
            # for each manip type check the grasp message contains the desired info
        return True

    def manip_goal_cb(self, goal: Manip.Goal):
        if self.active_goal is not None:
            self.get_logger().warn('Manip goal rejected because there is already an active goal.')
            return GoalResponse.REJECT
        else:
            # validate the goal
            if not self.validate_manip_goal(goal):
               return GoalResponse.REJECT
            self.active_goal = goal
            self.get_logger().debug('Manip goal accepted.')
            return GoalResponse.ACCEPT

    def manip_cancel_cb(self, request):
        #TODO stop current execution cleanly ?
        return CancelResponse.ACCEPT


    def compute_manip_pose(self, source_pose: PoseStamped,
                           use_offset: bool=False, offset_dir: Vector3Stamped=Vector3Stamped(), offset_distance: float=0.0,
                           apply_tool_offset: bool=True) -> Pose:
        pose_source_frame = deepcopy(source_pose)
        self.get_logger().info("GETTING FRAMES" + str(offset_dir.header.frame_id) + "\nand\n" + str(pose_source_frame.header.frame_id))
        if use_offset:
            offset_transformed = self.tcp_transforms.to_from_tcp_vec3_conversion(offset_dir, offset_dir.header.frame_id, pose_source_frame.header.frame_id)
            #  add offset
            dir = np.array([offset_transformed.x, offset_transformed.y, offset_transformed.z])
            offset = dir * offset_distance
            pose_source_frame.pose.position.x += offset[0]
            pose_source_frame.pose.position.y += offset[1]
            pose_source_frame.pose.position.z += offset[2]
        # convert to robot frame
        self.get_logger().debug(f'Manip computed pose source frame ({pose_source_frame.header.frame_id}).')
        return self.tcp_transforms.to_from_tcp_pose_conversion(pose_source_frame.pose,
                                                               pose_source_frame.header.frame_id,
                                                               self.fixed_frame, apply_tool_offset)

    def manip_execute_cb(self, goal_handle: ServerGoalHandle):
        self.get_logger().info('executing')
        request = self.active_goal
        result = Manip.Result()
        result.state.plan_state = PlanExecState.PLAN_UNKNOWN
        result.state.exec_state = PlanExecState.EXEC_UNKNOWN
        self.plan_first = self.active_goal.only_plan

        # initialize feedback
        self.manip_feedback.state.plan_state = PlanExecState.PLAN_UNKNOWN
        self.manip_feedback.state.exec_state = PlanExecState.EXEC_UNKNOWN
        self.manip_feedback.current_manip = ManipType.MANIP_UNKNOWN
        self.manip_feedback.current_step = 0
        goal_handle.publish_feedback(self.manip_feedback)

        # process the sequence
        while(not goal_handle.is_cancel_requested and self.manip_feedback.current_step < len(request.manipulation_sequence)):
            # get current manip
            self.manip_feedback.current_manip = request.manipulation_sequence[self.manip_feedback.current_step]
            manip = self.manip_feedback.current_manip
            self.get_logger().info(f'Manip Exec loop at step {self.manip_feedback.current_step} action type {self.MANIP_ACTIONS_TO_STR[manip]}')
            # update state
            self.manip_feedback.state.plan_state = PlanExecState.PLAN_RUNNING
            self.manip_feedback.state.exec_state = PlanExecState.EXEC_MOVING
            goal_handle.publish_feedback(self.manip_feedback)

            # increment for next step
            self.manip_feedback.current_step +=1

            # extract the correct info from the goal
            # reaching actions (non-cartesian)
            if manip in [ManipType.MANIP_REACH_PREGRASP, ManipType.MANIP_REACH_PREPLACE]:
                if manip == ManipType.MANIP_REACH_PREGRASP:
                    # compute pre-pick pose
                    reach_pose_robot_base_frame = self.compute_manip_pose(request.pick.grasp_pose, True,
                                                                          request.pick.pre_grasp_approach.direction,
                                                                          request.pick.pre_grasp_approach.desired_distance)
                    if reach_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(reach_pose_robot_base_frame, self.fixed_frame, "pre_grasp_pose_base_link", is_static=True)
                if manip == ManipType.MANIP_REACH_PREPLACE:
                    # compute pre-place pose
                    reach_pose_robot_base_frame = self.compute_manip_pose(request.place.place_pose, True,
                                                                          request.place.pre_place_approach.direction,
                                                                          request.place.pre_place_approach.desired_distance)
                    if reach_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(reach_pose_robot_base_frame, self.fixed_frame, "pre_place_pose_base_link", is_static=True)
            
                # perform the action
                # do the actual planning and execution
                if manip == ManipType.MANIP_REACH_PREGRASP:
                    ret = self.send_move_request(reach_pose_robot_base_frame, cartesian_trajectory=False,
                                                 planner_profile=request.planner_profile if request.planner_profile else "ompl", plan_only=self.plan_first)
                if manip == ManipType.MANIP_REACH_PREPLACE:
                    ret = self.send_move_request(reach_pose_robot_base_frame, cartesian_trajectory=False,
                                                             planner_profile=request.planner_profile if request.planner_profile else "ompl_with_constraints", plan_only=self.plan_first)

                if not self.did_manip_plan_succeed(ret, "Reach", goal_handle):
                    result.state.exec_state = PlanExecState.EXEC_ERROR
                    result.state.exec_message = "Failed to reach"
                    self.get_logger().warn(f'reach pre-place with ori {reach_pose_robot_base_frame.orientation} failed')
                    self.get_logger().warn(f'reach pre-place with pos {reach_pose_robot_base_frame.position} failed')
            
                    break
                else:
                    continue
            
            # move actions (cartesian)
            if manip in [ManipType.MANIP_MOVE_GRASP, ManipType.MANIP_MOVE_POSTGRASP, ManipType.MANIP_MOVE_PLACE, ManipType.MANIP_MOVE_POSTPLACE,
                         ManipType.MANIP_MOVE_GRASP_ADJUST, ManipType.MANIP_MOVE_PLACE_ADJUST]:
                # no offset
                if manip == ManipType.MANIP_MOVE_GRASP:
                    # compute pick pose
                    move_pose_robot_base_frame = self.compute_manip_pose(request.pick.grasp_pose)
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.fixed_frame, "grasp_pose_base_link", is_static=True)
                if manip == ManipType.MANIP_MOVE_PLACE:
                    # compute place pose
                    move_pose_robot_base_frame = self.compute_manip_pose(request.place.place_pose)
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.fixed_frame, "place_pose_base_link", is_static=True)
                # with offset
                if manip == ManipType.MANIP_MOVE_POSTGRASP:
                    # compute post-pick pose
                    move_pose_robot_base_frame = self.compute_manip_pose(request.pick.grasp_pose, True,
                                                                         request.pick.post_grasp_retreat.direction,
                                                                         request.pick.post_grasp_retreat.desired_distance)
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.fixed_frame, "post_grasp_pose_base_link", is_static=True)
                if manip == ManipType.MANIP_MOVE_POSTPLACE:
                    # compute post-place pose
                    move_pose_robot_base_frame = self.compute_manip_pose(request.place.place_pose, True,
                                                                         request.place.post_place_retreat.direction,
                                                                         request.place.post_place_retreat.desired_distance)
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.fixed_frame, "post_place_pose_base_link", is_static=True)
                if manip == ManipType.MANIP_MOVE_GRASP_ADJUST:
                    move_pose_robot_base_frame = self.compute_manip_pose(request.pick.grasp_pose, True,
                                                                          request.brick_grasp_clearance_compensation.direction,
                                                                          request.brick_grasp_clearance_compensation.desired_distance)
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.fixed_frame, "pre_grip_pose_compensation_link", is_static=True)
                if manip == ManipType.MANIP_MOVE_PLACE_ADJUST:
                    move_pose_robot_base_frame = self.compute_manip_pose(request.place.place_pose, True,
                                                                          request.brick_grasp_clearance_compensation.direction,
                                                                          request.brick_grasp_clearance_compensation.desired_distance)
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.fixed_frame, "pre_grip_pose_compensation_link", is_static=True)
                # perform the action
                # do the actual planning and execution
                ret = self.send_move_request(move_pose_robot_base_frame, cartesian_trajectory=True,
                                                         velocity_scaling_factor=self.default_velocity_scaling_factor,
                                                         planner_profile=request.planner_profile if request.planner_profile else "pilz_lin", plan_only=self.plan_first)

                if not self.did_manip_plan_succeed(ret, "Move", goal_handle):
                    result.state.exec_state = PlanExecState.EXEC_ERROR
                    result.state.exec_message = "Failed to Move"
                    break
                else:
                    continue
            # Attach/Detach actions
            if manip in [ManipType.MANIP_GRASP, ManipType.MANIP_RELEASE, ManipType.MANIP_GRIPPER_OPEN, ManipType.MANIP_GRIPPER_CLOSE]:
                if manip == ManipType.MANIP_GRASP or manip == ManipType.MANIP_GRIPPER_CLOSE:  
                    # First, we handle gripper actions
                    door_msg = GripperCommand.Goal()
                    door_msg.command.position = 0.022
                    self.gripper_cli.wait_for_server()
                    future = self.gripper_cli.send_goal(door_msg)
                    # additionally handle attach
                    if manip == ManipType.MANIP_GRASP:
                        # if success attach
                        if not request.disable_scene_handling:
                            ret = self.attach(request.object_id, self.tool_link, self.allowed_touch_links)
                        else:
                            continue
                        if not self.did_manip_plan_succeed(ret, "Gripper attach/detach/adjust", goal_handle):
                            result.state.exec_state = PlanExecState.EXEC_ERROR
                            result.state.exec_message = "Failed to attach/detach"
                            break
                        else:
                            continue
                if manip == ManipType.MANIP_RELEASE or manip == ManipType.MANIP_GRIPPER_OPEN:
                    # First, we handle gripper actions
                    door_msg = GripperCommand.Goal()
                    door_msg.command.position = 0.0
                    self.gripper_cli.wait_for_server()
                    future = self.gripper_cli.send_goal(door_msg)
                    # Additionally handle detach
                    if manip == ManipType.MANIP_RELEASE:
                        #if success detach
                        if not request.disable_scene_handling:
                            ret = self.detach(request.object_id)
                        else:
                            continue
                        if not self.did_manip_plan_succeed(ret, "Gripper attach/detach/adjust", goal_handle):
                            result.state.exec_state = PlanExecState.EXEC_ERROR
                            result.state.exec_message = "Failed to attach/detach"
                            break
                        else:
                            continue
            # end while loop
        if goal_handle.status != GoalStatus.STATUS_ABORTED:
            result.success = True
            goal_handle.succeed()
        self.active_goal = None
        return result

    def did_manip_plan_succeed(self, plan_exec_success: bool, action_name: str, goal_handle):
        if plan_exec_success:
            self.manip_feedback.state.plan_state = PlanExecState.PLAN_SUCCESS
            self.manip_feedback.state.exec_state = PlanExecState.EXEC_SUCCESS
            goal_handle.publish_feedback(self.manip_feedback)
            self.get_logger().debug(f'{action_name} succeeded.')
            return True
        else:
            self.manip_feedback.state.plan_state = PlanExecState.PLAN_ERROR
            self.manip_feedback.state.exec_state = PlanExecState.EXEC_ERROR
            goal_handle.publish_feedback(self.manip_feedback)
            self.get_logger().debug(f'{action_name} failed.')
            goal_handle.abort()
            return False
    
    def attach(self, id:str, attach_link:str, allowed_touch_links:list[str]):
        req = AttachObject.Request()
        req.id = id
        req.link_name = attach_link
        req.touch_links = allowed_touch_links
        response = self.attach_object_cli.call(req)
        if response.result.state != ServiceResult.SUCCESS:
            self.get_logger().error(f"Attach object {id} has failed.")
            return False
        self.get_logger().info(f"Successfully attached object {id} to {attach_link}.")
        return True
    
    def detach(self, id:str):
        req = DetachObject.Request()
        req.id = id
        response = self.detach_object_cli.call(req)
        if response.result.state != ServiceResult.SUCCESS:
            self.get_logger().error(f"Detach object {id} has failed.")
            return False
        self.get_logger().info(f"Successfully detached object {id}.")
        return True

def main(args=None):

    rclpy.init(args=args)

    executor = MultiThreadedExecutor()
    
    mc = RobotClient()

    try:
        rclpy.spin(mc, executor)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()