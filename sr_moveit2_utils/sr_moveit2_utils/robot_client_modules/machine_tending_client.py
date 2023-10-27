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
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle, GoalStatus
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sr_manipulation_interfaces.action import PlanMoveTo, Manip
from sr_manipulation_interfaces.msg import ManipType, PlanExecState, ServiceResult
from sr_manipulation_interfaces.srv import AttachObject, DetachObject

from geometry_msgs.msg import PoseStamped, Vector3, Vector3Stamped, Pose
from moveit_msgs.msg import Grasp
from sr_moveit2_utils.move_client import MoveClient
import moveit_msgs.msg
from sr_ros2_python_utils.transforms import TCPTransforms
import numpy as np

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


class MachineTendingClient(Node):
    def __init__(self):
        """
        Create a new client for managing manipulation requests.
        """
        super().__init__('machine_tending_client')

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
        # Services to MoveItWrapper
        # Moveit Motion or Scene changes should not happen in parallel, so use same Mutually exclusive callback group
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self.move_client = MoveClient(default_velocity_scaling_factor=1.0)

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
        
        self.tcp_transforms = TCPTransforms(self)

        self.chain_base_link = "base_link"

        self.get_logger().info('Robot Client ready')


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
            if len(goal.targets) > 1 and goal.targets[0].blending_radius > 0.0 and self.move_client.move_seq_cli is None:
                self.get_logger().error('PlanMoveTo goal rejected because there is no MoveToSeq service available.')
                return GoalResponse.REJECT

            self.active_goal = goal
            self.get_logger().debug('PlanMoveTo goal accepted.')
            return GoalResponse.ACCEPT
    
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
                
            ret = self.move_client.send_move_seq_request(poses, carts, velocity_scaling_factors,
                                  blending_radii, planner_profiles, request.allowed_planning_time)
            self.plan_move_to_feedback.state.plan_state = PlanExecState.PLAN_UNKNOWN
            self.plan_move_to_feedback.state.exec_state = PlanExecState.EXEC_UNKNOWN
            goal_handle.publish_feedback(self.plan_move_to_feedback)

        else:
            for i, target in enumerate(request.targets):
                if (target.pose.header.frame_id != self.chain_base_link):
                    pose = self.compute_manip_pose(target.pose)
                else:
                    pose = target.pose.pose

                self.plan_move_to_feedback.state.plan_message = f"planning for target {i}/{len(request.targets)}"
                self.plan_move_to_feedback.state.exec_message = f"executing for target {i}/{len(request.targets)}"
                goal_handle.publish_feedback(self.plan_move_to_feedback)

                velocity_scaling_factor = None
                if target.velocity_scaling_factor != 0.0:
                    velocity_scaling_factor = target.velocity_scaling_factor
                ret = self.move_client.send_move_request(pose, cartesian_trajectory=target.cart,
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
        request.manipulation_sequence[0]

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
                                                               self.chain_base_link, apply_tool_offset)

    def manip_execute_cb(self, goal_handle: ServerGoalHandle):
        self.get_logger().info('executing')
        request = self.active_goal
        result = Manip.Result()
        result.state.plan_state = PlanExecState.PLAN_UNKNOWN
        result.state.exec_state = PlanExecState.EXEC_UNKNOWN

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
                    # self.visualization_publisher.publish_pose_as_transform(reach_pose_robot_base_frame, self.chain_base_link, "pre_grasp_pose_base_link", is_static=True)
                if manip == ManipType.MANIP_REACH_PREPLACE:
                    # compute pre-place pose
                    reach_pose_robot_base_frame = self.compute_manip_pose(request.place.place_pose, True,
                                                                          request.place.pre_place_approach.direction,
                                                                          request.place.pre_place_approach.desired_distance)
                    if reach_pose_robot_base_frame is None:
                        break
                    # self.visualization_publisher.publish_pose_as_transform(reach_pose_robot_base_frame, self.chain_base_link, "pre_place_pose_base_link", is_static=True)
            
                # perform the action
                # do the actual planning and execution

                if manip == ManipType.MANIP_REACH_PREGRASP:
                    ret = self.move_client.send_move_request(reach_pose_robot_base_frame, cartesian_trajectory=False,
                                                 planner_profile=request.planner_profile if request.planner_profile else "ompl")
                if manip == ManipType.MANIP_REACH_PREPLACE:
                    ret = self.move_client.send_move_request(reach_pose_robot_base_frame, cartesian_trajectory=False,
                                                             planner_profile=request.planner_profile if request.planner_profile else "ompl_with_constraints")

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
                    # self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.chain_base_link, "grasp_pose_base_link", is_static=True)
                if manip == ManipType.MANIP_MOVE_PLACE:
                    # compute place pose
                    move_pose_robot_base_frame = self.compute_manip_pose(request.place.place_pose)
                    if move_pose_robot_base_frame is None:
                        break
                    # self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.chain_base_link, "place_pose_base_link", is_static=True)
                # with offset
                if manip == ManipType.MANIP_MOVE_POSTGRASP:
                    # compute post-pick pose
                    move_pose_robot_base_frame = self.compute_manip_pose(request.pick.grasp_pose, True,
                                                                         request.pick.post_grasp_retreat.direction,
                                                                         request.pick.post_grasp_retreat.desired_distance)
                    if move_pose_robot_base_frame is None:
                        break
                    # self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.chain_base_link, "post_grasp_pose_base_link", is_static=True)
                if manip == ManipType.MANIP_MOVE_POSTPLACE:
                    # compute post-place pose
                    move_pose_robot_base_frame = self.compute_manip_pose(request.place.place_pose, True,
                                                                         request.place.post_place_retreat.direction,
                                                                         request.place.post_place_retreat.desired_distance)
                    if move_pose_robot_base_frame is None:
                        break
                    # self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.chain_base_link, "post_place_pose_base_link", is_static=True)
                if manip == ManipType.MANIP_MOVE_GRASP_ADJUST:
                    move_pose_robot_base_frame = self.compute_manip_pose(request.pick.grasp_pose, True,
                                                                          request.brick_grasp_clearance_compensation.direction,
                                                                          request.brick_grasp_clearance_compensation.desired_distance)
                    if move_pose_robot_base_frame is None:
                        break
                    # self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.chain_base_link, "pre_grip_pose_compensation_link", is_static=True)
                if manip == ManipType.MANIP_MOVE_PLACE_ADJUST:
                    move_pose_robot_base_frame = self.compute_manip_pose(request.place.place_pose, True,
                                                                          request.brick_grasp_clearance_compensation.direction,
                                                                          request.brick_grasp_clearance_compensation.desired_distance)
                    if move_pose_robot_base_frame is None:
                        break
                    # self.visualization_publisher.publish_pose_as_transform(move_pose_robot_base_frame, self.chain_base_link, "pre_grip_pose_compensation_link", is_static=True)
                # perform the action
                # do the actual planning and execution
                ret = self.move_client.send_move_request(move_pose_robot_base_frame, cartesian_trajectory=True,
                                                         velocity_scaling_factor=0.1,
                                                         planner_profile=request.planner_profile if request.planner_profile else "pilz_lin")

                if not self.did_manip_plan_succeed(ret, "Move", goal_handle):
                    result.state.exec_state = PlanExecState.EXEC_ERROR
                    result.state.exec_message = "Failed to Move"
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
                

        self.get_logger().debug('PlanMoveTo execution done.')
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
        self.active_goal = None          
        return result

    def send_move_request(self, pose:Pose, cartesian_trajectory:bool=True, planner_profile:str="", velocity_scaling_factor=None):
        # use default velocity scaling if not defined
        if not velocity_scaling_factor:
            velocity_scaling_factor = self.default_velocity_scaling_factor
