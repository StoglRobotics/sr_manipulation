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
from typing import List, Optional, Sequence

import rclpy
import rclpy.duration
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse, ActionClient
from rclpy.action.server import ServerGoalHandle, GoalStatus
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from moveit_msgs.msg import Constraints, MotionSequenceItem, MoveItErrorCodes, OrientationConstraint, PositionConstraint
from moveit_msgs.action import MoveGroupSequence, ExecuteTrajectory_GetResult_Response
from shape_msgs.msg import SolidPrimitive
from sr_manipulation_interfaces.action import PlanMoveTo, Manip
from sr_manipulation_interfaces.msg import (
    ManipType,
    MoveWaypoint,
    PlanExecState,
    ServiceResult,
    TouchLinks,
)
from sr_manipulation_interfaces.srv import (
    ApplyTouchLinks,
    AttachObject,
    DetachObject,
    ExecuteJointTrajectory,
)

from control_msgs.action import GripperCommand, ParallelGripperCommand
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import PoseStamped, Vector3Stamped, Pose
from sr_moveit2_utils.moveit_client import MoveitClient
from sr_ros2_python_utils.transforms import TCPTransforms
from sr_ros2_python_utils.visualization_publishers import VisualizatonPublisher
import numpy as np
from std_srvs.srv import Trigger
from std_msgs.msg import String, Float64
from asyncio import Future


def wait_for_response(future, client):
    while rclpy.ok():
        rclpy.spin_once(client)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                client.get_logger().info(
                    f"Service call to move_to_pose or move_to_pose_seq failed {e!r}"
                )
                return None
            else:
                return response




def construct_goal_constraints(link_name: str, pose: PoseStamped,
                                tolerance_pos: float=1e-3, tolerance_angle: float = 1e-2):
    # Create the goal constraints message
    goal = Constraints()

    # Create and configure the position constraint
    position_constraint = PositionConstraint()
    position_constraint.link_name = link_name
    position_constraint.target_point_offset.x = 0.0
    position_constraint.target_point_offset.y = 0.0
    position_constraint.target_point_offset.z = 0.0

    # Define a spherical constraint region
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [tolerance_pos]  # Sphere radius
    position_constraint.constraint_region.primitives.append(sphere)

    # Position of the sphere
    position_constraint.constraint_region.primitive_poses.append(pose.pose)
    position_constraint.header = pose.header
    position_constraint.weight = 1.0

    # Add the position constraint to the goal
    goal.position_constraints.append(position_constraint)

    # Create and configure the orientation constraint
    orientation_constraint = OrientationConstraint()
    orientation_constraint.parameterization = OrientationConstraint.ROTATION_VECTOR
    orientation_constraint.link_name = link_name
    orientation_constraint.header = pose.header
    orientation_constraint.orientation = pose.pose.orientation
    orientation_constraint.absolute_x_axis_tolerance = tolerance_angle
    orientation_constraint.absolute_y_axis_tolerance = tolerance_angle
    orientation_constraint.absolute_z_axis_tolerance = tolerance_angle
    orientation_constraint.weight = 1.0

    # Add the orientation constraint to the goal
    goal.orientation_constraints.append(orientation_constraint)

    return goal

class RobotClient(Node):
    def __init__(self):
        """Create a new client for managing manipulation requests."""
        super().__init__("robot_client_node")

        # defaults
        self.active_goal = None
        self.plan_move_to_feedback = PlanMoveTo.Feedback()
        self.manip_feedback = Manip.Feedback()
        self.MANIP_ACTIONS_TO_STR = {
            ManipType.MANIP_MOVE_GRASP: "MoveTo Grasp Pose",
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
            ManipType.MANIP_GRIPPER_CLOSE: "Close Gripper",
            ManipType.EXECUTE_MOVE_SEQUENCE: "Execute the sequence stored in the queue."
        }
        self.SUPPORTED_MANIP_ACTIONS = list(self.MANIP_ACTIONS_TO_STR.keys())

        self.subscriber_callback_group = MutuallyExclusiveCallbackGroup()
        # Services to MoveItWrapper
        # Moveit Motion or Scene changes should not happen in parallel,
        # so use same Mutually exclusive callback group
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self.stop_trajectory_srv = self.create_service(
            Trigger,
            "/stop_trajectory_execution",
            self.stop_trajectory_cb,
            callback_group=self.service_callback_group,
        )
        self.execute_joint_trajectory_srv = self.create_service(
            ExecuteJointTrajectory,
            "/execute_joint_trajectory",
            self.execute_joint_trajectory_cb,
            callback_group=self.service_callback_group,
        )

        self.trajectory_exec_time_publisher = self.create_publisher(
            Float64, "/trajectory_execution_time", 10
        )

        self.stop_trajectory_publisher = self.create_publisher(
            String, "/trajectory_execution_event", 10
        )

        self.action_callback_group = MutuallyExclusiveCallbackGroup()
        self.plan_move_to_server = ActionServer(
            self,
            PlanMoveTo,
            "/plan_move_to",
            goal_callback=self.plan_move_to_goal_cb,
            cancel_callback=self.plan_move_to_cancel_cb,
            # handle_accepted_callback=self.plan_move_to_accepted_cb, # when this is used, execution does not occur
            execute_callback=self.plan_move_to_execute_cb,
            callback_group=self.action_callback_group,
        )

        self.manip_server = ActionServer(
            self,
            Manip,
            "/manipulate",
            goal_callback=self.manip_goal_cb,
            cancel_callback=self.manip_cancel_cb,
            # handle_accepted_callback=self.plan_move_to_accepted_cb, # when this is used, execution does not occur
            execute_callback=self.manip_execute_cb,
            callback_group=self.action_callback_group,
        )
        # TODO (anyone) please remove this!
        gripper_cmd_action_names = self.declare_parameter(
            "gripper_cmd_action_names", rclpy.Parameter.Type.STRING_ARRAY
        ).value
        if not gripper_cmd_action_names:
            gripper_cmd_action_names = []
        self.get_logger().info(f"Gripper action names: {gripper_cmd_action_names}")

        self.default_gripper_cmd_action_name = None
        if gripper_cmd_action_names:
            self.default_gripper_cmd_action_name = gripper_cmd_action_names[0]
      
        self.move_sequence : List[MotionSequenceItem] = []
        self.saved_plan = None

        self.attach_object_cli = self.create_client(
            AttachObject, "/attach_object", callback_group=self.service_callback_group
        )
        self.detach_object_cli = self.create_client(
            DetachObject, "/detach_object", callback_group=self.service_callback_group
        )
        self.apply_touch_links_cli = self.create_client(
            ApplyTouchLinks, "/apply_touch_links", callback_group=self.service_callback_group
        )
        self.tf_prefix = self.declare_parameter("tf_prefix", "").value
        self.chain_base_link = self.declare_parameter("chain_base_link", "base_link").value
        self.chain_tip_link = self.declare_parameter("chain_tip_link", "tcp_link").value
        self.tool_link = self.declare_parameter("tool_link", "gripper_link").value
        self.allowed_touch_links = self.declare_parameter(
            "allowed_touch_links", ["tcp_link", "gripper_base"]
        ).value
        self.fixed_frame = self.declare_parameter("fixed_frame", "world").value
        self.default_velocity_scaling_factor = self.declare_parameter(
            "default_velocity_scaling_factor", 0.1
        ).value
        self.default_acceleration_scaling_factor = self.declare_parameter(
            "default_acceleration_scaling_factor", 0.1
        ).value
        
        self.wait_for_action_server_timeout = self.declare_parameter(
            "wait_for_action_server_timeout", 5.0
        ).value

        self.use_parallel_gripper = self.declare_parameter(
            "use_parallel_gripper", True
        ).value

        self.get_logger().info(
            "Loaded params:\n"
            f"- tf_prefix: {self.tf_prefix}\n"
            f"- chain_base_link: {self.chain_base_link}\n"
            f"- chain_tip_link: {self.chain_tip_link}\n"
            f"- tool_link: {self.tool_link}\n"
            f"- allowed_touch_links: {self.allowed_touch_links}\n"
            f"- fixed_frame: {self.fixed_frame}\n"
            f"- default_velocity_scaling_factor: {self.default_velocity_scaling_factor}\n"
            f"- default_acceleration_scaling_factor: {self.default_acceleration_scaling_factor}\n"
            f"- use_parallel_gripper: {self.use_parallel_gripper}"
        )

                # Gripper ActionClients
        self.action_client_callback_group = MutuallyExclusiveCallbackGroup()
        if self.use_parallel_gripper: 
            self.gripper_clients = {
                gripper_cmd_action_name: ActionClient(
                    self,
                    ParallelGripperCommand,
                    gripper_cmd_action_name,
                    callback_group=self.action_client_callback_group,
                )
                for gripper_cmd_action_name in gripper_cmd_action_names
            }
        else:
            self.gripper_clients = {
                gripper_cmd_action_name: ActionClient(
                    self,
                    GripperCommand,
                    gripper_cmd_action_name,
                    callback_group=self.action_client_callback_group,
                )
                for gripper_cmd_action_name in gripper_cmd_action_names
            }

        self.move_sequence_action_client = ActionClient(
            self,
            MoveGroupSequence,
            "/sequence_move_group",
            callback_group=self.action_client_callback_group,
        )

        self.moveit_client = MoveitClient(
            node=self,
            pose_reference_frame=self.fixed_frame,
            end_effector_link=self.chain_tip_link,
        )

        self.tcp_transforms = TCPTransforms(self)
        self.visualization_publisher = VisualizatonPublisher(self)

        self.plan_first = True

        self.get_logger().info("Robot Client ready")

    def execute_joint_trajectory_cb(
        self,
        request: ExecuteJointTrajectory.Request,
        response: ExecuteJointTrajectory.Response,
    ):
        self.get_logger().info("Received new joint trajectory request...")
        start: Time = self.get_clock().now()
        success = self.moveit_client.execute_joint_trajectory(
            request.joint_trajectory, request.controller_names
        )
        exec_duration: rclpy.duration.Duration = self.get_clock().now() - start
        if success:
            self.get_logger().info("Execution succeeded in execute_joint_trajectory_cb")
            self.publish_execution_time(exec_duration, "execute_joint_trajectory_cb")
        else:
            self.get_logger().error("Execution failed in execute_joint_trajectory_cb")
        response.success = success
        return response

    def stop_trajectory_cb(self, request: Trigger.Request, response: Trigger.Response):
        msg = String()
        msg.data = "stop"
        self.stop_trajectory_publisher.publish(msg)
        response.success = True
        return response

    def publish_execution_time(self, duration: rclpy.duration.Duration, name: str):
        exec_time_s = duration.nanoseconds / 1e9
        self.get_logger().info(f"Execution duration of '{name}': {exec_time_s:.3f} seconds.")
        msg = Float64()
        msg.data = exec_time_s
        self.trajectory_exec_time_publisher.publish(msg)

    def send_move_request(
        self,
        pose: Pose,
        cartesian_trajectory: bool,
        planner_profile: str,
        plan_only: bool,
        end_effector_link: str = None,
        planning_group: str = None,
        velocity_scaling_factor: Optional[float] = None,
        acceleration_scaling_factor: Optional[float] = None,
        allowed_planning_time: Optional[float] = None,
        controller_names: Optional[List[str]] = None,
        preempt_ok: bool = False,
    ):
        if not velocity_scaling_factor:
            velocity_scaling_factor = self.default_velocity_scaling_factor
        if not acceleration_scaling_factor:
            acceleration_scaling_factor = self.default_acceleration_scaling_factor

        move_request_args_msg = (
            f"\npose={pose}, "
            f"\ncartesian_trajectory={cartesian_trajectory}, "
            f"\nplanner_profile={planner_profile}, "
            f"\nplan_only={plan_only}, "
            f"\nplanning_group={planning_group}, "
            f"\nvelocity_scaling_factor={velocity_scaling_factor}, "
            f"\nacceleration_scaling_factor={acceleration_scaling_factor}, "
            f"\nallowed_planning_time={allowed_planning_time}, "
            f"\ncontroller_names={controller_names}, "
            f"\npreempt_ok={preempt_ok}"
        )
        self.get_logger().info(f"Sending move request with {move_request_args_msg}")

        if plan_only is True or self.saved_plan is None:
            planned_trajectory = self.moveit_client.plan(
                pose=pose,
                end_effector_link=end_effector_link,
                cartesian_trajectory=cartesian_trajectory,
                planner_profile=planner_profile,
                planning_group=planning_group,
                velocity_scaling_factor=velocity_scaling_factor,
                acceleration_scaling_factor=acceleration_scaling_factor,
                allowed_planning_time=allowed_planning_time,
            )
            if planned_trajectory:
                self.saved_plan = planned_trajectory
                self.get_logger().info("Planning succeeded in send_move_request")
            else:
                self.get_logger().error(
                    f"Planning failed in send_move_request with {move_request_args_msg}"
                )
                return False

        # no need to execute if plan_only is True
        if plan_only:
            return True

        start: Time = self.get_clock().now()
        exec_success = self.moveit_client.execute(
            plan=self.saved_plan,
            controller_names=controller_names,
            preempt_ok=preempt_ok,
        )
        exec_duration: rclpy.duration.Duration = self.get_clock().now() - start

        self.saved_plan = None
        if exec_success:
            self.get_logger().info("Execution succeeded in send_move_request")
            self.publish_execution_time(exec_duration, "send_move_request")
            return True
        else:
            self.get_logger().error(
                f"Execution failed in send_move_request with {move_request_args_msg}"
            )
            return False

    def send_stop_request(self):
        self.req_stop = Trigger.Request()
        response = self.stop_trajectory_exec_cli.call(self.req_stop)
        future = self.stop_trajectory_exec_cli.call_async(self.req_stop)
        response = wait_for_response(future, self)
        if not response.success:
            self.get_logger().fatal("Stopping the robot has failed!")
            return False
        self.get_logger().info("Robot is successfully stopped.")
        return True

    def plan_move_to_goal_cb(self, goal: PlanMoveTo.Goal):
        self.get_logger().info("Received new PlanMoveTo goal...")

        if self.active_goal is not None:
            self.get_logger().error(
                "PlanMoveTo goal rejected because there is already an active goal."
            )
            return GoalResponse.REJECT
        else:
            # validate the goal
            # TODO(gwalck) check the target is a known frame ? otherwise converted it ?
            if len(goal.targets) == 0:
                self.get_logger().error("PlanMoveTo goal rejected because there is no target.")
                return GoalResponse.REJECT
            if (
                len(goal.targets) > 1
                and goal.targets[0].blending_radius > 0.0
                and self.move_seq_cli is None
            ):
                self.get_logger().error(
                    "PlanMoveTo goal rejected because there is no MoveToSeq service available."
                )
                return GoalResponse.REJECT

            self.active_goal = goal
            self.get_logger().info("PlanMoveTo goal accepted.")
            return GoalResponse.ACCEPT

    def plan_move_to_cancel_cb(self, request):
        # TODO handle stop trajectory if we where moving
        if self.send_stop_request():
            self.get_logger().info("Current goal and motion cancelled.")
            return CancelResponse.ACCEPT
        else:
            self.get_logger().error("Cannot cancel the current motion.")
            return CancelResponse.REJECT

    def plan_move_to_execute_cb(self, goal_handle: ServerGoalHandle):
        self.get_logger().info("PlanMoveTo executing...")
        # self.active_goal = goal_handle
        request: PlanMoveTo.Goal = self.active_goal  # goal_handle.request
        result = PlanMoveTo.Result()
        result.state.plan_state = PlanExecState.PLAN_UNKNOWN
        result.state.exec_state = PlanExecState.EXEC_UNKNOWN

        self.plan_move_to_feedback.state.plan_state = PlanExecState.PLAN_RUNNING
        self.plan_move_to_feedback.state.exec_state = PlanExecState.EXEC_MOVING
        goal_handle.publish_feedback(self.plan_move_to_feedback)
        # do the actual planning and execution
        self.get_logger().info("PlanMoveTo using MoveTo.")
        targets: Sequence[MoveWaypoint] = request.targets
        for i, target in enumerate(targets):
            if target.pose.header.frame_id != self.fixed_frame:
                pose = self.compute_manip_pose(target.pose)
            else:
                pose = target.pose.pose
            self.visualization_publisher.publish_pose_as_transform(
                pose, self.fixed_frame, "base_position", is_static=True
            )
            self.plan_move_to_feedback.state.plan_message = (
                f"planning for target {i}/{len(request.targets)}"
            )
            self.plan_move_to_feedback.state.exec_message = (
                f"executing for target {i}/{len(request.targets)}"
            )
            goal_handle.publish_feedback(self.plan_move_to_feedback)

            ret = self.send_move_request(
                pose,
                end_effector_link=target.end_effector_link,
                cartesian_trajectory=target.cart,
                planner_profile=target.planner_profile,
                velocity_scaling_factor=target.velocity_scaling_factor,
                allowed_planning_time=request.allowed_planning_time,
                plan_only=request.only_plan,
                planning_group=target.planning_group,
                controller_names=(request.controller_names if request.controller_names else None),
                preempt_ok=(request.preempt_ok if request.preempt_ok else False),
            )
            if not ret:
                self.plan_move_to_feedback.state.plan_message = (
                    f" failed plan for target {i}/{len(request.targets)}"
                )
                self.plan_move_to_feedback.state.exec_message = (
                    f" failed execution for target {i}/{len(request.targets)}"
                )
                break

        self.plan_move_to_feedback.state.plan_state = PlanExecState.PLAN_UNKNOWN
        self.plan_move_to_feedback.state.exec_state = PlanExecState.EXEC_UNKNOWN

        self.get_logger().info("PlanMoveTo execution done.")

        self.active_goal = None
        if not goal_handle.is_cancel_requested:
            if ret:
                result.state.plan_state = PlanExecState.PLAN_SUCCESS
                result.state.exec_state = PlanExecState.EXEC_SUCCESS
                result.success = True
                self.get_logger().info("PlanMoveTo succeeded.")
                goal_handle.succeed()
            else:
                result.state.plan_state = PlanExecState.PLAN_ERROR
                result.state.exec_state = PlanExecState.EXEC_ERROR
                result.success = False
                self.get_logger().info("PlanMoveTo failed.")
                goal_handle.abort()
        return result

    def validate_manip_goal(self, goal: Manip.Goal) -> bool:
        if not goal.manipulation_sequence:
            self.get_logger().warn("Manip goal rejected. no manipulation sequence provided.")
            return False
        for manip in goal.manipulation_sequence:
            if manip not in self.SUPPORTED_MANIP_ACTIONS:
                self.get_logger().warn(
                    f"Manip goal rejected. Type ({manip}) not in supported manip types."
                )
                return False
            # TODO(gwalck)
            # for each manip type check the grasp message contains the desired info
        return True

    def manip_goal_cb(self, goal: Manip.Goal):
        if self.active_goal is not None:
            self.get_logger().warn("Manip goal rejected because there is already an active goal.")
            return GoalResponse.REJECT
        else:
            # validate the goal
            if not self.validate_manip_goal(goal):
                return GoalResponse.REJECT
            self.active_goal = goal
            self.get_logger().info("Manip goal accepted.")
            return GoalResponse.ACCEPT

    def manip_cancel_cb(self, request):
        # TODO stop current execution cleanly ?
        return CancelResponse.ACCEPT

    def compute_manip_pose(
        self,
        source_pose: PoseStamped,
        use_offset: bool = False,
        offset_dir: Vector3Stamped = Vector3Stamped(),
        offset_distance: float = 0.0,
        apply_tool_offset: bool = True,
    ) -> Pose:
        pose_source_frame = deepcopy(source_pose)
        self.get_logger().info(
            "GETTING FRAMES: "
            f"offset_dir.header.frame_id={offset_dir.header.frame_id}, "
            f"pose_source_frame.header.frame_id={pose_source_frame.header.frame_id}"
        )
        if use_offset:
            offset_transformed = self.tcp_transforms.to_from_tcp_vec3_conversion(
                offset_dir,
                offset_dir.header.frame_id,
                pose_source_frame.header.frame_id,
            )
            #  add offset
            dir = np.array([offset_transformed.x, offset_transformed.y, offset_transformed.z])
            offset = dir * offset_distance
            pose_source_frame.pose.position.x += offset[0]
            pose_source_frame.pose.position.y += offset[1]
            pose_source_frame.pose.position.z += offset[2]
        # convert to robot frame
        self.get_logger().info(
            f"Manip computed pose source frame ({pose_source_frame.header.frame_id})."
        )
        return self.tcp_transforms.to_from_tcp_pose_conversion(
            pose_source_frame.pose,
            pose_source_frame.header.frame_id,
            self.fixed_frame,
            apply_tool_offset,
        )
    
    def create_move_seq_item(self, request: Manip.Goal, pose: Pose):
        item = MotionSequenceItem()
        item.blend_radius = request.blend_radius
        item.req.group_name = request.planning_group
        item.req.planner_id = "LIN" if "lin" in request.planner_profile.lower() else "PTP"
        item.req.pipeline_id = "pilz"
        item.req.allowed_planning_time = 5.0
        item.req.max_velocity_scaling_factor = request.velocity_scaling_factor
        item.req.max_acceleration_scaling_factor = request.acceleration_scaling_factor
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.fixed_frame
        pose_stamped.pose = pose
        item.req.goal_constraints.append(construct_goal_constraints(link_name=request.end_effector_link, pose=pose_stamped))
        return item

    def manip_execute_cb(self, goal_handle: ServerGoalHandle):
        self.get_logger().info("executing")
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
        while not goal_handle.is_cancel_requested and self.manip_feedback.current_step < len(
            request.manipulation_sequence
        ):
            # get current manip
            self.manip_feedback.current_manip = request.manipulation_sequence[
                self.manip_feedback.current_step
            ]
            manip = self.manip_feedback.current_manip
            self.get_logger().info(
                f"Manip Exec loop at step {self.manip_feedback.current_step} "
                f"action type {self.MANIP_ACTIONS_TO_STR[manip]}"
            )
            # update state
            self.manip_feedback.state.plan_state = PlanExecState.PLAN_RUNNING
            self.manip_feedback.state.exec_state = PlanExecState.EXEC_MOVING
            goal_handle.publish_feedback(self.manip_feedback)

            # increment for next step
            self.manip_feedback.current_step += 1

            # extract the correct info from the goal
            # reaching actions (non-cartesian)
            if manip in [
                ManipType.MANIP_REACH_PREGRASP,
                ManipType.MANIP_REACH_PREPLACE,
            ]:
                if manip == ManipType.MANIP_REACH_PREGRASP:
                    # compute pre-pick pose
                    reach_pose_robot_base_frame = self.compute_manip_pose(
                        request.pick.grasp_pose,
                        True,
                        request.pick.pre_grasp_approach.direction,
                        request.pick.pre_grasp_approach.desired_distance,
                    )
                    if reach_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(
                        reach_pose_robot_base_frame,
                        self.fixed_frame,
                        "pre_grasp_pose_base_link",
                        is_static=True,
                    )
                if manip == ManipType.MANIP_REACH_PREPLACE:
                    # compute pre-place pose
                    reach_pose_robot_base_frame = self.compute_manip_pose(
                        request.place.place_pose,
                        True,
                        request.place.pre_place_approach.direction,
                        request.place.pre_place_approach.desired_distance,
                    )
                    if reach_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(
                        reach_pose_robot_base_frame,
                        self.fixed_frame,
                        "pre_place_pose_base_link",
                        is_static=True,
                    )


                if request.add_to_blending_queue:
                    self.move_sequence.append(self.create_move_seq_item(request, reach_pose_robot_base_frame))
                else:
                    # perform the action
                    # do the actual planning and execution
                    if manip == ManipType.MANIP_REACH_PREGRASP:
                            ret = self.send_move_request(
                                reach_pose_robot_base_frame,
                                end_effector_link=request.end_effector_link,
                                cartesian_trajectory=request.cartesian_trajectory,
                                planner_profile=(
                                    # request.planner_profile if request.planner_profile else "ompl"
                                    request.planner_profile
                                    if request.planner_profile
                                    else "pilz_ptp"
                                ),
                                plan_only=self.plan_first,
                                planning_group=request.planning_group,
                                controller_names=(
                                    request.controller_names if request.controller_names else None
                                ),
                                velocity_scaling_factor=(
                                    request.velocity_scaling_factor
                                    if request.velocity_scaling_factor
                                    else None
                                ),
                                acceleration_scaling_factor=(
                                    request.acceleration_scaling_factor
                                    if request.acceleration_scaling_factor
                                    else None
                                ),
                                preempt_ok=(request.preempt_ok if request.preempt_ok else False),
                            )
                    if manip == ManipType.MANIP_REACH_PREPLACE:
                        ret = self.send_move_request(
                            reach_pose_robot_base_frame,
                            end_effector_link=request.end_effector_link,
                            cartesian_trajectory=request.cartesian_trajectory,
                            planner_profile=(
                                request.planner_profile
                                if request.planner_profile
                                # else "ompl_with_constraints"
                                else "pilz_ptp"
                            ),
                            plan_only=self.plan_first,
                            planning_group=request.planning_group,
                            controller_names=(
                                request.controller_names if request.controller_names else None
                            ),
                            velocity_scaling_factor=(
                                request.velocity_scaling_factor
                                if request.velocity_scaling_factor
                                else None
                            ),
                            acceleration_scaling_factor=(
                                request.acceleration_scaling_factor
                                if request.acceleration_scaling_factor
                                else None
                            ),
                            preempt_ok=(request.preempt_ok if request.preempt_ok else False),
                        )

                    if not self.did_manip_plan_succeed(ret, "Reach", goal_handle):
                        result.state.exec_state = PlanExecState.EXEC_ERROR
                        result.state.exec_message = "Failed to reach"
                        self.get_logger().warn(
                            f"reach pre-place with ori {reach_pose_robot_base_frame.orientation} failed"
                        )
                        self.get_logger().warn(
                            f"reach pre-place with pos {reach_pose_robot_base_frame.position} failed"
                        )

                        break
                    else:
                        continue

            # move actions (cartesian)
            if manip in [
                ManipType.MANIP_MOVE_GRASP,
                ManipType.MANIP_MOVE_POSTGRASP,
                ManipType.MANIP_MOVE_PLACE,
                ManipType.MANIP_MOVE_POSTPLACE,
                ManipType.MANIP_MOVE_GRASP_ADJUST,
                ManipType.MANIP_MOVE_PLACE_ADJUST,
            ]:
                # no offset
                if manip == ManipType.MANIP_MOVE_GRASP:
                    # compute pick pose
                    move_pose_robot_base_frame = self.compute_manip_pose(request.pick.grasp_pose)
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(
                        move_pose_robot_base_frame,
                        self.fixed_frame,
                        "grasp_pose_base_link",
                        is_static=True,
                    )
                    # Add objects to ignore collision when grasping
                    # FIXME(destogl): probably not needed it should be already set!
                    # enable_touch_links = TouchLinks()
                    # enable_touch_links.frame_id = request.object_id
                    # enable_touch_links.touch_links = list(request.pick.grasp_pose.header.frame_id)
                    # enable_touch_links.touch_links = list(set([request.pick.grasp_pose.header.frame_id + request.pick.allowed_touch_objects]))
                    # self.apply_touch_links(enable_touch_links, TouchLinks())
                if manip == ManipType.MANIP_MOVE_PLACE:
                    # compute place pose
                    move_pose_robot_base_frame = self.compute_manip_pose(request.place.place_pose)
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(
                        move_pose_robot_base_frame,
                        self.fixed_frame,
                        "place_pose_base_link",
                        is_static=True,
                    )
                    self.get_logger().warn(
                        f"SETTING PLACE ALLOWED COLLISIONS TO {request.place.allowed_touch_objects}"
                    )
                    # Add objects to ignore collision when placing
                    enable_touch_links = TouchLinks()
                    enable_touch_links.frame_id = request.object_id
                    enable_touch_links.touch_links = list(set([request.place.place_pose.header.frame_id] + request.place.allowed_touch_objects))
                    self.apply_touch_links(enable_touch_links, TouchLinks())
                # with offset
                if manip == ManipType.MANIP_MOVE_POSTGRASP:
                    # compute post-pick pose
                    move_pose_robot_base_frame = self.compute_manip_pose(
                        request.pick.grasp_pose,
                        True,
                        request.pick.post_grasp_retreat.direction,
                        request.pick.post_grasp_retreat.desired_distance,
                    )
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(
                        move_pose_robot_base_frame,
                        self.fixed_frame,
                        "post_grasp_pose_base_link",
                        is_static=True,
                    )
                if manip == ManipType.MANIP_MOVE_POSTPLACE:
                    # compute post-place pose
                    move_pose_robot_base_frame = self.compute_manip_pose(
                        request.place.place_pose,
                        True,
                        request.place.post_place_retreat.direction,
                        request.place.post_place_retreat.desired_distance,
                    )
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(
                        move_pose_robot_base_frame,
                        self.fixed_frame,
                        "post_place_pose_base_link",
                        is_static=True,
                    )
                if manip == ManipType.MANIP_MOVE_GRASP_ADJUST:
                    move_pose_robot_base_frame = self.compute_manip_pose(
                        request.pick.grasp_pose,
                        True,
                        request.brick_grasp_clearance_compensation.direction,
                        request.brick_grasp_clearance_compensation.desired_distance,
                    )
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(
                        move_pose_robot_base_frame,
                        self.fixed_frame,
                        "pre_grip_pose_compensation_link",
                        is_static=True,
                    )
                if manip == ManipType.MANIP_MOVE_PLACE_ADJUST:
                    move_pose_robot_base_frame = self.compute_manip_pose(
                        request.place.place_pose,
                        True,
                        request.brick_grasp_clearance_compensation.direction,
                        request.brick_grasp_clearance_compensation.desired_distance,
                    )
                    if move_pose_robot_base_frame is None:
                        break
                    self.visualization_publisher.publish_pose_as_transform(
                        move_pose_robot_base_frame,
                        self.fixed_frame,
                        "pre_grip_pose_compensation_link",
                        is_static=True,
                    )

                
                if request.add_to_blending_queue:
                    self.move_sequence.append(self.create_move_seq_item(request, move_pose_robot_base_frame))
                else:
                    # perform the action
                    # do the actual planning and execution
                    ret = self.send_move_request(
                        move_pose_robot_base_frame,
                        end_effector_link=request.end_effector_link,
                        cartesian_trajectory=(
                            request.cartesian_trajectory if request.planner_profile else True
                        ),
                        planner_profile=(
                            request.planner_profile if request.planner_profile else "pilz_lin"
                        ),
                        plan_only=self.plan_first,
                        planning_group=request.planning_group,
                        controller_names=(
                            request.controller_names if request.controller_names else None
                        ),
                        velocity_scaling_factor=(
                            request.velocity_scaling_factor
                            if request.velocity_scaling_factor
                            else None
                        ),
                        acceleration_scaling_factor=(
                            request.acceleration_scaling_factor
                            if request.acceleration_scaling_factor
                            else None
                        ),
                        preempt_ok=(request.preempt_ok if request.preempt_ok else False),
                    )

                    if not self.did_manip_plan_succeed(ret, "Move", goal_handle):
                        result.state.exec_state = PlanExecState.EXEC_ERROR
                        result.state.exec_message = "Failed to Move"
                        break
                    else:
                        disable_touch_links = TouchLinks()
                        if manip == ManipType.MANIP_MOVE_POSTGRASP:
                            # Remove allowed collisions after grasping
                            self.get_logger().warn(
                                    f"SETTING POSTGRASP DISALLOWED COLLISIONS TO {request.disable_allowed_touch_objects_after_pick}"
                            )
                            disable_touch_links.frame_id = request.object_id
                            disable_touch_links.touch_links = list(set([request.pick.grasp_pose.header.frame_id] + request.disable_allowed_touch_objects_after_pick))
                            self.apply_touch_links(TouchLinks(), disable_touch_links)
                        self.apply_touch_links(TouchLinks(), disable_touch_links)

                        continue

            # Execute the sequence
            if manip == ManipType.EXECUTE_MOVE_SEQUENCE and len(self.move_sequence) > 0:
                if not self.send_move_sequence_goal(timeout=self.wait_for_action_server_timeout):
                    goal_handle.abort()
                self.move_sequence = []
            
            # Attach/Detach actions
            if manip in [
                ManipType.MANIP_GRASP,
                ManipType.MANIP_RELEASE,
                ManipType.MANIP_GRIPPER_OPEN,
                ManipType.MANIP_GRIPPER_CLOSE,
            ]:
                if request.gripper_cmd_action_name:
                    gripper_cmd_action_name = request.gripper_cmd_action_name
                else:
                    gripper_cmd_action_name = self.default_gripper_cmd_action_name
                if manip == ManipType.MANIP_GRASP or manip == ManipType.MANIP_GRIPPER_CLOSE:
                    if gripper_cmd_action_name:
                        if self.use_parallel_gripper:
                            # First, we handle gripper actions
                            gripper_cmd = ParallelGripperCommand.Goal()
                            gripper_cmd.command.position = [ 0.05 ]
                            self.gripper_clients[gripper_cmd_action_name].wait_for_server()
                            self.gripper_clients[gripper_cmd_action_name].send_goal(gripper_cmd)
                        else:
                            # First, we handle gripper actions
                            gripper_cmd = GripperCommand.Goal()
                            gripper_cmd.command.position = 0.05
                            self.gripper_clients[gripper_cmd_action_name].wait_for_server()
                            self.gripper_clients[gripper_cmd_action_name].send_goal(gripper_cmd)

                    # additionally handle attach
                    if manip == ManipType.MANIP_GRASP:
                        # if success attach
                        if not request.disable_scene_handling:
                            ret = self.attach(
                                request.object_id,
                                request.end_effector_link,
                                list(set(self.allowed_touch_links + request.pick.allowed_touch_objects)),
                            )
                        else:
                            continue
                        if not self.did_manip_plan_succeed(
                            ret, "Gripper attach/detach/adjust", goal_handle
                        ):
                            result.state.exec_state = PlanExecState.EXEC_ERROR
                            result.state.exec_message = "Failed to attach/detach"
                            break
                        else:
                            continue
                if manip == ManipType.MANIP_RELEASE or manip == ManipType.MANIP_GRIPPER_OPEN:
                    if gripper_cmd_action_name:
                        if self.use_parallel_gripper:
                            # First, we handle gripper actions
                            gripper_cmd = ParallelGripperCommand.Goal()
                            gripper_cmd.command.position = [ 0.02 ]
                            self.gripper_clients[gripper_cmd_action_name].wait_for_server()
                            self.gripper_clients[gripper_cmd_action_name].send_goal(gripper_cmd)
                        else:
                            # First, we handle gripper actions
                            gripper_cmd = GripperCommand.Goal()
                            gripper_cmd.command.position = 0.02
                        self.gripper_clients[gripper_cmd_action_name].wait_for_server()
                        self.gripper_clients[gripper_cmd_action_name].send_goal(gripper_cmd)

                    # Additionally handle detach
                    if manip == ManipType.MANIP_RELEASE:
                        # if success detach
                        if not request.disable_scene_handling:
                            # TODO(destogl): do we need here disable allowed collisions?
                            ret = self.detach(request.object_id, request.place.place_pose.header.frame_id)
                        else:
                            continue
                        if not self.did_manip_plan_succeed(
                            ret, "Gripper attach/detach/adjust", goal_handle
                        ):
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
    

    def send_move_sequence_goal(self, timeout: Float64):
        goal = MoveGroupSequence.Goal()
        goal.request.items = self.move_sequence
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        if not self.move_sequence_action_client.wait_for_server(timeout):
            return False

        send_goal_future = self.move_sequence_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        print("type goal_handle: ", type(goal_handle))
        if not goal_handle.accepted:
            self.get_logger().info('Move sequence got rejected by ActionServer')
            return False

        self.get_logger().info('Move sequence accepted by ActionServer)')

        get_result_future: Future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result: ExecuteTrajectory_GetResult_Response = get_result_future.result()
        if not result:
            self.get_logger().error("No result received from the action server.")
            False

        result_result: MoveGroupSequence.Result = result.result
        if result_result.response.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("Move sequence executed successfully.")
            return True
        else:
            self.get_logger().error(
                f"Move sequence failed with error code: {result_result.response.error_code.val}"
            )
            return False
    

    def did_manip_plan_succeed(
        self, plan_exec_success: bool, action_name: str, goal_handle: ServerGoalHandle
    ):
        if plan_exec_success:
            self.manip_feedback.state.plan_state = PlanExecState.PLAN_SUCCESS
            self.manip_feedback.state.exec_state = PlanExecState.EXEC_SUCCESS
            goal_handle.publish_feedback(self.manip_feedback)
            self.get_logger().info(f"{action_name} succeeded.")
            return True
        else:
            self.manip_feedback.state.plan_state = PlanExecState.PLAN_ERROR
            self.manip_feedback.state.exec_state = PlanExecState.EXEC_ERROR
            goal_handle.publish_feedback(self.manip_feedback)
            self.get_logger().info(f"{action_name} failed.")
            goal_handle.abort()
            return False

    def attach(self, id: str, attach_link: str, allowed_touch_links: list[str]):
        req = AttachObject.Request()
        req.id = id
        req.link_name = attach_link
        req.touch_links = allowed_touch_links
        response: AttachObject.Response = self.attach_object_cli.call(req)
        if response.result.state != ServiceResult.SUCCESS:
            self.get_logger().error(f"Attach object {id} has failed.")
            return False
        self.get_logger().info(f"Successfully attached object {id} to {attach_link}.")
        return True

    def detach(self, id: str, detach_to_link: str = None):
        req = DetachObject.Request()
        req.id = id
        req.detach_to_link = detach_to_link
        response: DetachObject.Response = self.detach_object_cli.call(req)
        if response.result.state != ServiceResult.SUCCESS:
            self.get_logger().error(f"Detach object {id} has failed.")
            return False
        self.get_logger().info(f"Successfully detached object {id}.")
        return True

    def apply_touch_links(self, enable_touch_links: TouchLinks, disable_touch_links: TouchLinks):
        # if empty - nothing to do :)
        if not enable_touch_links.touch_links and not disable_touch_links.touch_links:
            return True

        req = ApplyTouchLinks.Request()
        req.enable_touch_links = [enable_touch_links]
        req.disable_touch_links = [disable_touch_links]
        response: ApplyTouchLinks.Response = self.apply_touch_links_cli.call(req)
        if response.result.state != ServiceResult.SUCCESS:
            self.get_logger().error(f"Apply Touch Links has failed: '{response.result.message}'")
            return False
        self.get_logger().info(f"Successfully applied touch links.")
        return True


def main(args=None):
    rclpy.init(args=args)

    executor = MultiThreadedExecutor()

    mc = RobotClient()
    executor.add_node(mc)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
