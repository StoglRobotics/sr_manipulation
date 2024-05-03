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

from dataclasses import dataclass
from typing import Dict
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    OrientationConstraint,
    PositionConstraint,
    RobotTrajectory,
    MotionPlanRequest,
    PlanningOptions,
    MoveItErrorCodes,
)
from moveit_msgs.srv import GetCartesianPath
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive


@dataclass
class PlannerProfile:
    name: str
    planning_pipeline: str
    planner_id: str
    num_planning_attempts: int
    is_cartonly: bool
    requires_cart_interpolation: bool
    use_constraints: bool
    allowed_planning_time: float


class MoveitClient:
    def __init__(
        self,
        node: Node,
        pose_reference_frame: str,
        end_effector_link: str,
    ):
        self.node = node
        self._pose_reference_frame = pose_reference_frame
        self._end_effector_link = end_effector_link

        self._planner_profiles_map: Dict[str, PlannerProfile] = {}

        self.declare_all_parameters()

        self._planning_group = self.node.get_parameter("planning_group").value

        self._valid_constraints = (
            len(self.node.get_parameter("constraints.orientation.absolute_tolerance").value) == 3
        )
        self._default_allowed_planning_time = self.node.get_parameter(
            "default_allowed_planning_time"
        ).value
        if self._default_allowed_planning_time <= 0.0:
            self._default_allowed_planning_time = 5.0

        planner_profiles_names = self.node.get_parameter("planner_profiles.names").value
        if len(planner_profiles_names) > 0:
            planning_pipelines = self.node.get_parameter(
                "planner_profiles.planning_pipelines"
            ).value
            planner_ids = self.node.get_parameter("planner_profiles.planner_ids").value
            use_constraints = self.node.get_parameter("planner_profiles.use_constraints").value
            is_cartonly_planners = self.node.get_parameter(
                "planner_profiles.is_cartonly_planners"
            ).value
            requires_cart_interpolations = self.node.get_parameter(
                "planner_profiles.requires_cart_interpolations"
            ).value
            allowed_planning_times = self.node.get_parameter(
                "planner_profiles.allowed_planning_times"
            ).value
            num_planning_attempts = self.node.get_parameter(
                "planner_profiles.num_planning_attempts"
            ).value

            if (
                len(planning_pipelines) == len(planner_profiles_names)
                and len(planner_ids) == len(planner_profiles_names)
                and len(use_constraints) == len(planner_profiles_names)
                and len(is_cartonly_planners) == len(planner_profiles_names)
                and len(requires_cart_interpolations) == len(planner_profiles_names)
                and len(allowed_planning_times) == len(planner_profiles_names)
                and len(num_planning_attempts) == len(planner_profiles_names)
            ):
                for i in range(len(planner_profiles_names)):
                    profile = PlannerProfile(
                        name=planner_profiles_names[i],
                        planning_pipeline=planning_pipelines[i],
                        planner_id=planner_ids[i],
                        num_planning_attempts=num_planning_attempts[i],
                        is_cartonly=is_cartonly_planners[i],
                        requires_cart_interpolation=requires_cart_interpolations[i],
                        use_constraints=self._valid_constraints and use_constraints[i],
                        allowed_planning_time=(
                            allowed_planning_times[i]
                            if allowed_planning_times[i] > 0.0
                            else self._default_allowed_planning_time
                        ),
                    )
                    self._planner_profiles_map[profile.name] = profile

            # add a default profile in any case
            default_profile = PlannerProfile(
                name="default",
                planning_pipeline="ompl",
                planner_id="",
                num_planning_attempts=1,
                use_constraints=False,
                requires_cart_interpolation=True,
                is_cartonly=False,
                allowed_planning_time=self._default_allowed_planning_time,
            )
            self._planner_profiles_map["default"] = default_profile

        # validate that default profile exists
        default_profile_name = self.node.get_parameter("default_profile_name").value
        if default_profile_name not in self._planner_profiles_map:
            self.node.set_parameters([rclpy.Parameter("default_profile_name", "default")])

        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self._get_cartesian_path_srv_cli = self.node.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
            callback_group=self.service_callback_group,
        )

        self.action_client_callback_group = MutuallyExclusiveCallbackGroup()
        self._move_group_client = ActionClient(
            self.node,
            MoveGroup,
            "/move_action",
            callback_group=self.action_client_callback_group,
        )
        self._execute_trajectory_client = ActionClient(
            self.node,
            ExecuteTrajectory,
            "/execute_trajectory",
            callback_group=self.action_client_callback_group,
        )

        self._trajectory_execution_timeout = self.node.get_parameter(
            "trajectory_execution_timeout"
        ).value

        self.node.get_logger().info("Moveit Client initialized.")

    def declare_all_parameters(self):
        self.node.declare_parameter("planning_group", rclpy.Parameter.Type.STRING)
        self.node.declare_parameter("planner_profiles.names", rclpy.Parameter.Type.STRING_ARRAY)
        self.node.declare_parameter(
            "planner_profiles.planning_pipelines", rclpy.Parameter.Type.STRING_ARRAY
        )
        self.node.declare_parameter("trajectory_execution_timeout", rclpy.Parameter.Type.DOUBLE)
        self.node.declare_parameter("default_allowed_planning_time", rclpy.Parameter.Type.DOUBLE)
        self.node.declare_parameter("default_profile_name", rclpy.Parameter.Type.STRING)
        self.node.declare_parameter(
            "planner_profiles.planner_ids", rclpy.Parameter.Type.STRING_ARRAY
        )
        self.node.declare_parameter(
            "planner_profiles.num_planning_attempts", rclpy.Parameter.Type.INTEGER_ARRAY
        )
        self.node.declare_parameter(
            "planner_profiles.is_cartonly_planners", rclpy.Parameter.Type.BOOL_ARRAY
        )
        self.node.declare_parameter(
            "planner_profiles.requires_cart_interpolations", rclpy.Parameter.Type.BOOL_ARRAY
        )
        self.node.declare_parameter(
            "planner_profiles.use_constraints", rclpy.Parameter.Type.BOOL_ARRAY
        )
        self.node.declare_parameter(
            "planner_profiles.allowed_planning_times", rclpy.Parameter.Type.DOUBLE_ARRAY
        )
        self.node.declare_parameter(
            "constraints.orientation.frame_id", rclpy.Parameter.Type.STRING
        )
        self.node.declare_parameter(
            "constraints.orientation.link_name", rclpy.Parameter.Type.STRING
        )
        self.node.declare_parameter(
            "constraints.orientation.orientation", rclpy.Parameter.Type.DOUBLE_ARRAY
        )
        self.node.declare_parameter(
            "constraints.orientation.absolute_tolerance",
            rclpy.Parameter.Type.DOUBLE_ARRAY,
        )
        self.node.declare_parameter("constraints.orientation.weight", rclpy.Parameter.Type.DOUBLE)
        self.node.declare_parameter("constraints.box.frame_id", rclpy.Parameter.Type.STRING)
        self.node.declare_parameter("constraints.box.link_name", rclpy.Parameter.Type.STRING)
        self.node.declare_parameter(
            "constraints.box.dimensions", rclpy.Parameter.Type.DOUBLE_ARRAY
        )
        self.node.declare_parameter("constraints.box.position", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.node.declare_parameter(
            "constraints.box.orientation", rclpy.Parameter.Type.DOUBLE_ARRAY
        )
        self.node.declare_parameter("constraints.box.weight", rclpy.Parameter.Type.DOUBLE)

    def initialize_path_constraints(self):
        orientation_weight = self.node.get_parameter("constraints.orientation.weight").value
        constraints = Constraints()
        constraints.name = "use_equality_constraints"
        if orientation_weight > 1e-3:
            orientation_constraint = OrientationConstraint()
            ori_frame_id = self.node.get_parameter("constraints.orientation.frame_id").value
            if ori_frame_id:
                orientation_constraint.header.frame_id = ori_frame_id
            else:
                orientation_constraint.header.frame_id = self._pose_reference_frame
            ori_link_name = self.node.get_parameter("constraints.orientation.link_name").value
            if ori_link_name:
                orientation_constraint.link_name = ori_link_name
            else:
                orientation_constraint.link_name = self._end_effector_link
            orientation = self.node.get_parameter("constraints.orientation.orientation").value
            orientation_constraint.orientation.x = orientation[0]
            orientation_constraint.orientation.y = orientation[1]
            orientation_constraint.orientation.z = orientation[2]
            orientation_constraint.orientation.w = orientation[3]
            tolerance = self.node.get_parameter("constraints.orientation.absolute_tolerance").value
            orientation_constraint.absolute_x_axis_tolerance = tolerance[0]
            orientation_constraint.absolute_y_axis_tolerance = tolerance[1]
            orientation_constraint.absolute_z_axis_tolerance = tolerance[2]

            orientation_constraint.weight = orientation_weight
            constraints.orientation_constraints.append(orientation_constraint)

        position_weight = self.node.get_parameter("constraints.box.weight").value
        if position_weight > 1e-3:
            position_constraint = PositionConstraint()
            pos_frame_id = self.node.get_parameter("constraints.box.frame_id").value
            if pos_frame_id:
                position_constraint.header.frame_id = pos_frame_id
            else:
                position_constraint.header.frame_id = self._pose_reference_frame
            pos_link_name = self.node.get_parameter("constraints.box.link_name").value
            if pos_link_name:
                position_constraint.link_name = pos_link_name
            else:
                position_constraint.link_name = self._end_effector_link

            box_bounding_volume = SolidPrimitive()
            box_bounding_volume.type = SolidPrimitive.BOX
            box_dimensions = self.node.get_parameter("constraints.box.dimensions").value
            box_bounding_volume.dimensions.append(box_dimensions[0])
            box_bounding_volume.dimensions.append(box_dimensions[1])
            box_bounding_volume.dimensions.append(box_dimensions[2])
            position_constraint.constraint_region.primitives.append(box_bounding_volume)

            box_pose = Pose()
            box_position = self.node.get_parameter("constraints.box.position").value
            box_pose.position.x = box_position[0]
            box_pose.position.y = box_position[1]
            box_pose.position.z = box_position[2]
            box_orientation = self.node.get_parameter("constraints.box.orientation").value
            box_pose.orientation.x = box_orientation[0]
            box_pose.orientation.y = box_orientation[1]
            box_pose.orientation.z = box_orientation[2]
            box_pose.orientation.w = box_orientation[3]
            position_constraint.constraint_region.primitive_poses.append(box_pose)
            position_constraint.weight = position_weight
            constraints.position_constraints.append(position_constraint)
        return constraints

    def initialize_goal_constraints(
        self, pose: Pose, pos_constraint_weight=1.0, ori_constraint_weight=1.0, ori_tolerance=1e-4
    ):
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self._pose_reference_frame
        position_constraint.link_name = self._end_effector_link
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        bounding_volume = SolidPrimitive()
        bounding_volume.type = SolidPrimitive.SPHERE
        bounding_volume.dimensions.append(1e-4)
        position_constraint.constraint_region.primitives.append(bounding_volume)

        position_constraint.constraint_region.primitive_poses.append(pose)
        position_constraint.weight = pos_constraint_weight

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self._pose_reference_frame
        orientation_constraint.link_name = self._end_effector_link
        orientation_constraint.orientation = pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = ori_tolerance
        orientation_constraint.absolute_y_axis_tolerance = ori_tolerance
        orientation_constraint.absolute_z_axis_tolerance = ori_tolerance
        orientation_constraint.weight = ori_constraint_weight

        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        return constraints

    def execute(self, plan: RobotTrajectory) -> bool:
        self.node.get_logger().info("Executing planned trajectory")
        self._execute_trajectory_client.wait_for_server()
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = plan
        exec_result: ExecuteTrajectory.Result = self._execute_trajectory_client.send_goal(
            goal
        ).result
        if exec_result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.node.get_logger().info("Trajectory execution succeeded.")
            return True
        else:
            self.node.get_logger().error(
                f"Trajectory execution failed with error code: {exec_result.error_code}"
            )
            return False

    def plan(
        self,
        pose: Pose,
        cartesian_trajectory: bool = False,
        planner_profile: str = "default",
        velocity_scaling_factor: float = 1.0,
        acceleration_scaling_factor: float = 1.0,
        allowed_planning_time: float = 5.0,
        get_cart_path_req_avoid_collisions: bool = True,
    ) -> RobotTrajectory:
        if not allowed_planning_time or allowed_planning_time <= 0.0:
            allowed_planning_time = self._default_allowed_planning_time

        self.node.get_logger().info(
            "inside plan(): "
            f"\npose={pose} "
            f"\ncartesian_trajectory={cartesian_trajectory} "
            f"\nplanner_profile={planner_profile} "
            f"\nvelocity_scaling_factor={velocity_scaling_factor} "
            f"\nmax_acceleration_scaling_factor={acceleration_scaling_factor} "
            f"\nallowed_planning_time={allowed_planning_time} "
            f"\nget_cart_path_req_avoid_collisions={get_cart_path_req_avoid_collisions} "
        )
        self._move_group_client.wait_for_server()
        goal = MoveGroup.Goal()

        profile = self._planner_profiles_map.get(planner_profile, None)
        if not profile:
            default_profile_name = self.node.get_parameter("default_profile_name").value
            profile = self._planner_profiles_map[default_profile_name]

        goal.request = MotionPlanRequest()
        goal.request.group_name = self._planning_group
        goal.request.pipeline_id = profile.planning_pipeline
        goal.request.planner_id = profile.planner_id
        goal.request.num_planning_attempts = profile.num_planning_attempts

        if profile.use_constraints:
            goal.request.path_constraints = self.initialize_path_constraints()

        goal.request.allowed_planning_time = profile.allowed_planning_time
        goal.request.max_velocity_scaling_factor = velocity_scaling_factor
        goal.request.max_acceleration_scaling_factor = acceleration_scaling_factor

        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        error_msg = (
            "Path planning failed for "
            f"group_name={goal.request.group_name} "
            f"planning_pipeline={goal.request.pipeline_id} "
            f"planner_id={goal.request.planner_id} "
            "with error code: {error_code}"
        )

        if not cartesian_trajectory:
            if profile.is_cartonly:
                raise ValueError(
                    f"The chosen planner ({profile.name}) does not support "
                    "non-cartesian trajectory planning."
                )
            self.node.get_logger().info("Using non-cartesian path planning.")
            goal.request.goal_constraints.append(self.initialize_goal_constraints(pose))
            action_result: MoveGroup.Result = self._move_group_client.send_goal(goal).result
            if action_result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.node.get_logger().info("Non-cartesian path planning succeeded.")
                return action_result.planned_trajectory
            else:
                self.node.get_logger().error(error_msg.format(error_code=action_result.error_code))
                return None
        else:
            self.node.get_logger().info("Using cartesian path planning.")
            if profile.requires_cart_interpolation:
                self.node.get_logger().info("Using cartesian interpolation.")
                get_cart_path_req = GetCartesianPath.Request()
                get_cart_path_req.start_state.is_diff = True
                get_cart_path_req.group_name = self._planning_group
                get_cart_path_req.header.frame_id = self._pose_reference_frame
                get_cart_path_req.header.stamp = self.node.get_clock().now().to_msg()
                get_cart_path_req.waypoints = [pose]
                get_cart_path_req.max_step = 0.001
                get_cart_path_req.jump_threshold = 0.0
                if profile.use_constraints:
                    get_cart_path_req.path_constraints = goal.request.path_constraints
                get_cart_path_req.avoid_collisions = get_cart_path_req_avoid_collisions
                get_cart_path_req.link_name = self._end_effector_link

                # sync call
                response: GetCartesianPath.Response = self._get_cartesian_path_srv_cli.call(
                    get_cart_path_req
                )
                if response.error_code.val == MoveItErrorCodes.SUCCESS:
                    self.node.get_logger().info("Cartesian path planning succeeded.")
                    return response.solution
                else:
                    self.node.get_logger().error(error_msg.format(error_code=response.error_code))
                    return None
            else:
                goal.request.goal_constraints.append(self.initialize_goal_constraints(pose))
                action_result: MoveGroup.Result = self._move_group_client.send_goal(goal).result
                if action_result.error_code.val == MoveItErrorCodes.SUCCESS:
                    self.node.get_logger().info("Cartesian path planning succeeded.")
                    return action_result.planned_trajectory
                else:
                    self.node.get_logger().error(
                        error_msg.format(error_code=action_result.error_code)
                    )
                    return None
