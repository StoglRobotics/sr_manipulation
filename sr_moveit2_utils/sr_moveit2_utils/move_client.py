import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sr_manipulation_interfaces.srv import MoveToPose, MoveToPoseSeq
from std_srvs.srv import Trigger


def wait_for_response(future, client):
    while rclpy.ok():
        rclpy.spin_once(client)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                client.get_logger().info(
                    "Service call to move_to_pose or move_to_pose_seq failed %r" % (e,)
                )
                return None
            else:
                return response


class MoveClient(Node):
    """
    Helper class to create services to the MoveItWrapper
    This class is not a standalone class and requires a node to be instantiated with correct callback groups
    """

    def __init__(self, default_velocity_scaling_factor=0.5):
        super().__init__("move_client")
        self.default_velocity_scaling_factor = default_velocity_scaling_factor

        self.move_cli = self.create_client(
            MoveToPose, "/move_to_pose"
        )  # , callback_group=svc_cbg)
        while not self.move_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                "move_to_pose service not available, waiting again..."
            )

        self.move_seq_cli = self.create_client(
            MoveToPoseSeq, "/move_to_pose_seq"
        )  # , callback_group=svc_cbg)
        if not self.move_seq_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().info(
                "move_to_pose_seq service not available, waiting again..."
            )
            self.move_seq_cli = None

        self.stop_trajectory_exec_cli = self.create_client(
            Trigger, "/stop_trajectory_execution"
        )  # ,
        # callback_group=svc_cbg)

        while not self.stop_trajectory_exec_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info(
                "stop_trajectory_execution service not available, waiting again..."
            )

    def send_move_request(
        self,
        pose: Pose,
        cartesian_trajectory: bool = True,
        planner_profile: str = "",
        velocity_scaling_factor=None,
        allowed_planning_time: float = 0.0,
        plan_only: bool = True,
    ):
        # use default velocity scaling if not defined
        if not velocity_scaling_factor:
            velocity_scaling_factor = self.default_velocity_scaling_factor

        self.req_move = MoveToPose.Request()
        self.req_move.pose = pose
        self.req_move.cart = cartesian_trajectory
        self.req_move.planner_profile = planner_profile
        self.req_move.velocity_scaling_factor = velocity_scaling_factor
        self.req_move.allowed_planning_time = allowed_planning_time
        self.req_move.only_plan = plan_only
        future = self.move_cli.call_async(self.req_move)
        response = wait_for_response(future, self)
        if not response.success:
            self.get_logger().error(f"Move to the pose {pose} has failed.")
            return False
        self.get_logger().debug(f"Successfully move to the {pose}.")
        return True

    def send_move_seq_request(
        self,
        poses: list[Pose],
        carts: list[bool] = [False],
        velocity_scaling_factors: list[float] = [0.1],
        blending_radii: list[float] = [0.05],
        profiles: list[str] = [""],
        allowed_planning_time: float = 0.0,
    ):
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

        self.get_logger().info(
            f"Received request to move to pose seq with a list of {len(poses)} poses."
        )
        future = self.move_seq_cli.call_async(req)
        response = wait_for_response(future, self)
        if not response.success:
            self.get_logger().error(f"Move to {len(poses)} poses has failed.")
            return False
        else:
            self.get_logger().debug(f"Successfully moved to {len(poses)} poses.")
        return True

    def send_stop_request(self):
        self.req_stop = Trigger.Request()
        response = self.stop_trajectory_exec_cli.call(self.req_stop)
        future = self.stop_trajectory_exec_cli.call_async(self.req_stop)
        response = wait_for_response(future, self)
        if not response.success:
            self.get_logger().fatal("Stopping the robot has failed!")
            return False
        self.get_logger().debug("Robot is successfully stopped.")
        return True
