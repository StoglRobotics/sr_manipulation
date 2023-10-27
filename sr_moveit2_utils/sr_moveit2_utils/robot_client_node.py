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
#
# Authors: Dr. Denis Stogl, Guillaume Walck
#

from copy import deepcopy
import time
import numpy as np
import rclpy
import importlib
from rclpy.executors import MultiThreadedExecutor
import threading
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle, GoalStatus

from sr_manipulation_interfaces.action import PlanMoveTo, Manip
from sr_manipulation_interfaces.msg import ManipType, PlanExecState, ServiceResult
from sr_manipulation_interfaces.srv import AttachObject, DetachObject

from geometry_msgs.msg import PoseStamped, Vector3, Vector3Stamped, Pose
from moveit_msgs.msg import Grasp
import moveit_msgs.msg

# from sr_moveit2_utils.scene_manager_client import SceneManagerClient
from sr_ros2_python_utils.visualization_publishers import VisualizatonPublisher


class RobotClient(Node):
    def __init__(self):
        super().__init__("robot_client")
        # Declare parameters
        self.declare_parameter('robot_modules', rclpy.Parameter.Type.STRING_ARRAY) 
        self.module_list = self.get_parameter("robot_modules").value
        # # Load in the clients
        self.client_container = []
        for module_name in self.module_list:
            module_split = module_name.split(".")
            module_file = importlib.import_module("sr_moveit2_utils.robot_client_modules."+module_split[0])
            module = getattr(module_file, module_split[1])
            self.client_container.append(module())        
        # Create all services based on the clients
        
    def add_modules_to_executor(self, executor):
        for client in self.client_container:
            executor.add_node(client)


def main(args=None):

    rclpy.init(args=args)

    executor = MultiThreadedExecutor()
    
    rc = RobotClient()
    executor.add_node(rc)
    rc.add_modules_to_executor(executor)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = rc.create_rate(10)
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    executor_thread.join()
