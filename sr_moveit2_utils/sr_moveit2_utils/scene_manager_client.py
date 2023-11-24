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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sr_manipulation_interfaces.srv import AttachObject, DetachObject
from sr_manipulation_interfaces.msg import ServiceResult


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


class SceneManagerClient(Node):
    def __init__(self):
        """
        Create a new client for managing scene with MoveIt2.
        """
        super().__init__('scene_manager_client')

        # Only a single action on the scene is allowed at a time, so use a MutuallyExclusiveCallbackGroup
        self.server_callback_group = MutuallyExclusiveCallbackGroup()
        # create service clients to Scene Manager for attach and detach
        self.attach_object_cli = self.create_client(AttachObject, "/attach_object")
        while not self.attach_object_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('/attach_object service not available, waiting again...')

        self.detach_object_cli = self.create_client(DetachObject, "/detach_object")
        while not self.detach_object_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('/detach_object service not available, waiting again...')
        self.get_logger().info('Scene Manager Client initialized')
        

    def attach(self, id:str, attach_link:str, allowed_touch_links:list[str]):
        req = AttachObject.Request()
        req.id = id
        req.link_name = attach_link
        req.touch_links = allowed_touch_links
        future = self.attach_object_cli.call_async(req)
        response = wait_for_response(future, self)
        if response.result.state != ServiceResult.SUCCESS:
            self.get_logger().error(f"Attach has failed.")
            return False
        self.get_logger().debug(f"Successfully attached object {id} to {attach_link}.")
        return True
    
    def detach(self, id:str):
        req = DetachObject.Request()
        req.id = id
        future = self.detach_object_cli.call_async(req)
        response = wait_for_response(future, self)
        if response.result.state != ServiceResult.SUCCESS:
            self.get_logger().error(f"Detach has failed.")
            return False
        self.get_logger().debug(f"Successfully detached object {id}.")
        return True

def main(args=None):

    rclpy.init(args=args)

    executor = MultiThreadedExecutor()
    
    sc = SceneManagerClient()

    try:
        rclpy.spin(sc, executor)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()