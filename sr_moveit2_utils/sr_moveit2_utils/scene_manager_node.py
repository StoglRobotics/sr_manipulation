#!/usr/bin/env python
# Copyright (c) 2023, Stogl Robotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author Guillaume Walck

import rclpy
from rclpy.executors import MultiThreadedExecutor
from sr_moveit2_utils.scene_manager import SceneManager

def main(args=None):

    rclpy.init(args=args)

    executor = MultiThreadedExecutor()

    path_to_mesh = "file://" + self.package_share_dir + "/resources/brick_pocket.stl"
    scene_manager = SceneManager("maurob_scene_manager", None, path_to_mesh)

    try:
        rclpy.spin(scene_manager, executor)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()