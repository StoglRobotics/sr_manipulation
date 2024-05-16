# Software License Agreement (BSD License)
# Copyright (c) 2012, Willow Garage, Inc.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of Willow Garage nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
# Author: Ioan Sucan, Sachin Chitta */
# based on planning_scene_interface.cpp
#
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

from rclpy.node import Node
from rclpy.time import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import numpy
from sr_manipulation_interfaces.srv import (
    AddObjects,
    RemoveObjects,
    AttachObject,
    DetachObject,
    GetObjectPose,
)
from sr_manipulation_interfaces.msg import ObjectDescriptor, ServiceResult
from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, Point

# from shape_msgs.msg import SolidPrimitive
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle

from std_srvs.srv import Trigger

# from sr_ros2_python_utils.transforms import TCPTransforms


# part of make_mesh copied from planning_scene_interface.py of moveit_commander
# original authors: Ioan Sucan, Felix Messmer, same License as above
try:
    import pyassimp
    from pyassimp import load as pyassimp_load
    from importlib.metadata import version as libversion
except (ImportError, ModuleNotFoundError) as error:
    pyassimp = False
    print(f"Failed to import pyassimp: {error}. Continuing without it ...")


class SceneManager(Node):
    def __init__(self, name, default_object_mesh_path="", scene_base_frame="world"):
        """
        Create a new client for managing scene with MoveIt2.

        :param name: Node name
        :param parent_node: Parent node object to access correct transformations
        :param default_object_mesh_path: default path to objects' meshes
        :param scene_base_frame: Base frame of the scene (default: "world")
        """
        super().__init__(name)

        # Check if correct version of pyassimp is used. There version have to be different then 4.1.4
        if pyassimp is False:
            self.get_logger().fatal("pyassimp library not found!")
            return None
        else:
            if libversion("pyassimp") == "4.1.4":
                self.get_logger().fatal(
                    "Pyassimp version '4.1.4' has a bug - please install a newer version, e.g., using 'pip3 install -U pyassimp'!"
                )
                return None

        # storage to track the objects
        self.object_in_the_scene_storage = {}  # dict[str, CollisionObject]
        self.attached_object_store = {}  # dict[str, AttachedCollisionObject]
        # objects not in the scene (not collision object)
        self.object_in_marker_storage = {}
        self.num_markers_counter = 0

        self.default_object_mesh_path = default_object_mesh_path
        self.scene_base_frame = scene_base_frame

        # self.tcp_transforms = TCPTransforms(parent_node)

        # Only a single action on the scene is allowed at a time, so use a MutuallyExclusiveCallbackGroup
        self.server_callback_group = MutuallyExclusiveCallbackGroup()
        # create services
        self.attach_srv = self.create_service(
            AttachObject,
            "attach_object",
            self.attach_object_cb,
            callback_group=self.server_callback_group,
        )
        self.detach_srv = self.create_service(
            DetachObject,
            "detach_object",
            self.detach_object_cb,
            callback_group=self.server_callback_group,
        )
        self.add_object_srv = self.create_service(
            AddObjects,
            "add_objects",
            self.add_objects_cb,
            callback_group=self.server_callback_group,
        )
        self.remove_object_srv = self.create_service(
            RemoveObjects,
            "remove_objects",
            self.remove_objects_cb,
            callback_group=self.server_callback_group,
        )
        self.get_pose_srv = self.create_service(
            GetObjectPose,
            "get_object_pose",
            self.get_object_pose_cb,
            callback_group=self.server_callback_group,
        )
        
        self.clear_scene_srv = self.create_service(
            Trigger,
            "clear_scene",
            self.clear_scene_cb,
            callback_group=self.server_callback_group,
        )

        self.outgoing_callback_group = MutuallyExclusiveCallbackGroup()
        # Use a separate group for the clients or publisher called from within the servers. It can be Reentrant, especially to let more messages going out

        # create publishers to MoveIt2/RViz
        self.planning_scene_diff_publisher = self.create_publisher(
            PlanningScene, "/planning_scene", 1, callback_group=self.outgoing_callback_group
        )
        self.markerarray_publisher = self.create_publisher(
            MarkerArray, "/scene_manager_markers", 1, callback_group=self.outgoing_callback_group
        )
        # self.collision_object_publisher = self.create_publisher(CollisionObject, "collision_object", 1)

        # create service clients to MoveIt2
        self.planning_scene_diff_cli = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
            callback_group=self.outgoing_callback_group,
        )
        while not self.planning_scene_diff_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().info("apply_planning_scene service not available, waiting again...")

        self.get_logger().info("Scene Manager initialized")

    # make_mesh copied from planning_scene_interface.py of moveit_commander
    # original authors: Ioan Sucan, Felix Messmer, same License as above
    def make_mesh(self, uri: str, scale=(1, 1, 1)):
        try:
            # TODO(gwalck) handle uri in a cleaner way, maybe even with packages
            if uri.startswith("file://"):
                filename = uri[7 : len(uri)]  # noqa: E203 - whitespace before ':'. Better readable
            else:
                filename = uri
            # with pyassimp_load('/home/guillaumew/workspaces/kh_kuka/install/moveit_wrapper/share/moveit_wrapper/resources/brick_pocket.stl') as scene:
            with pyassimp_load(filename) as scene:
                if not scene.meshes or len(scene.meshes) == 0:
                    self.get_logger().warn("There are no meshes in the file")
                    return None
                if len(scene.meshes[0].faces) == 0:
                    self.get_logger().warn("There are no faces in the mesh")
                    return None

                mesh = Mesh()
                first_face = scene.meshes[0].faces[0]
                if hasattr(first_face, "__len__"):
                    for face in scene.meshes[0].faces:
                        if len(face) == 3:
                            triangle = MeshTriangle()
                            triangle.vertex_indices = numpy.array(
                                [face[0], face[1], face[2]], dtype=numpy.uint32
                            )
                            mesh.triangles.append(triangle)
                elif hasattr(first_face, "indices"):
                    for face in scene.meshes[0].faces:
                        if len(face.indices) == 3:
                            triangle = MeshTriangle()
                            triangle.vertex_indices = numpy.array(
                                [face.indices[0], face.indices[1], face.indices[2]],
                                dtype=numpy.uint32,
                            )
                            mesh.triangles.append(triangle)
                else:
                    self.get_logger().warn(
                        "Unable to build triangles from mesh due to mesh object structure"
                    )
                    return None
                for vertex in scene.meshes[0].vertices:
                    point = Point()
                    point.x = vertex[0] * scale[0]
                    point.y = vertex[1] * scale[1]
                    point.z = vertex[2] * scale[2]
                    mesh.vertices.append(point)
        except Exception as e:
            self.get_logger().error(f"Failed to load mesh file {filename}, {e}")
            return None
        return mesh

    def collision_from_object_descriptor(self, obj: ObjectDescriptor) -> CollisionObject:
        col_object = CollisionObject()
        col_object.id = obj.id
        col_object.header.frame_id = obj.pose.header.frame_id
        col_object.pose = obj.pose.pose
        # use given offset
        pose = obj.shape_pose
        if not obj.paths_to_mesh:
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [obj.bbox_size.x, obj.bbox_size.y, obj.bbox_size.z]
            col_object.primitives.append(primitive)
            col_object.primitive_poses.append(pose)
        else:
            for mesh_filename in obj.paths_to_mesh:
                mesh = self.make_mesh(mesh_filename)
                if mesh is not None:
                    col_object.meshes.append(mesh)
                    col_object.mesh_poses.append(pose)
        return col_object

    def get_object_pose_cb(
        self, request: GetObjectPose.Request, response: GetObjectPose.Response
    ) -> GetObjectPose.Response:
        # check if the object exists at all
        self.get_logger().debug("Scene Manager get object pose")
        if request.id not in self.object_in_the_scene_storage:
            response.result.state = ServiceResult.NOTFOUND
        else:
            pose = self.get_object_stamped_pose(request.id)
            if pose is not None:
                response.result.state = ServiceResult.SUCCESS
                response.result.pose = pose
            else:
                response.result.state = ServiceResult.FAILED
        return response

    def get_object_stamped_pose(self, object_id: str):
        if object_id not in self.object_in_the_scene_storage.keys():
            self.get_logger().error(
                f"Object '{object_id}' is not known to the scene manager. Did you add it?"
            )
            return None

        object = self.object_in_the_scene_storage[object_id]

        pose = PoseStamped()
        pose.header = deepcopy(object.header)
        pose.pose = deepcopy(object.pose)

        return pose

    def detach_object_cb(
        self, request: DetachObject.Request, response: DetachObject.Response
    ) -> DetachObject.Response:
        # check if the object exists at all
        if request.id not in self.attached_object_store:
            response.result.state = ServiceResult.NOTFOUND
        else:
            ret = self.detach_object(request.id)
            if ret:
                response.result.state = ServiceResult.SUCCESS
            else:
                response.result.state = ServiceResult.FAILED
        return response

    def detach_object(self, object_id: str, detach_to_link=None):
        # object must be removed from the robot (store type is AttachedCollisionObject)
        attached_collision_object_to_detach = self.attached_object_store.pop(object_id, None)

        # frame1 = object_to_detach.object.header.frame_id
        frame2 = attached_collision_object_to_detach.link_name

        # relativePoseStamped = self.get_relative_pose_stamped(frame1, frame2)
        # if(object_to_detach.primitive_poses):
        #    object_to_detach.primitive_poses.append(relativePoseStamped.pose)
        # else:
        #    object_to_detach.primitive_poses[0] = relativePoseStamped.pose
        self.get_logger().debug(f"Detaching the object {object_id} from given frame {frame2}.")
        attached_collision_object_to_detach.object.operation = CollisionObject.REMOVE

        ret = self.detach_collision_object(attached_collision_object_to_detach, detach_to_link)

        if not ret:
            self.get_logger().error(f"Detaching object {object_id} has failed!")
            # re-add the object to the scene storage
            self.attached_object_store[object_id] = attached_collision_object_to_detach store
            return False
        self.get_logger().debug(f"Object {object_id} is successfully detached.")
        return True

    def detach_collision_object(
        self, attached_collision_object: AttachedCollisionObject, detach_to_link: str
    ):
        # object_pose_in_attach_link_name = self.tcp_transforms.to_from_tcp_pose_conversion(object.pose, object.header.frame_id, link_name, False)
        detached_collision_object = attached_collision_object

        # Add object again to the scene
        if detach_to_link is None:
            detach_to_link = self.scene_base_frame

        # get the collision (including its mesh) from the attached object
        collision_object_to_add = deepcopy(attached_collision_object.object)
        collision_object_to_add.header.frame_id = (
            detach_to_link  # TODO(gwalck) unsure about that but is in the tuto
        )
        collision_object_to_add.operation = CollisionObject.ADD

        # object_pose_in_detach_to_link_name = self.tcp_transforms.to_from_tcp_pose_conversion(collision_object_to_add.pose, collision_object_to_add.header.frame_id, detach_to_link, False)
        # collision_object_to_add.mesh_poses = [object_pose_in_detach_to_link_name]

        detached_collision_object.object.operation = CollisionObject.REMOVE
        # detached_collision_object.object.mesh_poses = [object_pose_in_attach_link_name]

        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        # add the object to the scene
        # currently not needed, the object is moved back to the scene automatically (https://github.com/ros-planning/moveit2/issues/1069)
        # planning_scene.world.collision_objects.append(collision_object_to_add)
        # detach from to the robot
        planning_scene.robot_state.attached_collision_objects.append(detached_collision_object)
        planning_scene.robot_state.is_diff = True
        ret = self.apply_planning_scene(planning_scene)
        if ret:
            # store the object back to the scene
            self.attached_object_store[collision_object_to_add.id] = collision_object_to_add
        return ret

    def attach_object_cb(
        self, request: AttachObject.Request, response: AttachObject.Response
    ) -> AttachObject.Response:
        if request.id not in self.object_in_the_scene_storage:
            response.result.state = ServiceResult.NOTFOUND
            self.get_logger().warn("Scene Manager Object Not found")
        else:
            ret = self.attach_object(request.id, request.link_name, request.touch_links)
            if ret:
                self.get_logger().debug("Scene Manager Object attached")
                response.result.state = ServiceResult.SUCCESS
            else:
                self.get_logger().warn("Scene Manager Object attach failed")
                response.result.state = ServiceResult.FAILED
        return response

    # for more detailed info on attach, detach operations and planning scene please look at the link below
    # https://moveit.picknik.ai/foxy/doc/planning_scene_ros_api/planning_scene_ros_api_tutorial.html
    def attach_object(self, object_id: str, link_name: str, touch_links: list[str]):
        # object must be removed from the world
        object_to_attach = self.object_in_the_scene_storage.pop(object_id, None)

        self.get_logger().debug(
            f"Initial pose of {object_to_attach.id} in {object_to_attach.header.frame_id} is {object_to_attach.pose}"
        )
        self.get_logger().debug(f"Trying to attach it to {link_name}")

        collision_object_to_remove = CollisionObject()
        collision_object_to_remove.id = object_to_attach.id
        collision_object_to_remove.header.frame_id = (
            self.scene_base_frame
        )  # TODO(gwalck) unsure about that but is in the tuto
        collision_object_to_remove.operation = CollisionObject.REMOVE

        # TODO(gwalck) cleanup the duplicate objects coming form the fusion of 2 functions
        # object_pose_in_attach_link_name = self.tcp_transforms.to_from_tcp_pose_conversion(object.pose, object.header.frame_id, link_name, False)
        # self.get_logger().debug(f"Calculated target pose of {object_to_attach.id} in {link_name} is {object_pose_in_attach_link_name}")

        attached_collision_object = AttachedCollisionObject()
        attached_collision_object.object = object_to_attach
        attached_collision_object.link_name = link_name
        attached_collision_object.touch_links = touch_links
        # attached_collision_object.object.mesh_poses = [object_pose_in_attach_link_name]

        planning_scene = PlanningScene()
        # NOT IN THE TUTO
        planning_scene.is_diff = True
        # remove the object from the scene SHOWS A WARNING, DISABLING
        #planning_scene.world.collision_objects.append(collision_object_to_remove)
        # attach it to the robot
        planning_scene.robot_state.attached_collision_objects.append(attached_collision_object)
        planning_scene.robot_state.is_diff = True
        ret = self.apply_planning_scene(planning_scene)
        if ret:
            # add the attached object to the store
            self.attached_object_store[object_id] = attached_collision_object
            self.get_logger().debug(
                f"Object {object_id} is successfully attached to the {link_name} link."
            )
        else:
            self.get_logger().warn(
                f"Attaching object {object_id} to the link {link_name} has failed!"
            )
            # add the object back to storage
            self.object_in_the_scene_storage[object_id] = object_to_attach
        return ret

    def apply_planning_scene(self, planning_scene: PlanningScene):
        ps_req = ApplyPlanningScene.Request()
        ps_req.scene = planning_scene
        # the call can be synchronous as it  lives in its own cbg
        response = self.planning_scene_diff_cli.call(ps_req)
        if not response.success:
            return False
        return True

    def add_objects_cb(
        self, request: AddObjects.Request, response: AddObjects.Response
    ) -> AddObjects.Response:
        self.get_logger().debug("Adding the objects into the world at the given location.")
        added_object_ids = self.add_objects(request.objects, request.as_marker)
        if added_object_ids:
            if len(request.objects) == len(added_object_ids):
                response.result.state = ServiceResult.SUCCESS
            else:
                response.result.state = ServiceResult.PARTIAL

            response.result.message = f"Added {len(added_object_ids)} objects"
            response.added_object_ids = added_object_ids
        else:
            response.result.state = ServiceResult.FAILED
            response.result.message = "No objects added"

        return response

    def add_objects(self, objects: ObjectDescriptor, as_markers=False) -> list[int]:
        objects_to_add = []
        added_object_ids = []
        object_mesh_paths = []  # needed for markers to get the full resource

        for obj in objects:
            object_to_add = self.collision_from_object_descriptor(obj)

            object_to_add.operation = CollisionObject.ADD

            if as_markers:
                self.object_in_marker_storage[object_to_add.id] = deepcopy(object_to_add)
                if len(obj.paths_to_mesh):
                    object_mesh_paths.append(obj.paths_to_mesh[0])
                else:
                    object_mesh_paths.append("")
            else:
                self.object_in_the_scene_storage[object_to_add.id] = deepcopy(object_to_add)
            objects_to_add.append(deepcopy(object_to_add))
            added_object_ids.append(obj.id)

        if as_markers:
            self.publish_as_marker(objects_to_add, object_mesh_paths)
        else:
            self.publish_planning_scene(objects_to_add)
            # TODO(gwalck) check if objects were added
        return added_object_ids

    def remove_objects_cb(
        self, request: RemoveObjects.Request, response: RemoveObjects.Response
    ) -> RemoveObjects.Response:
        self.get_logger().debug("Removing the objects from the world.")
        removed_object_ids = self.remove_objects(request.ids)
        if removed_object_ids:
            if len(request.ids) == len(removed_object_ids):
                response.result.state = ServiceResult.SUCCESS
            else:
                response.result.state = ServiceResult.PARTIAL

            response.result.message = f"Removed {len(removed_object_ids)} objects"
        else:
            response.result.state = ServiceResult.FAILED
            response.result.message = "No objects removed"

        return response

    def remove_objects(self, object_ids: list[str]) -> list[int]:
        marker_objects_to_remove = []
        scene_objects_to_remove = []
        attached_objects_to_remove = []
        
        removed_maker_object_ids = []
        removed_scene_object_ids = []

        for id in object_ids:
            object_to_remove = CollisionObject()
            object_to_remove.id = id
            object_to_remove.operation = CollisionObject.REMOVE
            if object_to_remove.id in self.object_in_marker_storage:
                marker_objects_to_remove.append(deepcopy(object_to_remove))
                removed_maker_object_ids.append(object_to_remove.id)
                self.object_in_marker_storage.pop(object_to_remove.id, None)
            if object_to_remove.id in self.object_in_the_scene_storage:
                scene_objects_to_remove.append(deepcopy(object_to_remove))
                removed_scene_object_ids.append(object_to_remove.id)
                self.object_in_the_scene_storage.pop(object_to_remove.id, None)
            if object_to_remove.id in self.attached_object_store:
                scene_objects_to_remove.append(deepcopy(object_to_remove))
                removed_scene_object_ids.append(object_to_remove.id)
                self.attached_object_store.pop(object_to_remove.id, None)

        if len(removed_maker_object_ids):
            self.publish_as_marker(marker_objects_to_remove)
            # TODO(gwalck) check if objects were removed
        if len(removed_scene_object_ids):
            self.publish_planning_scene(scene_objects_to_remove)
            # TODO(gwalck) check if objects were removed
        removed_object_ids = removed_maker_object_ids + removed_scene_object_ids
        return removed_object_ids

    def clear_scene_cb(self, request: Trigger.Request, response: Trigger.Response):
        self.get_logger().debug("Clearing the scene.")
        ret = self.clear_scene()
        response.success = True
        response.message = "Scene cleared"
            
        return response
        

    def clear_scene(self):
        objects_to_remove = []
        objects_to_remove.extend([obj.id for obj in self.object_in_the_scene_storage.values()])
        objects_to_remove.extend([obj.id for obj in self.object_in_marker_storage.values()])
        objects_to_remove.extend([obj.id for obj in self.attached_object_store.values()])
        
        self.remove_all_markers()
        return self.remove_objects(objects_to_remove)
        
        
    # end def

    def publish_planning_scene(self, objects: list[CollisionObject]) -> None:
        planning_scene = PlanningScene()
        planning_scene.world.collision_objects = objects
        planning_scene.is_diff = True
        self.apply_planning_scene(planning_scene)
        # do not use the publisher, for some reason if a new frame is required that is already there, it won't find it
        # self.planning_scene_diff_publisher.publish(planning_scene)

    def remove_all_markers(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.ns = "scene_manager_markers"
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.markerarray_publisher.publish(marker_array)

    def publish_as_marker(
        self, objects: list[CollisionObject], mesh_paths: list[str] = []
    ) -> None:
        marker_array = MarkerArray()
        if len(mesh_paths) != len(objects):
            mesh_paths = [""] * len(objects)
        for obj, mesh_path in zip(objects, mesh_paths):
            marker = Marker()
            marker.header = obj.header
            marker.ns = "scene_manager_markers"
            marker.text = obj.id
            self.num_markers_counter += 1
            marker.id = self.num_markers_counter
            # TODO handle primitives, for now only mesh
            marker.type = Marker.MESH_RESOURCE
            # TODO handle the already loaded mesh
            # mesh_file = MeshFile()
            # marker.mesh_file = mesh_file
            marker.action = Marker.ADD if obj.operation == CollisionObject.ADD else Marker.DELETE
            marker.mesh_resource = mesh_path
            marker.pose = obj.pose
            # TODO handle local frame transform
            # marker.pose
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = Duration(seconds=10000).to_msg()

            marker_array.markers.append(deepcopy(marker))

            # add a text as well
            self.num_markers_counter += 1
            marker.id = self.num_markers_counter
            marker.scale.z = 0.04
            marker.type = Marker.TEXT_VIEW_FACING
            marker_array.markers.append(deepcopy(marker))

        self.markerarray_publisher.publish(marker_array)
