####################################################
#################### Deprecated ####################
string[] paths_to_mesh
# this field is used in two ways:
# 1. bounding box size (in case mesh not provided)
# 2. scale factor for the mesh (in case mesh provided)
geometry_msgs/Vector3 bbox_size
# relative pose of mesh/primitive
geometry_msgs/Pose shape_pose
#################### Deprecated ####################

####################################################
# ID of the object, will be used for as name in the Planning scene and 'frame_id' in TF
string id

# The object's pose relative to the header frame.
# The shapes and subframe poses are defined relative to this pose.
geometry_msgs/PoseStamped pose

# Solid geometric primitives
shape_msgs/SolidPrimitive[] primitives
geometry_msgs/Pose[] primitive_poses

# Meshes
# either mesh_paths+mesh_scales or meshes can be used, but not both
string[] mesh_paths
geometry_msgs/Vector3[] mesh_scales # when not defined, [1,1,1] is assumed
shape_msgs/Mesh[] meshes
geometry_msgs/Pose[] mesh_poses

# Named subframes on the object. Use these to define points of interest on the object that you want
# to plan with (e.g. "tip", "spout", "handle"). The id of the object will be prepended to the subframe.
# If an object with the id "screwdriver" and a subframe "tip" is in the scene, you can use the frame
# "screwdriver/tip" for planning.
# The length of the subframe_names and subframe_poses has to be identical.
string[] subframe_names
geometry_msgs/Pose[] subframe_poses

# whether the object should be moved also when the parent is moved
string parent_object_id

# set of links that the added object is allowed to touch (and that are not part of the parent link)
string[] allowed_touch_links
