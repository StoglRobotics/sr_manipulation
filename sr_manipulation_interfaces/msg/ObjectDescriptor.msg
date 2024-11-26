# ID of the object, will be used for as name in the Planning scene and 'frame_id' in TF
string id
geometry_msgs/PoseStamped pose
string[] paths_to_mesh
# bounding box size (in case mesh not provided)
geometry_msgs/Vector3 bbox_size
# relative pose of mesh/primitive
geometry_msgs/Pose shape_pose
VisTransform[] transforms

# set of links that the added object is allowed to touch (and that are not part of the parent link)
string[] allowed_touch_links
