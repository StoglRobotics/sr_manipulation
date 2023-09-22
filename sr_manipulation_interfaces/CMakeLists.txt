cmake_minimum_required(VERSION 3.8)
project(sr_manipulation_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  std_msgs
  geometry_msgs
  moveit_msgs
  shape_msgs
  action_msgs
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

set(msg_files
  "msg/ServiceResult.msg"
  "msg/ObjectDescriptor.msg"
  "msg/PlanExecState.msg"
  "msg/ManipType.msg"
  "msg/MoveWaypoint.msg"
)

set(action_files
  "action/PlanMoveTo.action"
  "action/Manip.action"
)

set(srv_files
  srv/AddObjects.srv
  srv/RemoveObjects.srv
  srv/GetObjectPose.srv
  srv/AttachObject.srv
  srv/DetachObject.srv
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES builtin_interfaces ${THIS_PACKAGE_INCLUDE_DEPENDS}
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
