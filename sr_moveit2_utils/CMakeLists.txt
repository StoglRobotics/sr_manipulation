cmake_minimum_required(VERSION 3.8)
project(sr_moveit2_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


set(THIS_PACKAGE_INCLUDE_DEPENDS
  rclcpp
  rclcpp_lifecycle
  rclpy
  moveit_core
  moveit_msgs
  sr_manipulation_interfaces
)

# find dependencies
find_package(ament_cmake REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

#install(DIRECTORY
#  config launch
#DESTINATION share/${PROJECT_NAME}/
#)

ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  sr_moveit2_utils/scene_manager_node.py
DESTINATION lib/${PROJECT_NAME}
)

ament_package()
