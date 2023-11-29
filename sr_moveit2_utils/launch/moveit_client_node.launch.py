import os
import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
    IncludeLaunchDescription,
)
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path) as file:
            return yaml.safe_load(file)
    except OSError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    # Robot
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="template_mt_cell_configuration",
            description="Description package of the cell. Usually the argument is not set, \
        it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "configuration_package",
            default_value="template_mt_cell_configuration",
            description="Description package of the cell. Usually the argument is not set, \
        it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value="",
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )
    # moveit
    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_config_package",
            default_value="template_mt_cell_configuration",
            description="Start robot with mock hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "semantic_description_file",
            default_value="cnc_cell.srdf.xacro",
            description="Semantic robot description file (SRDF/XACRO) with of the robot or \
            application. The expected location of the file is '<moveit_config_package>/srdf/'.",
        )
    )
    # Urdf
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_description_file",
            default_value="cnc_cell.urdf.xacro",
            description="YAML file with the workspace description. \
            The expected location of the file is '<configuration_package>/workspace_definitions/'.",
        )
    )
    # Initialize Arguments
    description_package = LaunchConfiguration("description_package")
    prefix = LaunchConfiguration("prefix")
    # use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    moveit_config_package = LaunchConfiguration("moveit_config_package")
    semantic_description_file = LaunchConfiguration("semantic_description_file")
    robot_description_file = LaunchConfiguration("robot_description_file")
    
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare(description_package), "urdf", robot_description_file]
            ),
            " ",
            "prefix:=",
            prefix,
            " ",
            "robot_name:=cnc_cell",
            " ",
        ]
    )
    # Get connected joints
    robot_description = {"robot_description": robot_description_content}

    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(description_package), "rviz", "cnc_cell.rviz"]
    )
    # Moveit2 arguments for the robot_client_node

    # Moveit2 configuration
    config_package = "template_mt_cell_configuration"
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare(moveit_config_package), "srdf", semantic_description_file]
            ),
            " ",
            "robot_name:=",
            "",
            " ",
            "prefix:=",
            prefix,
        ]
    )
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    kinematics_yaml = load_yaml(config_package, "config/kinematics.yaml")

    move_group_config = {
        "planning_pipelines": ["ompl"],
        # "capabilities": [""],
    }

    joint_limits_yaml = load_yaml(config_package, "config/joint_limits.yaml")

    # Ompl
    ompl_planning_pipeline_config = {
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/ResolveConstraintFrames default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        }
    }

    ompl_planning_yaml = load_yaml(config_package, "config/ompl_planning.yaml")

    ompl_planning_pipeline_config["ompl"].update(ompl_planning_yaml)

    robot_description_planning_config = {"robot_description_planning": joint_limits_yaml}

    robot_description_planning_config["robot_description_planning"].update(joint_limits_yaml)

    # Setup trajectory execution
    moveit_simple_controllers_yaml = load_yaml(config_package, "config/ros2_controllers.yaml")
    moveit_controllers = {
        "moveit_simple_controller_manager": moveit_simple_controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 100.0,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # MoveIt2 node
    moveit_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            move_group_config,
            ompl_planning_pipeline_config,
            robot_description_planning_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
        ],
    )
    # Rviz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            kinematics_yaml,
        ],
    )
    
    return LaunchDescription(
        declared_arguments
        + [
            moveit_node,
            rviz_node,

        ]
    )