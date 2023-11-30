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
            rviz_node

        ]
    )