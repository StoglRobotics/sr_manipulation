# sr_manipulation package

Stogl Robotics helpers and wrapper to use with MoveIt2.


## Robot Client

Example usage in the launch files:
```
robot_client_node = Node(
    package="sr_moveit2_utils",
    executable="robot_client_node",
    parameters=[
        {"simulation": use_mock_hardware},
    ],
    output={
        "stdout": "screen",
        "stderr": "screen",
    },
)
```
