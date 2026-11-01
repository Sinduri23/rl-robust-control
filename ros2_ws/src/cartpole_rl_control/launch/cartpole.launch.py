from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model_path", description="Absolute path to models/policy.onnx"),
        DeclareLaunchArgument("mass_scale", default_value="1.0"),
        Node(package="cartpole_rl_control", executable="sim_node",
             parameters=[{"mass_scale": LaunchConfiguration("mass_scale")}]),
        Node(package="cartpole_rl_control", executable="policy_node",
             parameters=[{"model_path": LaunchConfiguration("model_path")}]),
    ])
