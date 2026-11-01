from glob import glob

from setuptools import setup

package_name = "cartpole_rl_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools", "onnxruntime", "numpy"],
    zip_safe=True,
    description="Deep RL CartPole controller (ONNX) as ROS2 nodes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = cartpole_rl_control.policy_node:main",
            "sim_node = cartpole_rl_control.sim_node:main",
        ],
    },
)
