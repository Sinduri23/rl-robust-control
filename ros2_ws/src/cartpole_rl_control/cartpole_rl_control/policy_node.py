"""
ROS2 controller node: subscribes to the robot state, publishes the action from the RL policy.

  sub  /cartpole/state   std_msgs/Float32MultiArray   [x, x_dot, theta, theta_dot]
  pub  /cartpole/action  std_msgs/Int32               0 = left, 1 = right
  param model_path (string)

On a real machine, /cartpole/state would come from sensor drivers and /cartpole/action
would go to the motor driver; this node would not change.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32

from cartpole_rl_control.core import OnnxPolicy


class PolicyNode(Node):
    def __init__(self):
        super().__init__("cartpole_policy")
        path = self.declare_parameter("model_path", "policy.onnx").value
        self.policy = OnnxPolicy(path)
        self.pub = self.create_publisher(Int32, "/cartpole/action", 10)
        self.create_subscription(Float32MultiArray, "/cartpole/state", self.on_state, 10)
        self.get_logger().info(f"Loaded policy from {path}")

    def on_state(self, msg: Float32MultiArray):
        self.pub.publish(Int32(data=self.policy.act(msg.data)))


def main():
    rclpy.init()
    rclpy.spin(PolicyNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
