"""
ROS2 simulator node: stands in for the real machine.

Publishes the state at 50 Hz, applies the latest action it received, resets when the pole
falls, and logs how long each episode lasted.

  pub  /cartpole/state   std_msgs/Float32MultiArray
  sub  /cartpole/action  std_msgs/Int32
  param mass_scale (double, default 1.0)
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32

from cartpole_rl_control.core import CartPoleSim


class SimNode(Node):
    def __init__(self):
        super().__init__("cartpole_sim")
        self.sim = CartPoleSim(mass_scale=self.declare_parameter("mass_scale", 1.0).value)
        self.action, self.steps = 0, 0
        self.pub = self.create_publisher(Float32MultiArray, "/cartpole/state", 10)
        self.create_subscription(Int32, "/cartpole/action", self.on_action, 10)
        self.create_timer(self.sim.tau, self.tick)   # 0.02 s = 50 Hz control loop

    def on_action(self, msg: Int32):
        self.action = msg.data

    def tick(self):
        done = self.sim.step(self.action)
        self.steps += 1
        if done or self.steps >= 500:
            self.get_logger().info(f"episode ended after {self.steps} steps")
            self.sim.reset()
            self.steps = 0
        self.pub.publish(Float32MultiArray(data=[float(v) for v in self.sim.state]))


def main():
    rclpy.init()
    rclpy.spin(SimNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
