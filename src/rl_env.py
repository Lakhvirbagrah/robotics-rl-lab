import rospy
import numpy as np

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist


class TurtlebotEnv:

    def __init__(self):

        self.state = None
        self.prev_width = 0.0

        rospy.Subscriber("/yolo_state", Float32MultiArray, self.yolo_callback)

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        rospy.sleep(1)

    def yolo_callback(self, msg):
        self.state = np.array(msg.data)

    def execute_action(self, action):

        cmd = Twist()

        if action == 0:
            cmd.linear.x = 0.2

        elif action == 1:
            cmd.linear.x = -0.2

        elif action == 2:
            cmd.angular.z = 0.5

        elif action == 3:
            cmd.angular.z = -0.5

        self.cmd_pub.publish(cmd)

    def compute_reward(self, state):

        center_x = state[0]
        width = state[1]

        reward = 0
        done = False

        error = abs(center_x - 0.5)
        reward += (1 - error)

        reward += 10 * (width - self.prev_width)

        self.prev_width = width

        if width > 0.5:
            reward += 100
            done = True

        return reward, done

    def step(self, action):

        self.execute_action(action)
        rospy.sleep(0.2)

        if self.state is None:
            return None, -1, False

        next_state = self.state.copy()

        reward, done = self.compute_reward(next_state)

        return next_state, reward, done

    def reset(self):
        self.prev_width = 0.0

        while self.state is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        return self.state.copy()