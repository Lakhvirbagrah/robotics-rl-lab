import rospy
import numpy as np

from std_msgs.msg import Float32MultiArray
from robots.turtlebot3 import TurtleBot3


class TurtlebotEnv:
    def __init__(self, task):
        self.task = task
        self.observation = None

        rospy.Subscriber(
            "/yolo_state",
            Float32MultiArray,
            self.yolo_callback
        )

        self.robot = TurtleBot3()

        rospy.sleep(1)

    def yolo_callback(self, msg):
        self.observation = np.array(msg.data, dtype=np.float32)

    def execute_action(self, action):
        self.robot.execute_action(action)

    def step(self, action):
        self.execute_action(action)

        rospy.sleep(0.2)

        if self.observation is None:
            return None, -1.0, False

        observation = self.observation.copy()

        next_state = self.task.get_state(observation)
        reward = self.task.compute_reward(observation)
        done = self.task.is_done(observation)

        return next_state, reward, done

    def reset(self):
        self.task.reset()

        while self.observation is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        observation = self.observation.copy()

        return self.task.get_state(observation)