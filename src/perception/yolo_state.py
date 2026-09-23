import rospy
import numpy as np

from std_msgs.msg import Float32MultiArray


class YoloStateProvider:
    def __init__(self, topic="/yolo_state"):
        self.observation = None

        rospy.Subscriber(
            topic,
            Float32MultiArray,
            self._callback
        )

    def _callback(self, msg):
        self.observation = np.array(
            msg.data,
            dtype=np.float32
        )

    def get_observation(self):
        if self.observation is None:
            return None

        return self.observation.copy()