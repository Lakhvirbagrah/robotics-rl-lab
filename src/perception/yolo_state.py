import rospy
import numpy as np

from std_msgs.msg import Float32MultiArray


class YoloStateProvider:

    def __init__(self, topic="/yolo_state"):
        self.observation = None
        self.sequence = 0

        rospy.Subscriber(
            topic,
            Float32MultiArray,
            self._callback,
            queue_size=1
        )

    def _callback(self, msg):
        self.observation = np.array(
            msg.data,
            dtype=np.float32
        )

        self.sequence += 1

    def get_observation(self):
        if self.observation is None:
            return None

        return self.observation.copy()

    def get_sequence(self):
        return self.sequence

    def wait_for_new_observation(
        self,
        previous_sequence,
        timeout=2.0
    ):
        start_time = rospy.get_time()

        while not rospy.is_shutdown():

            if self.sequence > previous_sequence:
                return self.get_observation()

            if rospy.get_time() - start_time >= timeout:
                return None

            rospy.sleep(0.01)

        return None