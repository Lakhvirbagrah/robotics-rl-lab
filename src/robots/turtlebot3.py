import rospy

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


class TurtleBot3:
    def __init__(
        self,
        cmd_vel_topic="/cmd_vel",
        model_name="turtlebot3"
    ):
        self.model_name = model_name

        self.cmd_pub = rospy.Publisher(
            cmd_vel_topic,
            Twist,
            queue_size=10
        )

        rospy.wait_for_service(
            "/gazebo/set_model_state"
        )

        self.set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state",
            SetModelState
        )

    def execute_action(self, action):
        cmd = Twist()

        if action == "forward":
            cmd.linear.x = 0.2

        elif action == "backward":
            cmd.linear.x = -0.2

        elif action == "turn_left":
            cmd.angular.z = 0.5

        elif action == "turn_right":
            cmd.angular.z = -0.5

        elif action == "stop":
            pass

        else:
            raise ValueError(
                f"Unknown robot action: {action}"
            )

        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(
            Twist()
        )

    def reset_pose(
        self,
        x=0.0,
        y=0.0,
        yaw=0.0
    ):
        state = ModelState()

        state.model_name = self.model_name

        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.0

        # Quaternion for yaw-only rotation
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
        state.pose.orientation.w = 1.0

        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0

        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        state.reference_frame = "world"

        self.set_model_state(
            state
        )