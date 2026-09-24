import rospy
from geometry_msgs.msg import Twist


class TurtleBot3:
    def __init__(self, cmd_vel_topic="/cmd_vel"):
        self.cmd_pub = rospy.Publisher(
            cmd_vel_topic,
            Twist,
            queue_size=10
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
        cmd = Twist()
        self.cmd_pub.publish(cmd)