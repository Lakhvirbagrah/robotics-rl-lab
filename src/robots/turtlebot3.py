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

        if action == 0:
            cmd.linear.x = 0.2

        elif action == 1:
            cmd.linear.x = -0.2

        elif action == 2:
            cmd.angular.z = 0.5

        elif action == 3:
            cmd.angular.z = -0.5

        elif action == 4:
            pass

        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())