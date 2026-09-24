import rospy

from std_srvs.srv import Empty

from perception.yolo_state import YoloStateProvider
from robots.turtlebot3 import TurtleBot3


class TurtlebotEnv:

    def __init__(
        self,
        task,
        action_duration=0.1
    ):
        self.task = task
        self.action_duration = action_duration

        self.robot = TurtleBot3()
        self.perception = YoloStateProvider()

        rospy.wait_for_service(
            "/gazebo/reset_world"
        )

        self.reset_world = rospy.ServiceProxy(
            "/gazebo/reset_world",
            Empty
        )

        # Fixed training pose based on the
        # actual cricket ball position in Gazebo.
        #
        # Cricket ball:
        # x = -2.46246
        # y = -3.96862
        #
        # Robot starts 1.5 m behind it,
        # facing toward +Y.
        self.robot_reset_x = -2.46246
        self.robot_reset_y = -5.46862
        self.robot_reset_yaw = 1.5708

    def reset(self):
        # Stop any command from the previous episode.
        self.robot.stop()

        rospy.sleep(0.1)

        # Remember current perception sequence.
        # We will wait for a NEW YOLO observation
        # after resetting the robot.
        previous_sequence = (
            self.perception.get_sequence()
        )

        # Reset the Gazebo world first.
        # This restores the cricket ball and
        # other world objects.
        try:
            self.reset_world()

        except rospy.ServiceException as error:
            rospy.logerr(
                f"Gazebo world reset failed: {error}"
            )

        # Give Gazebo time to finish the reset.
        rospy.sleep(0.2)

        # Now manually put TurtleBot near
        # the cricket ball.
        self.robot.reset_pose(
            x=self.robot_reset_x,
            y=self.robot_reset_y,
            yaw=self.robot_reset_yaw
        )

        # Give Gazebo time to apply the teleport.
        rospy.sleep(0.2)

        # Make sure the robot starts stationary.
        self.robot.stop()

        # Reset task-specific internal state.
        self.task.reset()

        # Wait for a fresh YOLO observation
        # generated AFTER the robot was moved.
        observation = (
            self.perception.wait_for_new_observation(
                previous_sequence,
                timeout=2.0
            )
        )

        while (
            observation is None
            and not rospy.is_shutdown()
        ):
            rospy.logwarn(
                "Waiting for fresh YOLO observation after episode reset..."
            )

            previous_sequence = (
                self.perception.get_sequence()
            )

            observation = (
                self.perception.wait_for_new_observation(
                    previous_sequence,
                    timeout=2.0
                )
            )

        rospy.loginfo(
            "Episode reset | "
            f"x={self.robot_reset_x:.2f} "
            f"y={self.robot_reset_y:.2f} "
            f"yaw={self.robot_reset_yaw:.3f}"
        )

        return self.task.get_state(
            observation
        )

    def step(self, action_index):
        # Convert DQN action index into
        # semantic task action.
        action = self.task.get_action(
            action_index
        )

        # Execute robot command.
        self.robot.execute_action(
            action
        )

        # Let the robot execute the action.
        rospy.sleep(
            self.action_duration
        )

        # Get latest YOLO observation.
        observation = (
            self.perception.get_observation()
        )

        if observation is None:
            return (
                None,
                0.0,
                False
            )

        # Convert perception into RL state.
        next_state = self.task.get_state(
            observation
        )

        # Calculate reward.
        reward = self.task.compute_reward(
            observation
        )

        # Check task success.
        done = self.task.is_done(
            observation
        )

        if done:
            self.robot.stop()

        return (
            next_state,
            reward,
            done
        )