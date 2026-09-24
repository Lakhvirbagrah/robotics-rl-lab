import random

import rospy

from perception.yolo_state import YoloStateProvider
from robots.turtlebot3 import TurtleBot3


class TurtlebotEnv:

    def __init__(
        self,
        task,
        action_duration=0.1,
        reset_x=-1.61982,
        reset_y=-2.0,
        reset_yaw_base=1.5708,
        reset_yaw_jitter=0.35
    ):
        self.task = task

        self.action_duration = action_duration

        self.reset_x = reset_x
        self.reset_y = reset_y

        self.reset_yaw_base = reset_yaw_base
        self.reset_yaw_jitter = reset_yaw_jitter

        self.robot = TurtleBot3()

        self.perception = YoloStateProvider()

    def reset(self):
        # Stop any movement left from previous episode
        self.robot.stop()

        # Remember current perception sequence so we can
        # wait for a new YOLO result after teleporting.
        previous_sequence = (
            self.perception.get_sequence()
        )

        # Small random orientation around the direction
        # facing the Stop Sign.
        direction = random.choice(
            [-1.0, 1.0]
        )

        yaw_offset = direction * random.uniform(
            0.20,
            0.35
        )

        reset_yaw = (
            self.reset_yaw_base
            + yaw_offset
        )
        # Reset TurtleBot near the target.
        self.robot.reset_pose(
            x=self.reset_x,
            y=self.reset_y,
            yaw=reset_yaw
        )

        # Reset task-specific state such as previous_error.
        self.task.reset()

        # Wait for a genuinely new YOLO observation
        # after the Gazebo teleport.
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
                "Waiting for fresh YOLO observation..."
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
            f"Episode reset | "
            f"x={self.reset_x:.2f} "
            f"y={self.reset_y:.2f} "
            f"yaw={reset_yaw:.3f}"
        )

        return self.task.get_state(
            observation
        )

    def step(self, action_index):
        # Convert RL action number to semantic robot command.
        action = self.task.get_action(
            action_index
        )

        self.robot.execute_action(
            action
        )

        # Allow robot to execute action.
        rospy.sleep(
            self.action_duration
        )

        observation = (
            self.perception.get_observation()
        )

        if observation is None:
            return None, 0.0, False

        next_state = self.task.get_state(
            observation
        )

        reward = self.task.compute_reward(
            observation
        )

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