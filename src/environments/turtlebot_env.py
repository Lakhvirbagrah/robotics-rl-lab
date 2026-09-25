#!/usr/bin/env python3

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

        self.action_duration = (
            action_duration
        )

        # ----------------------------------------------------
        # Robot
        # ----------------------------------------------------

        self.robot = TurtleBot3()

        # ----------------------------------------------------
        # Perception
        # ----------------------------------------------------

        self.perception = (
            YoloStateProvider()
        )

        # ----------------------------------------------------
        # Gazebo reset service
        # ----------------------------------------------------

        rospy.wait_for_service(
            "/gazebo/reset_world"
        )

        self.reset_world = (
            rospy.ServiceProxy(
                "/gazebo/reset_world",
                Empty
            )
        )

        # ----------------------------------------------------
        # Default controlled reset pose
        # ----------------------------------------------------

        self.default_reset_x = (
            -2.46246
        )

        self.default_reset_y = (
            -5.46862
        )

        self.default_reset_yaw = (
            1.5708
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        x=None,
        y=None,
        yaw=None
    ):

        # ----------------------------------------------------
        # Use default values if no custom pose supplied
        # ----------------------------------------------------

        if x is None:
            x = self.default_reset_x

        if y is None:
            y = self.default_reset_y

        if yaw is None:
            yaw = self.default_reset_yaw

        # ----------------------------------------------------
        # Stop robot before resetting
        # ----------------------------------------------------

        self.robot.stop()

        rospy.sleep(
            0.1
        )

        # ----------------------------------------------------
        # Remember perception sequence
        # ----------------------------------------------------

        previous_sequence = (
            self.perception.get_sequence()
        )

        # ----------------------------------------------------
        # Reset Gazebo world
        # ----------------------------------------------------

        try:

            self.reset_world()

        except rospy.ServiceException as error:

            rospy.logerr(
                f"Gazebo reset failed: {error}"
            )

        rospy.sleep(
            0.2
        )

        # ----------------------------------------------------
        # Teleport robot to requested pose
        # ----------------------------------------------------

        self.robot.reset_pose(
            x=x,
            y=y,
            yaw=yaw
        )

        rospy.sleep(
            0.2
        )

        # ----------------------------------------------------
        # Ensure robot is stationary
        # ----------------------------------------------------

        self.robot.stop()

        # ----------------------------------------------------
        # Reset task state
        # ----------------------------------------------------

        self.task.reset()

        # ----------------------------------------------------
        # Wait for fresh perception
        # ----------------------------------------------------

        observation = (
            self.perception.wait_for_new_observation(
                previous_sequence,
                timeout=2.0
            )
        )

        if observation is None:

            rospy.logwarn(
                "No fresh YOLO observation "
                "after environment reset."
            )

            observation = (
                self.perception.get_observation()
            )

        # ----------------------------------------------------
        # Convert perception to RL state
        # ----------------------------------------------------

        state = self.task.get_state(
            observation
        )

        rospy.loginfo(
            f"Episode reset | "
            f"x={x:.2f} "
            f"y={y:.2f} "
            f"yaw={yaw:.3f}"
        )

        return state

    # ========================================================
    # Step
    # ========================================================

    def step(
        self,
        action
    ):

        # ----------------------------------------------------
        # Convert RL action index to semantic action
        # ----------------------------------------------------

        semantic_action = (
            self.task.get_action(
                action
            )
        )

        # ----------------------------------------------------
        # Execute robot action
        # ----------------------------------------------------

        self.robot.execute_action(
            semantic_action
        )

        # ----------------------------------------------------
        # Allow action to take effect
        # ----------------------------------------------------

        rospy.sleep(
            self.action_duration
        )

        # ----------------------------------------------------
        # Get latest YOLO observation
        # ----------------------------------------------------

        observation = (
            self.perception.get_observation()
        )

        # ----------------------------------------------------
        # Construct RL state
        # ----------------------------------------------------

        next_state = (
            self.task.get_state(
                observation
            )
        )

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        reward = (
        self.task.compute_reward(
            observation
        ))

        # ----------------------------------------------------
        # Done condition
        # ----------------------------------------------------

        done = (
            self.task.is_done(
                observation
            )
        )

        if done:

            self.robot.stop()

        return (
            next_state,
            reward,
            done
        )