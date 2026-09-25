#!/usr/bin/env python3

import argparse
import copy
import time

import rospy
import yaml

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState

from agents.factory import create_agent
from environments.turtlebot_env import TurtlebotEnv
from tasks.factory import create_task


# ============================================================
# Configuration
# ============================================================

def load_config(config_path):

    with open(
        config_path,
        "r"
    ) as file:

        return yaml.safe_load(
            file
        )


# ============================================================
# Gazebo ball controller
# ============================================================

class BallController:

    def __init__(
        self,
        model_name="cricket_ball"
    ):

        self.model_name = model_name

        rospy.wait_for_service(
            "/gazebo/get_model_state"
        )

        rospy.wait_for_service(
            "/gazebo/set_model_state"
        )

        self.get_model_state_service = (
            rospy.ServiceProxy(
                "/gazebo/get_model_state",
                GetModelState
            )
        )

        self.set_model_state_service = (
            rospy.ServiceProxy(
                "/gazebo/set_model_state",
                SetModelState
            )
        )

    # --------------------------------------------------------
    # Read ball pose
    # --------------------------------------------------------

    def get_pose(self):

        response = (
            self.get_model_state_service(
                self.model_name,
                "world"
            )
        )

        if not response.success:

            raise RuntimeError(
                f"Could not read model "
                f"{self.model_name}: "
                f"{response.status_message}"
            )

        return copy.deepcopy(
            response.pose
        )

    # --------------------------------------------------------
    # Set ball pose
    # --------------------------------------------------------

    def set_pose(
        self,
        pose
    ):

        model_state = ModelState()

        model_state.model_name = (
            self.model_name
        )

        model_state.reference_frame = (
            "world"
        )

        model_state.pose = copy.deepcopy(
            pose
        )

        # Keep ball stationary after teleport.
        model_state.twist.linear.x = 0.0
        model_state.twist.linear.y = 0.0
        model_state.twist.linear.z = 0.0

        model_state.twist.angular.x = 0.0
        model_state.twist.angular.y = 0.0
        model_state.twist.angular.z = 0.0

        response = (
            self.set_model_state_service(
                model_state
            )
        )

        if not response.success:

            raise RuntimeError(
                f"Could not move model "
                f"{self.model_name}: "
                f"{response.status_message}"
            )


# ============================================================
# Fresh observation after ball movement
# ============================================================

def get_fresh_state_after_ball_move(
    env,
    task,
    previous_sequence
):

    observation = (
        env.perception.wait_for_new_observation(
            previous_sequence,
            timeout=2.0
        )
    )

    if observation is None:

        rospy.logwarn(
            "No fresh YOLO observation "
            "after ball movement."
        )

        observation = (
            env.perception.get_observation()
        )

    state = (
        task.get_state(
            observation
        )
    )

    return state


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    parser.add_argument(
        "--episodes-per-position",
        type=int,
        default=5
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    # --------------------------------------------------------
    # ROS
    # --------------------------------------------------------

    rospy.init_node(
        "ball_position_robustness_evaluator"
    )

    # --------------------------------------------------------
    # Task
    # --------------------------------------------------------

    task = create_task(
        config
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = TurtlebotEnv(
        task,
        action_duration=config[
            "training"
        ].get(
            "action_duration",
            0.1
        )
    )

    # --------------------------------------------------------
    # DQN
    # --------------------------------------------------------

    agent = create_agent(
        config,
        task
    )

    model_path = (
        config[
            "paths"
        ]["model"]
    )

    agent.load(
        model_path
    )

    # Pure greedy evaluation.
    agent.epsilon = 0.0

    max_steps = (
        config[
            "training"
        ].get(
            "max_steps_per_episode",
            200
        )
    )

    # --------------------------------------------------------
    # Ball controller
    # --------------------------------------------------------

    ball = BallController(
        model_name="cricket_ball"
    )

    # --------------------------------------------------------
    # Reset world once so we can read the true baseline
    # cricket-ball position from Gazebo.
    # --------------------------------------------------------

    env.reset()

    baseline_ball_pose = (
        ball.get_pose()
    )

    baseline_ball_x = (
        baseline_ball_pose.position.x
    )

    baseline_ball_y = (
        baseline_ball_pose.position.y
    )

    baseline_ball_z = (
        baseline_ball_pose.position.z
    )

    print()
    print(
        "Baseline ball position:"
    )

    print(
        f"x={baseline_ball_x:.4f} "
        f"y={baseline_ball_y:.4f} "
        f"z={baseline_ball_z:.4f}"
    )

    # --------------------------------------------------------
    # Controlled robot pose
    #
    # We use a yaw from our earlier probe where the target
    # was clearly visible.
    # --------------------------------------------------------

    robot_x = -2.46246
    robot_y = -5.46862
    robot_yaw = 1.1345

    # --------------------------------------------------------
    # Ball test positions
    #
    # Labels refer to WORLD coordinates, not camera left/right.
    # --------------------------------------------------------

    ball_positions = [

        (
            "baseline",
            0.0,
            0.0
        ),

        (
            "world_x_minus_30cm",
            -0.30,
            0.0
        ),

        (
            "world_x_plus_30cm",
            0.30,
            0.0
        ),

        (
            "world_y_minus_30cm",
            0.0,
            -0.30
        ),

        (
            "world_y_plus_30cm",
            0.0,
            0.30
        ),
    ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_episodes = 0
    total_successes = 0

    all_rewards = []
    all_successful_steps = []

    print()
    print(
        "=========================================="
    )

    print(
        "BALL POSITION ROBUSTNESS EVALUATION"
    )

    print(
        "=========================================="
    )

    print(
        "Model:",
        model_path
    )

    print(
        "Epsilon:",
        agent.epsilon
    )

    print(
        "Episodes per ball position:",
        args.episodes_per_position
    )

    print()

    # ========================================================
    # Ball-position loop
    # ========================================================

    for (
        position_name,
        x_offset,
        y_offset
    ) in ball_positions:

        position_successes = 0

        position_rewards = []
        position_steps = []

        test_ball_x = (
            baseline_ball_x
            + x_offset
        )

        test_ball_y = (
            baseline_ball_y
            + y_offset
        )

        print()
        print(
            "------------------------------------------"
        )

        print(
            "Testing ball position:",
            position_name
        )

        print(
            f"Ball x={test_ball_x:.3f} "
            f"y={test_ball_y:.3f}"
        )

        print(
            "------------------------------------------"
        )

        # ====================================================
        # Episodes
        # ====================================================

        for episode in range(
            args.episodes_per_position
        ):

            # ------------------------------------------------
            # reset_world restores the ball to its original
            # world position, so reset the robot/world FIRST.
            # ------------------------------------------------

            env.reset(
                x=robot_x,
                y=robot_y,
                yaw=robot_yaw
            )

            # ------------------------------------------------
            # Save current perception sequence BEFORE moving
            # the target.
            # ------------------------------------------------

            previous_sequence = (
                env.perception.get_sequence()
            )

            # ------------------------------------------------
            # Build test ball pose
            # ------------------------------------------------

            test_pose = copy.deepcopy(
                baseline_ball_pose
            )

            test_pose.position.x = (
                test_ball_x
            )

            test_pose.position.y = (
                test_ball_y
            )

            # Preserve original Z and orientation.
            test_pose.position.z = (
                baseline_ball_z
            )

            # ------------------------------------------------
            # Move ball
            # ------------------------------------------------

            ball.set_pose(
                test_pose
            )

            rospy.sleep(
                0.2
            )

            # ------------------------------------------------
            # Reset task history because the target changed.
            # ------------------------------------------------

            task.reset()

            # ------------------------------------------------
            # Get perception corresponding to NEW ball pose.
            # ------------------------------------------------

            state = (
                get_fresh_state_after_ball_move(
                    env,
                    task,
                    previous_sequence
                )
            )

            print(
                f"Initial state: "
                f"{state}"
            )

            done = False
            step_count = 0
            total_reward = 0.0

            episode_start = (
                time.time()
            )

            # ================================================
            # Greedy policy
            # ================================================

            while (
                not done
                and
                step_count < max_steps
                and
                not rospy.is_shutdown()
            ):

                action = (
                    agent.select_action(
                        state
                    )
                )

                (
                    next_state,
                    reward,
                    done
                ) = env.step(
                    action
                )

                if next_state is None:

                    continue

                state = next_state

                total_reward += (
                    reward
                )

                step_count += 1

            env.robot.stop()

            episode_time = (
                time.time()
                - episode_start
            )

            total_episodes += 1

            all_rewards.append(
                total_reward
            )

            if done:

                position_successes += 1

                total_successes += 1

                position_steps.append(
                    step_count
                )

                all_successful_steps.append(
                    step_count
                )

                result = "SUCCESS"

            else:

                result = "FAIL"

            position_rewards.append(
                total_reward
            )

            print(
                f"{position_name:22s} | "
                f"Episode {episode:02d} | "
                f"{result:7s} | "
                f"Steps: {step_count:3d} | "
                f"Reward: {total_reward:7.2f} | "
                f"Time: {episode_time:.2f}s"
            )

        # ====================================================
        # Per-position summary
        # ====================================================

        success_rate = (
            position_successes
            / args.episodes_per_position
            * 100.0
        )

        average_reward = (
            sum(
                position_rewards
            )
            / len(
                position_rewards
            )
        )

        print()

        print(
            f"{position_name} summary"
        )

        print(
            f"Successes: "
            f"{position_successes}/"
            f"{args.episodes_per_position}"
        )

        print(
            f"Success rate: "
            f"{success_rate:.1f}%"
        )

        print(
            f"Average reward: "
            f"{average_reward:.2f}"
        )

        if position_steps:

            average_steps = (
                sum(
                    position_steps
                )
                / len(
                    position_steps
                )
            )

            print(
                f"Average successful steps: "
                f"{average_steps:.1f}"
            )

        else:

            print(
                "Average successful steps: N/A"
            )

    # ========================================================
    # Restore original ball position
    # ========================================================

    try:

        env.reset()

        ball.set_pose(
            baseline_ball_pose
        )

    except Exception as error:

        rospy.logwarn(
            f"Could not restore original ball pose: "
            f"{error}"
        )

    # ========================================================
    # Overall summary
    # ========================================================

    overall_success_rate = (
        total_successes
        / total_episodes
        * 100.0
    )

    average_reward = (
        sum(
            all_rewards
        )
        / len(
            all_rewards
        )
    )

    print()
    print(
        "=========================================="
    )

    print(
        "BALL POSITION ROBUSTNESS RESULTS"
    )

    print(
        "=========================================="
    )

    print(
        f"Total episodes: "
        f"{total_episodes}"
    )

    print(
        f"Successes: "
        f"{total_successes}"
    )

    print(
        f"Failures: "
        f"{total_episodes - total_successes}"
    )

    print(
        f"Overall success rate: "
        f"{overall_success_rate:.1f}%"
    )

    print(
        f"Average reward: "
        f"{average_reward:.2f}"
    )

    if all_successful_steps:

        average_steps = (
            sum(
                all_successful_steps
            )
            / len(
                all_successful_steps
            )
        )

        print(
            f"Average successful steps: "
            f"{average_steps:.1f}"
        )

        print(
            f"Best successful episode: "
            f"{min(all_successful_steps)} steps"
        )

        print(
            f"Worst successful episode: "
            f"{max(all_successful_steps)} steps"
        )

    print(
        "=========================================="
    )

    env.robot.stop()


if __name__ == "__main__":

    main()