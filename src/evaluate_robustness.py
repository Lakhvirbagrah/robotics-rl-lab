#!/usr/bin/env python3

import argparse
import math
import time

import rospy
import yaml

from agents.factory import create_agent
from environments.turtlebot_env import TurtlebotEnv
from tasks.factory import create_task


# ============================================================
# Configuration
# ============================================================

def load_config(
    config_path
):

    with open(
        config_path,
        "r"
    ) as file:

        return yaml.safe_load(
            file
        )


# ============================================================
# Degrees to radians
# ============================================================

def degrees_to_radians(
    degrees
):

    return (
        degrees
        * math.pi
        / 180.0
    )


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
        "--episodes-per-yaw",
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
    # ROS node
    # --------------------------------------------------------

    rospy.init_node(
        "robotics_rl_robustness_evaluator"
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

    action_duration = (
        config[
            "training"
        ].get(
            "action_duration",
            0.1
        )
    )

    env = TurtlebotEnv(
        task,
        action_duration=action_duration
    )

    # --------------------------------------------------------
    # Agent
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

    # --------------------------------------------------------
    # PURE GREEDY EVALUATION
    # --------------------------------------------------------

    agent.epsilon = 0.0

    # --------------------------------------------------------
    # Episode limit
    # --------------------------------------------------------

    max_steps = (
        config[
            "training"
        ].get(
            "max_steps_per_episode",
            200
        )
    )

    # --------------------------------------------------------
    # Baseline reset pose
    # --------------------------------------------------------

    reset_x = -2.46246
    reset_y = -5.46862

    baseline_yaw = 1.5708

    # --------------------------------------------------------
    # Robustness test offsets
    # --------------------------------------------------------

    yaw_offsets_degrees = [
    -45.0,
    -35.0,

    -30.0,
    -25.0,

    -15.0,
    -5.0]

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    total_successes = 0
    total_episodes = 0

    all_steps = []
    all_rewards = []

    print()
    print(
        "=========================================="
    )

    print(
        "DQN ROBUSTNESS EVALUATION"
    )

    print(
        "=========================================="
    )

    print(
        "Model:",
        model_path
    )

    print(
        "Evaluation epsilon:",
        agent.epsilon
    )

    print(
        "Episodes per yaw:",
        args.episodes_per_yaw
    )

    print()

    # ========================================================
    # Test each yaw
    # ========================================================

    for yaw_offset_deg in (
        yaw_offsets_degrees
    ):

        yaw_offset_rad = (
            degrees_to_radians(
                yaw_offset_deg
            )
        )

        test_yaw = (
            baseline_yaw
            + yaw_offset_rad
        )

        yaw_successes = 0

        yaw_steps = []

        yaw_rewards = []

        print()
        print(
            "------------------------------------------"
        )

        print(
            f"Testing yaw offset: "
            f"{yaw_offset_deg:+.1f}°"
        )

        print(
            f"Absolute yaw: "
            f"{test_yaw:.4f} rad"
        )

        print(
            "------------------------------------------"
        )

        # ====================================================
        # Episodes for this yaw
        # ====================================================

        for episode in range(
            args.episodes_per_yaw
        ):

            state = env.reset(
                x=reset_x,
                y=reset_y,
                yaw=test_yaw
            )

            done = False

            step_count = 0

            total_reward = 0.0

            episode_start = (
                time.time()
            )

            while (
                not done
                and
                step_count < max_steps
                and
                not rospy.is_shutdown()
            ):

                # --------------------------------------------
                # Greedy DQN action
                # --------------------------------------------

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

            # ------------------------------------------------
            # Stop robot
            # ------------------------------------------------

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

                yaw_successes += 1

                total_successes += 1

                yaw_steps.append(
                    step_count
                )

                all_steps.append(
                    step_count
                )

                result = "SUCCESS"

            else:

                result = "FAIL"

            yaw_rewards.append(
                total_reward
            )

            print(
                f"Yaw {yaw_offset_deg:+5.1f}° | "
                f"Episode {episode:02d} | "
                f"{result} | "
                f"Steps: {step_count:3d} | "
                f"Reward: {total_reward:6.2f} | "
                f"Time: {episode_time:.2f}s"
            )

        # ====================================================
        # Summary for this yaw
        # ====================================================

        yaw_success_rate = (
            yaw_successes
            / args.episodes_per_yaw
            * 100.0
        )

        average_reward = (
            sum(
                yaw_rewards
            )
            / len(
                yaw_rewards
            )
        )

        print()

        print(
            f"Yaw {yaw_offset_deg:+.1f}° summary"
        )

        print(
            f"Successes: "
            f"{yaw_successes}/"
            f"{args.episodes_per_yaw}"
        )

        print(
            f"Success rate: "
            f"{yaw_success_rate:.1f}%"
        )

        print(
            f"Average reward: "
            f"{average_reward:.2f}"
        )

        if yaw_steps:

            average_steps = (
                sum(
                    yaw_steps
                )
                / len(
                    yaw_steps
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
    # Overall result
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
        "ROBUSTNESS EVALUATION RESULTS"
    )

    print(
        "=========================================="
    )

    print(
        f"Total episodes: "
        f"{total_episodes}"
    )

    print(
        f"Total successes: "
        f"{total_successes}"
    )

    print(
        f"Total failures: "
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

    if all_steps:

        average_steps = (
            sum(
                all_steps
            )
            / len(
                all_steps
            )
        )

        print(
            f"Average successful steps: "
            f"{average_steps:.1f}"
        )

        print(
            f"Best successful episode: "
            f"{min(all_steps)} steps"
        )

        print(
            f"Worst successful episode: "
            f"{max(all_steps)} steps"
        )

    print(
        "=========================================="
    )

    env.robot.stop()


if __name__ == "__main__":
    main()