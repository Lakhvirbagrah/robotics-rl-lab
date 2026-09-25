#!/usr/bin/env python3

import argparse
import time

import rospy
import yaml

from agents.factory import create_agent
from environments.turtlebot_env import TurtlebotEnv
from tasks.factory import create_task


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


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

    config = load_config(
        args.config
    )

    rospy.init_node(
        "position_robustness_evaluator"
    )

    task = create_task(
        config
    )

    env = TurtlebotEnv(
        task,
        action_duration=config[
            "training"
        ].get(
            "action_duration",
            0.1
        )
    )

    agent = create_agent(
        config,
        task
    )

    model_path = config[
        "paths"
    ]["model"]

    agent.load(
        model_path
    )

    # --------------------------------------------------------
    # Pure greedy evaluation
    # --------------------------------------------------------

    agent.epsilon = 0.0

    max_steps = config[
        "training"
    ].get(
        "max_steps_per_episode",
        200
    )

    # --------------------------------------------------------
    # Baseline pose
    # --------------------------------------------------------

    baseline_x = -2.46246
    baseline_y = -5.46862

    # Important:
    # use a yaw where the ball is visible
    # and roughly in the useful camera region.
    baseline_yaw = 1.1345

    # --------------------------------------------------------
    # Nearby position tests
    # --------------------------------------------------------

    positions = [
        (
            "baseline",
            baseline_x,
            baseline_y
        ),

        (
            "left_30cm",
            baseline_x - 0.30,
            baseline_y
        ),

        (
            "right_30cm",
            baseline_x + 0.30,
            baseline_y
        ),

        (
            "closer_30cm",
            baseline_x,
            baseline_y + 0.30
        ),

        (
            "farther_30cm",
            baseline_x,
            baseline_y - 0.30
        ),
    ]

    total_successes = 0
    total_episodes = 0

    all_successful_steps = []
    all_rewards = []

    print()
    print(
        "=========================================="
    )
    print(
        "POSITION ROBUSTNESS EVALUATION"
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
        "Episodes per position:",
        args.episodes_per_position
    )

    print()

    # ========================================================
    # Position loop
    # ========================================================

    for (
        position_name,
        reset_x,
        reset_y
    ) in positions:

        position_successes = 0

        position_steps = []

        position_rewards = []

        print()
        print(
            "------------------------------------------"
        )

        print(
            f"Testing position: "
            f"{position_name}"
        )

        print(
            f"x={reset_x:.3f} "
            f"y={reset_y:.3f} "
            f"yaw={baseline_yaw:.4f}"
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

            state = env.reset(
                x=reset_x,
                y=reset_y,
                yaw=baseline_yaw
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

                total_reward += reward

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
                f"{position_name:14s} | "
                f"Episode {episode:02d} | "
                f"{result:7s} | "
                f"Steps: {step_count:3d} | "
                f"Reward: {total_reward:6.2f} | "
                f"Time: {episode_time:.2f}s"
            )

        # ====================================================
        # Position summary
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
        "POSITION ROBUSTNESS RESULTS"
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