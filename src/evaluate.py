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
        required=True,
        help="Path to experiment YAML configuration"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes"
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    rospy.init_node(
        "robotics_rl_evaluator"
    )

    # -------------------------------------------------
    # Create task
    # -------------------------------------------------

    task = create_task(
        config
    )

    print()
    print(
        "Experiment:",
        config["experiment"]["name"]
    )

    print(
        "Task:",
        task.__class__.__name__
    )

    print(
        "State dimension:",
        task.get_state_dim()
    )

    print(
        "Action dimension:",
        task.get_action_dim()
    )

    print(
        "Actions:",
        task.get_actions()
    )

    # -------------------------------------------------
    # Create environment
    # -------------------------------------------------

    env = TurtlebotEnv(
        task,
        action_duration=config[
            "training"
        ].get(
            "action_duration",
            0.1
        )
    )

    # -------------------------------------------------
    # Create agent
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Evaluation mode
    # -------------------------------------------------

    agent.epsilon = 0.0

    max_steps = config[
        "training"
    ].get(
        "max_steps_per_episode",
        200
    )

    print()
    print(
        "===================================="
    )
    print(
        "DQN EVALUATION STARTED"
    )
    print(
        "===================================="
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
        "Episodes:",
        args.episodes
    )

    print()

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    successes = 0
    failures = 0

    total_steps = 0
    total_rewards = 0.0
    total_time = 0.0

    successful_steps = []

    # -------------------------------------------------
    # Evaluation episodes
    # -------------------------------------------------

    for episode in range(
        args.episodes
    ):

        state = env.reset()

        total_reward = 0.0
        step_count = 0
        done = False

        episode_start = (
            time.time()
        )

        print()
        print(
            f"---------- Episode {episode:02d} ----------"
        )

        while (
            not done
            and step_count < max_steps
            and not rospy.is_shutdown()
        ):

            # -----------------------------------------
            # Greedy DQN action
            # -----------------------------------------

            action = agent.select_action(
                state
            )

            action_name = task.get_action(
                action
            )

            # -----------------------------------------
            # Print policy decision
            # -----------------------------------------

            print(
                f"Episode {episode:02d} | "
                f"Step {step_count:03d} | "
                f"State: {state} | "
                f"Action: {action} "
                f"({action_name})"
            )

            # -----------------------------------------
            # Execute action
            # -----------------------------------------

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

        # -------------------------------------------------
        # Episode finished
        # -------------------------------------------------

        env.robot.stop()

        episode_time = (
            time.time()
            - episode_start
        )

        total_steps += step_count
        total_rewards += total_reward
        total_time += episode_time

        if done:
            successes += 1

            successful_steps.append(
                step_count
            )

            result = "SUCCESS"

        else:
            failures += 1

            result = "FAIL"

        print()
        print(
            f"Episode {episode:02d} | "
            f"{result} | "
            f"Steps: {step_count} | "
            f"Reward: {total_reward:.2f} | "
            f"Time: {episode_time:.2f}s"
        )

    # -------------------------------------------------
    # Final results
    # -------------------------------------------------

    success_rate = (
        successes
        / args.episodes
        * 100.0
    )

    average_reward = (
        total_rewards
        / args.episodes
    )

    average_steps = (
        total_steps
        / args.episodes
    )

    average_time = (
        total_time
        / args.episodes
    )

    print()
    print(
        "===================================="
    )
    print(
        "EVALUATION RESULTS"
    )
    print(
        "===================================="
    )

    print(
        f"Total episodes: "
        f"{args.episodes}"
    )

    print(
        f"Successes: "
        f"{successes}"
    )

    print(
        f"Failures: "
        f"{failures}"
    )

    print(
        f"Success rate: "
        f"{success_rate:.1f}%"
    )

    print(
        f"Average reward: "
        f"{average_reward:.2f}"
    )

    print(
        f"Average steps: "
        f"{average_steps:.1f}"
    )

    if successful_steps:

        average_success_steps = (
            sum(successful_steps)
            / len(successful_steps)
        )

        best_success = min(
            successful_steps
        )

        worst_success = max(
            successful_steps
        )

        print(
            f"Average successful steps: "
            f"{average_success_steps:.1f}"
        )

        print(
            f"Best successful episode: "
            f"{best_success} steps"
        )

        print(
            f"Worst successful episode: "
            f"{worst_success} steps"
        )

    print(
        f"Average episode time: "
        f"{average_time:.2f}s"
    )

    print(
        "===================================="
    )

    env.robot.stop()


if __name__ == "__main__":
    main()