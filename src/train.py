#!/usr/bin/env python3

import argparse
import csv
import os
import pickle
import time

import rospy
import yaml

from agents.factory import create_agent
from environments.turtlebot_env import TurtlebotEnv
from rl_utils.replay_buffer import ReplayBuffer
from tasks.factory import create_task


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def save_replay_buffer(
    replay_buffer,
    replay_path
):
    os.makedirs(
        os.path.dirname(replay_path),
        exist_ok=True
    )

    with open(
        replay_path,
        "wb"
    ) as file:
        pickle.dump(
            replay_buffer,
            file
        )


def load_replay_buffer(
    replay_path,
    default_capacity=100000
):
    if os.path.exists(replay_path):
        try:
            with open(
                replay_path,
                "rb"
            ) as file:
                replay_buffer = pickle.load(
                    file
                )

            print(
                f"Loaded replay buffer: "
                f"{len(replay_buffer)} transitions"
            )

            return replay_buffer

        except Exception as error:
            print(
                "Could not load replay buffer:",
                error
            )

    return ReplayBuffer(
        default_capacity
    )


def append_episode_log(
    log_path,
    episode,
    total_reward,
    step_count,
    success,
    episode_time,
    steps_per_second,
    epsilon
):
    os.makedirs(
        os.path.dirname(log_path),
        exist_ok=True
    )

    file_exists = os.path.exists(
        log_path
    )

    with open(
        log_path,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        if not file_exists:
            writer.writerow(
                [
                    "episode",
                    "reward",
                    "steps",
                    "success",
                    "episode_time",
                    "steps_per_second",
                    "epsilon"
                ]
            )

        writer.writerow(
            [
                episode,
                total_reward,
                step_count,
                int(success),
                episode_time,
                steps_per_second,
                epsilon
            ]
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML configuration"
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Load configuration
    # -------------------------------------------------

    config = load_config(
        args.config
    )

    rospy.init_node(
        "robotics_rl_trainer"
    )

    print()
    print(
        "Experiment:",
        config["experiment"]["name"]
    )

    # -------------------------------------------------
    # Create task
    # -------------------------------------------------

    task = create_task(
        config
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
    # Create DQN agent
    # -------------------------------------------------

    agent = create_agent(
    config,
    task)

    print(
        "Agent:",
        agent.__class__.__name__
    )

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------

    model_path = config[
        "paths"
    ]["model"]

    replay_path = config[
        "paths"
    ]["replay_buffer"]

    reward_log_path = config[
        "paths"
    ]["reward_log"]

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    # -------------------------------------------------
    # Replay buffer
    # -------------------------------------------------

    replay_capacity = config.get(
        "replay_buffer",
        {}
    ).get(
        "capacity",
        100000
    )

    replay_buffer = load_replay_buffer(
        replay_path,
        replay_capacity
    )

    print(
        "Replay Buffer:",
        replay_buffer.__class__.__name__
    )

    # -------------------------------------------------
    # Load existing checkpoint
    # -------------------------------------------------

    if os.path.exists(
        model_path
    ):
        try:
            agent.load(
                model_path
            )

            print(
                "Loaded model checkpoint:",
                model_path
            )

        except Exception as error:
            print(
                "Could not load checkpoint:",
                error
            )

    # -------------------------------------------------
    # Training configuration
    # -------------------------------------------------

    episodes = config[
        "training"
    ].get(
        "episodes",
        30
    )

    batch_size = config[
        "training"
    ].get(
        "batch_size",
        32
    )

    max_steps_per_episode = config[
        "training"
    ].get(
        "max_steps_per_episode",
        200
    )

    checkpoint_interval = config[
        "training"
    ].get(
        "checkpoint_interval",
        10
    )

    log_interval_steps = config[
        "training"
    ].get(
        "log_interval_steps",
        20
    )

    # -------------------------------------------------
    # Main training loop
    # -------------------------------------------------

    for episode in range(
        episodes
    ):

        # =============================================
        # IMPORTANT:
        # RESET GAZEBO AT THE START OF EVERY EPISODE
        # =============================================

        state = env.reset()

        if state is None:
            rospy.logwarn(
                "Environment reset returned no state."
            )

            continue

        total_reward = 0.0
        step_count = 0
        done = False

        episode_start_time = (
            time.time()
        )

        # ---------------------------------------------
        # Episode loop
        # ---------------------------------------------

        while (
            not done
            and
            step_count
            < max_steps_per_episode
            and
            not rospy.is_shutdown()
        ):

            # Select action
            action = agent.select_action(
                state
            )

            # Execute action
            (
                next_state,
                reward,
                done
            ) = env.step(
                action
            )

            if next_state is None:
                continue

            # Store transition
            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            # Train DQN
            if (
                len(replay_buffer)
                >= batch_size
            ):
                agent.train(
                    replay_buffer,
                    batch_size
                )

            state = next_state

            total_reward += reward

            step_count += 1

            # -----------------------------------------
            # Progress output
            # -----------------------------------------

            if (
                step_count
                % log_interval_steps
                == 0
            ):

                elapsed = (
                    time.time()
                    - episode_start_time
                )

                steps_per_second = (
                    step_count / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"Episode {episode} | "
                    f"Step {step_count}/"
                    f"{max_steps_per_episode} | "
                    f"Reward "
                    f"{total_reward:.2f} | "
                    f"Epsilon "
                    f"{agent.epsilon:.3f} | "
                    f"Speed "
                    f"{steps_per_second:.2f} "
                    f"steps/s"
                )

        # -------------------------------------------------
        # Episode finished
        # -------------------------------------------------

        env.robot.stop()

        episode_time = (
            time.time()
            - episode_start_time
        )

        steps_per_second = (
            step_count / episode_time
            if episode_time > 0
            else 0.0
        )

        if done:

            print(
                f"Episode {episode} "
                f"completed successfully."
            )

        else:

            print(
                f"Episode {episode} ended "
                f"because maximum step limit "
                f"({max_steps_per_episode}) "
                f"was reached."
            )

        print(
            f"Episode {episode} finished | "
            f"Steps: {step_count} | "
            f"Reward: {total_reward:.2f} | "
            f"Time: {episode_time:.2f}s | "
            f"Speed: "
            f"{steps_per_second:.2f} "
            f"steps/s"
        )

        # -------------------------------------------------
        # Save episode results
        # -------------------------------------------------

        append_episode_log(
            reward_log_path,
            episode,
            total_reward,
            step_count,
            done,
            episode_time,
            steps_per_second,
            agent.epsilon
        )

        # -------------------------------------------------
        # Decay epsilon ONCE PER EPISODE
        # -------------------------------------------------

        agent.decay_epsilon()

        # -------------------------------------------------
        # Save checkpoint
        # -------------------------------------------------

        if (
            (episode + 1)
            % checkpoint_interval
            == 0
        ):

            agent.save(
                model_path
            )

            save_replay_buffer(
                replay_buffer,
                replay_path
            )

            print(
                f"Checkpoint saved "
                f"after episode "
                f"{episode + 1}"
            )

    # -------------------------------------------------
    # Final save
    # -------------------------------------------------

    env.robot.stop()

    agent.save(
        model_path
    )

    save_replay_buffer(
        replay_buffer,
        replay_path
    )

    print(
        "Training finished"
    )


if __name__ == "__main__":
    main()