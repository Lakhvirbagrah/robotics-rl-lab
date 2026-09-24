#!/usr/bin/env python3

import argparse
import os
import pickle
import time
import yaml

import rospy

from environments.turtlebot_env import TurtlebotEnv
from agents.factory import create_agent
from rl_utils.replay_buffer import ReplayBuffer
from tasks.factory import create_task


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a robotics reinforcement-learning experiment"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/object_centering_dqn.yaml",
        help="Path to experiment configuration file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    rospy.init_node("robotics_rl_trainer")

    # --------------------------------------------------
    # Build experiment
    # --------------------------------------------------

    task = create_task(config)

    env = TurtlebotEnv(
    task,

    action_duration=config["training"].get(
        "action_duration",
        0.1
    ),

    reset_x=config["training"].get(
        "reset_x",
        -1.61982
    ),

    reset_y=config["training"].get(
        "reset_y",
        -2.0
    ),

    reset_yaw_base=config["training"].get(
        "reset_yaw_base",
        1.5708
    ),

    reset_yaw_jitter=config["training"].get(
        "reset_yaw_jitter",
        0.35
    )
)

    agent = create_agent(
        config,
        task
    )

    buffer = ReplayBuffer()

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    model_path = config["paths"]["model"]

    replay_buffer_path = config["paths"]["replay_buffer"]

    reward_log_path = config["paths"]["reward_log"]

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(replay_buffer_path),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(reward_log_path),
        exist_ok=True
    )

    # --------------------------------------------------
    # Load checkpoints
    # --------------------------------------------------

    if os.path.exists(model_path):
        agent.load(model_path)

        print(
            f"Loaded model checkpoint: {model_path}"
        )

    if os.path.exists(replay_buffer_path):
        with open(
            replay_buffer_path,
            "rb"
        ) as f:
            buffer.buffer = pickle.load(f)

        print(
            f"Loaded replay buffer: {replay_buffer_path}"
        )

    # --------------------------------------------------
    # Training settings
    # --------------------------------------------------

    episodes = config["training"]["episodes"]

    batch_size = config["training"]["batch_size"]

    checkpoint_interval = config["training"].get(
        "checkpoint_interval",
        10
    )

    max_steps = config["training"][
        "max_steps_per_episode"
    ]

    log_interval_steps = config[
        "training"
    ].get(
        "log_interval_steps",
        20
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0.0

        done = False

        step_count = 0

        episode_start_time = time.time()

        while (
            not done
            and step_count < max_steps
            and not rospy.is_shutdown()
        ):

            action = agent.select_action(
                state
            )

            next_state, reward, done = env.step(
                action
            )

            if next_state is None:
                continue

            buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.train(
                buffer,
                batch_size=batch_size
            )

            state = next_state

            total_reward += reward

            step_count += 1

            # ------------------------------------------
            # Reduced step logging
            # ------------------------------------------

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
                    f"Step {step_count}/{max_steps} | "
                    f"Reward {total_reward:.2f} | "
                    f"Epsilon {agent.epsilon:.3f} | "
                    f"Speed {steps_per_second:.2f} steps/s"
                )

        # --------------------------------------------------
        # Episode summary
        # --------------------------------------------------

        episode_time = (
            time.time()
            - episode_start_time
        )

        steps_per_second = (
            step_count / episode_time
            if episode_time > 0
            else 0.0
        )

        if (
            step_count >= max_steps
            and not done
        ):
            print(
                f"Episode {episode} ended because "
                f"maximum step limit "
                f"({max_steps}) was reached."
            )

        elif done:
            print(
                f"Episode {episode} completed "
                f"successfully."
            )

        print(
            f"Episode {episode} finished | "
            f"Steps: {step_count} | "
            f"Reward: {total_reward:.2f} | "
            f"Time: {episode_time:.2f}s | "
            f"Speed: {steps_per_second:.2f} steps/s"
        )

        # --------------------------------------------------
        # Episode-based epsilon decay
        # --------------------------------------------------

        agent.decay_epsilon()

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        with open(
            reward_log_path,
            "a"
        ) as f:
            f.write(
                f"{episode},"
                f"{total_reward},"
                f"{step_count},"
                f"{done},"
                f"{episode_time},"
                f"{steps_per_second}\n"
            )

        # --------------------------------------------------
        # Checkpoints
        # --------------------------------------------------

        if (
            episode
            % checkpoint_interval
            == 0
        ):

            agent.save(
                model_path
            )

            with open(
                replay_buffer_path,
                "wb"
            ) as f:
                pickle.dump(
                    buffer.buffer,
                    f
                )

    print("Training finished")


if __name__ == "__main__":
    main()