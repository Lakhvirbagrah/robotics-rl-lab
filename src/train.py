#!/usr/bin/env python3

import argparse
import os
import pickle
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

    env = TurtlebotEnv(task)

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

    max_steps = config["training"]["max_steps_per_episode"]

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0.0
        done = False
        step_count = 0

        while (
            not done
            and step_count < max_steps
            and not rospy.is_shutdown()
        ):

            action = agent.select_action(state)

            next_state, reward, done = env.step(action)

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

            print(
                f"Episode {episode} | "
                f"Step: {step_count}/{max_steps} | "
                f"TotalReward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Reward: {reward:.3f}"
            )

        # --------------------------------------------------
        # Episode end reason
        # --------------------------------------------------

        if step_count >= max_steps and not done:
            print(
                f"Episode {episode} ended because maximum "
                f"step limit ({max_steps}) was reached."
            )

        elif done:
            print(
                f"Episode {episode} completed successfully "
                f"in {step_count} steps."
            )

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        with open(
            reward_log_path,
            "a"
        ) as f:
            f.write(
                f"{episode},{total_reward},{step_count},{done}\n"
            )
        agent.decay_epsilon()
        # --------------------------------------------------
        # Checkpoints
        # --------------------------------------------------

        if episode % checkpoint_interval == 0:

            agent.save(model_path)

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