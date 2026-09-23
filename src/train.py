#!/usr/bin/env python3

import rospy
import numpy as np
import os
import pickle

from rl_env import TurtlebotEnv
from dqn import Agent
from replay_buffer import ReplayBuffer


def main():

    rospy.init_node("dqn_trainer")

    env = TurtlebotEnv()
    agent = Agent()
    buffer = ReplayBuffer()

    # ---------------- LOAD CHECKPOINTS ---------------- #

    if os.path.exists("models/dqn_model.pth"):
        agent.load("models/dqn_model.pth")

    if os.path.exists("memory/replay_buffer.pkl"):
        with open("memory/replay_buffer.pkl", "rb") as f:
            buffer.buffer = pickle.load(f)
        print("Replay buffer loaded")

    reward_log = []

    episodes = 500

    for ep in range(episodes):

        state = env.reset()
        state = np.array(state[:2])

        total_reward = 0
        done = False

        while not done and not rospy.is_shutdown():

            action = agent.select_action(state)

            next_state, reward, done = env.step(action)

            if next_state is None:
                continue

            next_state = np.array(next_state[:2])

            buffer.push(state, action, reward, next_state, done)

            agent.train(buffer)

            state = next_state
            total_reward += reward

        # ---------------- LOGGING ---------------- #

            print(f"Episode {ep} | TotalReward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}, Reward {reward}")

        reward_log.append(total_reward)

        with open("logs/rewards.txt", "a") as f:
            f.write(f"{ep},{total_reward}\n")

        # ---------------- SAVE CHECKPOINTS ---------------- #

        if ep % 10 == 0:

            agent.save("models/dqn_model.pth")

            with open("memory/replay_buffer.pkl", "wb") as f:
                pickle.dump(buffer.buffer, f)

    print("Training finished")


if __name__ == "__main__":
    main()