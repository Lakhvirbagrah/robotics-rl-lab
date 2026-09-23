#!/usr/bin/env python3

import rospy
import numpy as np
import os
import pickle
import yaml
from environments.turtlebot_env import TurtlebotEnv
from agents.factory import create_agent
from rl_utils.replay_buffer import ReplayBuffer
from tasks.factory import create_task

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():

    rospy.init_node("dqn_trainer")
    config = load_config("configs/object_centering_dqn.yaml")

    task = create_task(config)
    env = TurtlebotEnv(task)
    agent = create_agent(config, task)
    # ---------------- LOAD CHECKPOINTS ---------------- #

    if os.path.exists("models/dqn_model.pth"):
        agent.load("models/dqn_model.pth")

    if os.path.exists("memory/replay_buffer.pkl"):
        with open("memory/replay_buffer.pkl", "rb") as f:
            buffer.buffer = pickle.load(f)
        print("Replay buffer loaded")

    reward_log = []

    episodes = config["training"]["episodes"]

    for ep in range(episodes):

        state = env.reset()
        state = np.array(state)

        total_reward = 0
        done = False

        while not done and not rospy.is_shutdown():

            action = agent.select_action(state)

            next_state, reward, done = env.step(action)

            if next_state is None:
                continue

            next_state = np.array(next_state)

            buffer.push(state, action, reward, next_state, done)

            agent.train(buffer,batch_size=config["training"]["batch_size"])

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