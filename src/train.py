#!/usr/bin/env python3

import rospy
import numpy as np
import os
import pickle
import yaml
from environments.turtlebot_env import TurtlebotEnv
from agents.dqn import Agent
from replay_buffer import ReplayBuffer
from tasks.object_centering import ObjectCenteringTask

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():

    rospy.init_node("dqn_trainer")
    config = load_config("configs/object_centering_dqn.yaml")

    task = ObjectCenteringTask(
    center_target=config["task"]["center_target"],
    tolerance=config["task"]["tolerance"]
)
    env = TurtlebotEnv(task)
    agent = Agent(
    state_dim=task.get_state_dim(),
    action_dim=task.get_action_dim(),
    gamma=config["agent"]["gamma"],
    epsilon=config["agent"]["epsilon_start"],
    epsilon_min=config["agent"]["epsilon_min"],
    epsilon_decay=config["agent"]["epsilon_decay"],
    lr=config["agent"]["learning_rate"]
)
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