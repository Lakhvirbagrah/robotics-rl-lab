import numpy as np
from rl_env import TurtlebotEnv

print("Creating environment...")
env = TurtlebotEnv()

print("Resetting environment...")
state = env.reset()

print("Reset complete!")
print("Initial state:", state)

for i in range(20):
    action = np.random.randint(0, 4)

    print("Action:", action)

    next_state, reward, done = env.step(action)

    print("State:", next_state)
    print("Reward:", reward)
    print("Done:", done)

    if done:
        break
