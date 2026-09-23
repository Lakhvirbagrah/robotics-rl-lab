import random
import pickle
from collections import deque
import os


class ReplayBuffer:

    def __init__(self, capacity=10000):

        self.buffer = deque(maxlen=capacity)

    # ---------------- STORE EXPERIENCE ---------------- #

    def push(self, state, action, reward, next_state, done):

        self.buffer.append((
            state,
            action,
            reward,
            next_state,
            done
        ))

    # ---------------- SAMPLE MINI-BATCH ---------------- #

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones

    # ---------------- SIZE ---------------- #

    def __len__(self):

        return len(self.buffer)

    # ---------------- SAVE BUFFER ---------------- #

    def save(self, path="memory/replay_buffer.pkl"):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

        print(f"[ReplayBuffer] Saved to {path}")

    # ---------------- LOAD BUFFER ---------------- #

    def load(self, path="memory/replay_buffer.pkl"):

        if os.path.exists(path):

            with open(path, "rb") as f:
                self.buffer = pickle.load(f)

            print(f"[ReplayBuffer] Loaded from {path}")

        else:

            print("[ReplayBuffer] No saved buffer found, starting fresh")