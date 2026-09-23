import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dqn import DQN

class Agent:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.loss_fn = nn.MSELoss()

    def select_action(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(5)

        state = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state)

        return torch.argmax(q_values).item()

    def train(self, replay_buffer, batch_size=32):

        if replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        next_q_values = self.target_net(next_states).max(1)[0]

        target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)