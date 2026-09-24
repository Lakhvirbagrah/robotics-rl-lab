import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim=5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class Agent:
    def __init__(
        self,
        state_dim,
        action_dim=5,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        lr=1e-3,
        target_update_interval=100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_update_interval = target_update_interval
        self.training_steps = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_net = DQN(
            state_dim,
            action_dim
        ).to(self.device)

        self.target_net = DQN(
            state_dim,
            action_dim
        ).to(self.device)

        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=lr
        )

        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(
                self.action_dim
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(
                state_tensor
            )

        return q_values.argmax(
            dim=1
        ).item()

    def train(
        self,
        replay_buffer,
        batch_size=32
    ):
        if len(replay_buffer) < batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = replay_buffer.sample(
            batch_size
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        q_values = self.q_net(
            states
        )

        current_q = q_values.gather(
            1,
            actions
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(
                next_states
            ).max(1)[0]

            target_q = rewards + (
                1 - dones
            ) * self.gamma * next_q

        loss = self.loss_fn(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.training_steps += 1

        if (
            self.training_steps
            % self.target_update_interval
            == 0
        ):
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def save(self, path):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "training_steps": self.training_steps
            },
            path
        )

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.q_net.load_state_dict(
            checkpoint["q_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon
        )

        self.training_steps = checkpoint.get(
            "training_steps",
            0
        )