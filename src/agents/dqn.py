#!/usr/bin/env python3

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# DQN Neural Network
# ============================================================

class DQN(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                state_dim,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                action_dim
            )
        )

    def forward(
        self,
        state
    ):
        return self.network(
            state
        )


# ============================================================
# DQN Agent
# ============================================================

class Agent:

    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        target_update_interval=100
    ):

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        self.state_dim = state_dim
        self.action_dim = action_dim

        # ----------------------------------------------------
        # Hyperparameters
        # ----------------------------------------------------

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_update_interval = (
            target_update_interval
        )

        # Number of gradient updates performed
        self.training_steps = 0

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "DQN device:",
            self.device
        )

        if torch.cuda.is_available():

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        # ----------------------------------------------------
        # Networks
        # ----------------------------------------------------

        self.policy_net = DQN(
            state_dim,
            action_dim
        ).to(
            self.device
        )

        self.target_net = DQN(
            state_dim,
            action_dim
        ).to(
            self.device
        )

        # Start with identical networks
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        # Target network is never directly trained
        self.target_net.eval()

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        self.loss_function = (
            nn.SmoothL1Loss()
        )

    # ========================================================
    # Action Selection
    # ========================================================

    def select_action(
        self,
        state
    ):

        # ----------------------------------------------------
        # Exploration
        # ----------------------------------------------------

        if random.random() < self.epsilon:

            return random.randrange(
                self.action_dim
            )

        # ----------------------------------------------------
        # Exploitation
        # ----------------------------------------------------

        state_array = np.asarray(
            state,
            dtype=np.float32
        )

        state_tensor = (
            torch.from_numpy(
                state_array
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():

            q_values = self.policy_net(
                state_tensor
            )

            action = (
                q_values
                .argmax(
                    dim=1
                )
                .item()
            )

        return action

    # ========================================================
    # DQN Training
    # ========================================================

    def train(
        self,
        replay_buffer,
        batch_size
    ):

        # ----------------------------------------------------
        # Not enough samples yet
        # ----------------------------------------------------

        if len(
            replay_buffer
        ) < batch_size:

            return None

        # ----------------------------------------------------
        # Sample replay buffer
        # ----------------------------------------------------

        batch = replay_buffer.sample(
            batch_size
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = batch

        # ====================================================
        # IMPORTANT PERFORMANCE FIX
        #
        # Instead of:
        #
        # torch.tensor(list_of_numpy_arrays)
        #
        # we first build one contiguous NumPy array.
        # ====================================================

        states_np = np.asarray(
            states,
            dtype=np.float32
        )

        next_states_np = np.asarray(
            next_states,
            dtype=np.float32
        )

        actions_np = np.asarray(
            actions,
            dtype=np.int64
        )

        rewards_np = np.asarray(
            rewards,
            dtype=np.float32
        )

        dones_np = np.asarray(
            dones,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        states_tensor = (
            torch.from_numpy(
                states_np
            )
            .to(self.device)
        )

        next_states_tensor = (
            torch.from_numpy(
                next_states_np
            )
            .to(self.device)
        )

        actions_tensor = (
            torch.from_numpy(
                actions_np
            )
            .long()
            .to(self.device)
        )

        rewards_tensor = (
            torch.from_numpy(
                rewards_np
            )
            .to(self.device)
        )

        dones_tensor = (
            torch.from_numpy(
                dones_np
            )
            .to(self.device)
        )

        # ====================================================
        # Current Q-values
        # ====================================================

        q_values = self.policy_net(
            states_tensor
        )

        current_q_values = (
            q_values
            .gather(
                1,
                actions_tensor.unsqueeze(1)
            )
            .squeeze(1)
        )

        # ====================================================
        # Target Q-values
        # ====================================================

        with torch.no_grad():

            next_q_values = (
                self.target_net(
                    next_states_tensor
                )
                .max(
                    dim=1
                )[0]
            )

            target_q_values = (
                rewards_tensor
                +
                self.gamma
                *
                next_q_values
                *
                (
                    1.0
                    - dones_tensor
                )
            )

        # ====================================================
        # Loss
        # ====================================================

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # ====================================================
        # Backpropagation
        # ====================================================

        self.optimizer.zero_grad()

        loss.backward()

        # Protect against extreme gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        # ----------------------------------------------------
        # Training-step counter
        # ----------------------------------------------------

        self.training_steps += 1

        # ----------------------------------------------------
        # Target network update
        # ----------------------------------------------------

        if (
            self.training_steps
            % self.target_update_interval
            == 0
        ):

            self.update_target_network()

        return loss.item()

    # ========================================================
    # Target Network
    # ========================================================

    def update_target_network(
        self
    ):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

    # ========================================================
    # Epsilon Decay
    # ========================================================

    def decay_epsilon(
        self
    ):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay
        )

    # ========================================================
    # Save Checkpoint
    # ========================================================

    def save(
        self,
        model_path
    ):

        directory = os.path.dirname(
            model_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        checkpoint = {

            "policy_net_state_dict":
                self.policy_net.state_dict(),

            "target_net_state_dict":
                self.target_net.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "epsilon":
                self.epsilon,

            "training_steps":
                self.training_steps,

            "state_dim":
                self.state_dim,

            "action_dim":
                self.action_dim,
        }

        torch.save(
            checkpoint,
            model_path
        )

    # ========================================================
    # Load Checkpoint
    # ========================================================

    def load(
        self,
        model_path
    ):

        if not os.path.exists(
            model_path
        ):

            print(
                "Model checkpoint not found:",
                model_path
            )

            return False

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # ----------------------------------------------------
        # New checkpoint format
        # ----------------------------------------------------

        if (
            isinstance(
                checkpoint,
                dict
            )
            and
            "policy_net_state_dict"
            in checkpoint
        ):

            self.policy_net.load_state_dict(
                checkpoint[
                    "policy_net_state_dict"
                ]
            )

            if (
                "target_net_state_dict"
                in checkpoint
            ):

                self.target_net.load_state_dict(
                    checkpoint[
                        "target_net_state_dict"
                    ]
                )

            else:

                self.target_net.load_state_dict(
                    checkpoint[
                        "policy_net_state_dict"
                    ]
                )

            if (
                "optimizer_state_dict"
                in checkpoint
            ):

                try:

                    self.optimizer.load_state_dict(
                        checkpoint[
                            "optimizer_state_dict"
                        ]
                    )

                    # Move optimizer state tensors
                    # to the current device.
                    for optimizer_state in (
                        self.optimizer.state.values()
                    ):

                        for key, value in (
                            optimizer_state.items()
                        ):

                            if torch.is_tensor(
                                value
                            ):

                                optimizer_state[
                                    key
                                ] = value.to(
                                    self.device
                                )

                except Exception as error:

                    print(
                        "Optimizer state was not loaded:",
                        error
                    )

            self.epsilon = checkpoint.get(
                "epsilon",
                self.epsilon
            )

            self.training_steps = (
                checkpoint.get(
                    "training_steps",
                    0
                )
            )

        # ----------------------------------------------------
        # Compatibility with old plain state_dict checkpoints
        # ----------------------------------------------------

        else:

            self.policy_net.load_state_dict(
                checkpoint
            )

            self.target_net.load_state_dict(
                checkpoint
            )

        self.policy_net.to(
            self.device
        )

        self.target_net.to(
            self.device
        )

        self.target_net.eval()

        print(
            "Loaded DQN checkpoint:",
            model_path
        )

        print(
            "Loaded epsilon:",
            round(
                self.epsilon,
                4
            )
        )

        print(
            "Loaded training steps:",
            self.training_steps
        )

        return True