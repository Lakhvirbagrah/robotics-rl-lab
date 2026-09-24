import numpy as np

from tasks.base_task import BaseTask


class ObjectCenteringTask:

    def __init__(
        self,
        center_target=0.5,
        tolerance=0.05
    ):
        self.center_target = center_target
        self.tolerance = tolerance

        self.previous_error = None

    def reset(self):
        self.previous_error = None

    def get_state_dim(self):
        return 2

    def get_actions(self):
        return [
            "turn_left",
            "turn_right",
            "stop"
        ]

    def get_action_dim(self):
        return len(
            self.get_actions()
        )

    def get_action(self, action_index):
        return self.get_actions()[
            action_index
        ]

    def get_state(self, observation):
        detected = observation[0]
        center_x = observation[1]

        return np.array(
            [
                center_x,
                detected
            ],
            dtype=np.float32
        )

    def compute_reward(self, observation):
        detected = observation[0]

        # Target lost
        if detected < 0.5:
            self.previous_error = None

            return -0.05

        center_x = observation[1]

        error = abs(
            center_x
            - self.center_target
        )

        # Strong success reward
        if error <= self.tolerance:
            self.previous_error = error

            return 5.0

        # First valid detection after reset
        if self.previous_error is None:
            self.previous_error = error

            # Small reward for simply
            # keeping the target visible
            return 0.05

        improvement = (
            self.previous_error
            - error
        )

        self.previous_error = error

        # Scale movement toward/away
        # from center so DQN gets
        # a stronger learning signal.
        reward = improvement * 10.0

        # Small visibility reward
        reward += 0.02

        return float(reward)

    def is_done(self, observation):
        detected = observation[0]

        if detected < 0.5:
            return False

        center_x = observation[1]

        error = abs(
            center_x
            - self.center_target
        )

        return (
            error <= self.tolerance
        )