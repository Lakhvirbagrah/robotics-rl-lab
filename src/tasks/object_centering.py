import numpy as np

from tasks.base_task import BaseTask


class ObjectCenteringTask(BaseTask):

    def __init__(self, center_target=0.5, tolerance=0.05):
        self.center_target = center_target
        self.tolerance = tolerance
        self.previous_error = None

    def reset(self):
        self.previous_error = None

    def get_state_dim(self):
        return 1

    def get_actions(self):
        return [
            "turn_left",
            "turn_right",
            "stop"
        ]

    def get_action_dim(self):
        return len(self.get_actions())

    def get_action(self, action_index):
        return self.get_actions()[action_index]

    def get_state(self, observation):
        center_x = observation[0]

        return np.array(
            [center_x],
            dtype=np.float32
        )

    def compute_reward(self, observation):
        center_x = observation[0]

        error = abs(center_x - self.center_target)

        if self.previous_error is None:
            self.previous_error = error
            return 0.0

        reward = self.previous_error - error

        self.previous_error = error

        if error <= self.tolerance:
            reward += 1.0

        return reward

    def is_done(self, observation):
        center_x = observation[0]

        error = abs(center_x - self.center_target)

        return error <= self.tolerance