class BaseTask:
    """
    Base interface for robotics reinforcement-learning tasks.

    A task defines:
    - how observations are converted into RL state
    - how reward is calculated
    - when an episode succeeds
    - when an episode terminates
    """

    def reset(self):
        pass

    def get_state(self, observation):
        raise NotImplementedError

    def compute_reward(self, observation):
        raise NotImplementedError

    def is_done(self, observation):
        raise NotImplementedError