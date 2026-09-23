from agents.dqn import Agent


def create_agent(config, task):
    algorithm_name = config["algorithm"]["name"]

    if algorithm_name == "dqn":
        return Agent(
            state_dim=task.get_state_dim(),
            action_dim=task.get_action_dim(),
            gamma=config["agent"]["gamma"],
            epsilon=config["agent"]["epsilon_start"],
            epsilon_min=config["agent"]["epsilon_min"],
            epsilon_decay=config["agent"]["epsilon_decay"],
            lr=config["agent"]["learning_rate"]
        )

    raise ValueError(f"Unknown algorithm: {algorithm_name}")
