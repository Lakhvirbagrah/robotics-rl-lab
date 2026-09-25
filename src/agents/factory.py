from agents.dqn import Agent


def create_agent(
    config,
    task
):
    algorithm_name = config[
        "algorithm"
    ]["name"].lower()

    if algorithm_name == "dqn":

        agent_config = config.get(
            "agent",
            {}
        )

        return Agent(
            state_dim=task.get_state_dim(),

            action_dim=task.get_action_dim(),

            gamma=agent_config.get(
                "gamma",
                0.99
            ),

            learning_rate=agent_config.get(
                "learning_rate",
                0.001
            ),

            epsilon_start=agent_config.get(
                "epsilon_start",
                1.0
            ),

            epsilon_min=agent_config.get(
                "epsilon_min",
                0.05
            ),

            epsilon_decay=agent_config.get(
                "epsilon_decay",
                0.995
            ),

            target_update_interval=agent_config.get(
                "target_update_interval",
                100
            )
        )

    raise ValueError(
        f"Unsupported algorithm: {algorithm_name}"
    )