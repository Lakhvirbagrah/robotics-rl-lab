from tasks.object_centering import ObjectCenteringTask


def create_task(config):
    task_name = config["task"]["name"]

    if task_name == "object_centering":
        return ObjectCenteringTask(
            center_target=config["task"]["center_target"],
            tolerance=config["task"]["tolerance"]
        )

    raise ValueError(f"Unknown task: {task_name}")