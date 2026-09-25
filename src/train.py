#!/usr/bin/env python3

import argparse
import csv
import os
import pickle
import time

import mlflow
import psutil
import rospy
import yaml

from agents.factory import create_agent
from environments.turtlebot_env import TurtlebotEnv
from rl_utils.replay_buffer import ReplayBuffer
from tasks.factory import create_task


# ============================================================
# Optional NVIDIA monitoring
# ============================================================

try:
    import pynvml

    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# ============================================================
# System monitoring
# ============================================================

def get_system_metrics():
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
    }

    if not NVIDIA_AVAILABLE:
        return metrics

    try:
        pynvml.nvmlInit()

        handle = (
            pynvml.nvmlDeviceGetHandleByIndex(0)
        )

        temperature = (
            pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
        )

        utilization = (
            pynvml.nvmlDeviceGetUtilizationRates(
                handle
            )
        )

        memory = (
            pynvml.nvmlDeviceGetMemoryInfo(
                handle
            )
        )

        power_watts = (
            pynvml.nvmlDeviceGetPowerUsage(
                handle
            )
            / 1000.0
        )

        metrics.update(
            {
                "gpu_temperature_c": float(
                    temperature
                ),
                "gpu_utilization_percent": float(
                    utilization.gpu
                ),
                "gpu_memory_used_mb": float(
                    memory.used
                    / 1024
                    / 1024
                ),
                "gpu_memory_total_mb": float(
                    memory.total
                    / 1024
                    / 1024
                ),
                "gpu_power_watts": float(
                    power_watts
                ),
            }
        )

    except Exception as error:
        rospy.logwarn(
            f"GPU monitoring unavailable: {error}"
        )

    return metrics


# ============================================================
# Replay buffer
# ============================================================

def save_replay_buffer(
    replay_buffer,
    replay_path
):
    directory = os.path.dirname(
        replay_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        replay_path,
        "wb"
    ) as file:
        pickle.dump(
            replay_buffer,
            file
        )


def load_replay_buffer(
    replay_path,
    capacity=100000
):
    if os.path.exists(
        replay_path
    ):
        try:
            with open(
                replay_path,
                "rb"
            ) as file:
                loaded_buffer = pickle.load(
                    file
                )

            # Make sure this is the new reusable
            # ReplayBuffer and not an old deque.
            if hasattr(
                loaded_buffer,
                "push"
            ):
                print(
                    "Loaded replay buffer:",
                    len(loaded_buffer),
                    "transitions"
                )

                return loaded_buffer

            print(
                "Old replay buffer format "
                "detected. Starting fresh."
            )

        except Exception as error:
            print(
                "Could not load replay buffer:",
                error
            )

    return ReplayBuffer(
        capacity
    )


# ============================================================
# CSV logging
# ============================================================

def append_episode_log(
    log_path,
    episode,
    reward,
    steps,
    success,
    epsilon,
    average_loss,
    episode_time,
    steps_per_second
):
    directory = os.path.dirname(
        log_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    file_exists = os.path.exists(
        log_path
    )

    with open(
        log_path,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        if not file_exists:
            writer.writerow(
                [
                    "episode",
                    "reward",
                    "steps",
                    "success",
                    "epsilon",
                    "average_loss",
                    "episode_time",
                    "steps_per_second",
                ]
            )

        writer.writerow(
            [
                episode,
                reward,
                steps,
                int(success),
                epsilon,
                average_loss,
                episode_time,
                steps_per_second,
            ]
        )


# ============================================================
# MLflow helpers
# ============================================================

def setup_mlflow(config, project_root):

    tracking_uri = "http://127.0.0.1:5000"

    mlflow.set_tracking_uri(
        tracking_uri
    )

    print(
        "MLflow tracking URI:",
        mlflow.get_tracking_uri()
    )

    mlflow.set_experiment(
        config["experiment"]["name"]
    )


def log_mlflow_parameters(
    config,
    task,
    episodes,
    batch_size,
    max_steps
):
    training_config = config.get(
        "training",
        {}
    )

    agent_config = config.get(
        "agent",
        {}
    )

    algorithm_config = config.get(
        "algorithm",
        {}
    )

    parameters = {
        "algorithm": algorithm_config.get(
            "name",
            "unknown"
        ),
        "state_dim": task.get_state_dim(),
        "action_dim": task.get_action_dim(),
        "actions": str(
            task.get_actions()
        ),
        "episodes": episodes,
        "batch_size": batch_size,
        "max_steps_per_episode": max_steps,
        "action_duration": training_config.get(
            "action_duration",
            0.1
        ),
        "gamma": agent_config.get(
            "gamma",
            "not_set"
        ),
        "learning_rate": agent_config.get(
            "learning_rate",
            "not_set"
        ),
        "epsilon_start": agent_config.get(
            "epsilon_start",
            "not_set"
        ),
        "epsilon_min": agent_config.get(
            "epsilon_min",
            "not_set"
        ),
        "epsilon_decay": agent_config.get(
            "epsilon_decay",
            "not_set"
        ),
        "target_update_interval": (
            agent_config.get(
                "target_update_interval",
                "not_set"
            )
        ),
        "target_class": "sports ball",
    }

    mlflow.log_params(
        parameters
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML configuration"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    # Repository root:
    # first/
    #
    # train.py is:
    # first/src/train.py

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    # --------------------------------------------------------
    # ROS
    # --------------------------------------------------------

    rospy.init_node(
        "robotics_rl_trainer"
    )

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    experiment_name = config[
        "experiment"
    ]["name"]

    print()
    print(
        "===================================="
    )
    print(
        "ROBOTICS RL TRAINING"
    )
    print(
        "===================================="
    )

    print(
        "Experiment:",
        experiment_name
    )

    # --------------------------------------------------------
    # Task
    # --------------------------------------------------------

    task = create_task(
        config
    )

    print(
        "Task:",
        task.__class__.__name__
    )

    print(
        "State dimension:",
        task.get_state_dim()
    )

    print(
        "Action dimension:",
        task.get_action_dim()
    )

    print(
        "Actions:",
        task.get_actions()
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    action_duration = config[
        "training"
    ].get(
        "action_duration",
        0.1
    )

    env = TurtlebotEnv(
        task,
        action_duration=action_duration
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    agent = create_agent(
        config,
        task
    )

    print(
        "Agent:",
        agent.__class__.__name__
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    model_path = config[
        "paths"
    ]["model"]

    replay_path = config[
        "paths"
    ]["replay_buffer"]

    reward_log_path = config[
        "paths"
    ]["reward_log"]

    model_directory = os.path.dirname(
        model_path
    )

    if model_directory:
        os.makedirs(
            model_directory,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Replay buffer
    # --------------------------------------------------------

    replay_capacity = config.get(
        "replay_buffer",
        {}
    ).get(
        "capacity",
        100000
    )

    replay_buffer = load_replay_buffer(
        replay_path,
        replay_capacity
    )

    print(
        "Replay Buffer:",
        replay_buffer.__class__.__name__
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    if os.path.exists(
        model_path
    ):
        try:
            agent.load(
                model_path
            )

            print(
                "Loaded model checkpoint:",
                model_path
            )

        except Exception as error:
            print(
                "Could not load model checkpoint:",
                error
            )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    training_config = config[
        "training"
    ]

    episodes = training_config.get(
        "episodes",
        30
    )

    batch_size = training_config.get(
        "batch_size",
        32
    )

    max_steps_per_episode = (
        training_config.get(
            "max_steps_per_episode",
            200
        )
    )

    checkpoint_interval = (
        training_config.get(
            "checkpoint_interval",
            10
        )
    )

    log_interval_steps = (
        training_config.get(
            "log_interval_steps",
            20
        )
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    setup_mlflow(
        config,
        project_root
    )

    # Running success counter
    successful_episodes = 0

    training_start_time = (
        time.time()
    )

    # ========================================================
    # MLflow run
    # ========================================================

    with mlflow.start_run() as run:

        print()
        print(
            "MLflow Run ID:",
            run.info.run_id
        )

        print(
            "MLflow experiment:",
            experiment_name
        )

        print()

        log_mlflow_parameters(
            config,
            task,
            episodes,
            batch_size,
            max_steps_per_episode
        )

        # Save experiment configuration immediately
        if os.path.exists(
            args.config
        ):
            mlflow.log_artifact(
                args.config,
                artifact_path="config"
            )

        # ====================================================
        # Main episode loop
        # ====================================================

        for episode in range(
            episodes
        ):

            # ------------------------------------------------
            # IMPORTANT:
            # reset Gazebo every episode
            # ------------------------------------------------

            state = env.reset()

            if state is None:
                rospy.logwarn(
                    "Environment reset returned no state."
                )

                continue

            total_reward = 0.0
            step_count = 0
            done = False

            episode_losses = []

            episode_start_time = (
                time.time()
            )

            # ================================================
            # Step loop
            # ================================================

            while (
                not done
                and step_count
                < max_steps_per_episode
                and not rospy.is_shutdown()
            ):

                # --------------------------------------------
                # Action selection
                # --------------------------------------------

                action = agent.select_action(
                    state
                )

                # --------------------------------------------
                # Environment step
                # --------------------------------------------

                (
                    next_state,
                    reward,
                    done
                ) = env.step(
                    action
                )

                if next_state is None:
                    continue

                # --------------------------------------------
                # Replay memory
                # --------------------------------------------

                replay_buffer.push(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                # --------------------------------------------
                # DQN training
                # --------------------------------------------

                if (
                    len(replay_buffer)
                    >= batch_size
                ):
                    loss = agent.train(
                        replay_buffer,
                        batch_size
                    )

                    if loss is not None:
                        episode_losses.append(
                            float(loss)
                        )

                state = next_state

                total_reward += reward

                step_count += 1

                # --------------------------------------------
                # Terminal progress
                # --------------------------------------------

                if (
                    step_count
                    % log_interval_steps
                    == 0
                ):

                    elapsed = (
                        time.time()
                        - episode_start_time
                    )

                    speed = (
                        step_count
                        / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    print(
                        f"Episode {episode} | "
                        f"Step {step_count}/"
                        f"{max_steps_per_episode} | "
                        f"Reward {total_reward:.2f} | "
                        f"Epsilon {agent.epsilon:.3f} | "
                        f"Speed {speed:.2f} steps/s"
                    )

            # ================================================
            # Episode complete
            # ================================================

            env.robot.stop()

            episode_time = (
                time.time()
                - episode_start_time
            )

            steps_per_second = (
                step_count
                / episode_time
                if episode_time > 0
                else 0.0
            )

            average_loss = (
                sum(episode_losses)
                / len(episode_losses)
                if episode_losses
                else 0.0
            )

            # ------------------------------------------------
            # Success statistics
            # ------------------------------------------------

            if done:
                successful_episodes += 1

                print(
                    f"Episode {episode} "
                    f"completed successfully."
                )

            else:
                print(
                    f"Episode {episode} ended because "
                    f"maximum step limit "
                    f"({max_steps_per_episode}) "
                    f"was reached."
                )

            success_rate_running = (
                successful_episodes
                / (episode + 1)
                * 100.0
            )

            print(
                f"Episode {episode} finished | "
                f"Steps: {step_count} | "
                f"Reward: {total_reward:.2f} | "
                f"Time: {episode_time:.2f}s | "
                f"Speed: {steps_per_second:.2f} "
                f"steps/s"
            )

            # ------------------------------------------------
            # System metrics
            # ------------------------------------------------

            system_metrics = (
                get_system_metrics()
            )

            # ------------------------------------------------
            # MLflow metrics
            # ------------------------------------------------

            mlflow.log_metric(
                "reward",
                total_reward,
                step=episode
            )

            mlflow.log_metric(
                "steps",
                step_count,
                step=episode
            )

            mlflow.log_metric(
                "success",
                int(done),
                step=episode
            )

            mlflow.log_metric(
                "success_rate_running",
                success_rate_running,
                step=episode
            )

            mlflow.log_metric(
                "epsilon",
                agent.epsilon,
                step=episode
            )

            mlflow.log_metric(
                "average_loss",
                average_loss,
                step=episode
            )

            mlflow.log_metric(
                "episode_time_seconds",
                episode_time,
                step=episode
            )

            mlflow.log_metric(
                "steps_per_second",
                steps_per_second,
                step=episode
            )

            for (
                metric_name,
                metric_value
            ) in system_metrics.items():

                mlflow.log_metric(
                    metric_name,
                    metric_value,
                    step=episode
                )

            # ------------------------------------------------
            # CSV log
            # ------------------------------------------------

            append_episode_log(
                reward_log_path,
                episode,
                total_reward,
                step_count,
                done,
                agent.epsilon,
                average_loss,
                episode_time,
                steps_per_second
            )

            # ------------------------------------------------
            # Epsilon decay
            # ONCE per episode
            # ------------------------------------------------

            agent.decay_epsilon()

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                (episode + 1)
                % checkpoint_interval
                == 0
            ):

                agent.save(
                    model_path
                )

                save_replay_buffer(
                    replay_buffer,
                    replay_path
                )

                print(
                    f"Checkpoint saved after "
                    f"episode {episode + 1}"
                )

        # ====================================================
        # Training complete
        # ====================================================

        env.robot.stop()

        total_training_time = (
            time.time()
            - training_start_time
        )

        # ----------------------------------------------------
        # Final model
        # ----------------------------------------------------

        agent.save(
            model_path
        )

        save_replay_buffer(
            replay_buffer,
            replay_path
        )

        # ----------------------------------------------------
        # Final metrics
        # ----------------------------------------------------

        final_success_rate = (
            successful_episodes
            / episodes
            * 100.0
            if episodes > 0
            else 0.0
        )

        mlflow.log_metric(
            "final_success_rate",
            final_success_rate
        )

        mlflow.log_metric(
            "total_training_time_seconds",
            total_training_time
        )

        # ----------------------------------------------------
        # Artifacts
        # ----------------------------------------------------

        if os.path.exists(
            model_path
        ):
            mlflow.log_artifact(
                model_path,
                artifact_path="models"
            )

        if os.path.exists(
            replay_path
        ):
            mlflow.log_artifact(
                replay_path,
                artifact_path="replay_buffer"
            )

        if os.path.exists(
            reward_log_path
        ):
            mlflow.log_artifact(
                reward_log_path,
                artifact_path="logs"
            )

        print()
        print(
            "===================================="
        )

        print(
            "TRAINING FINISHED"
        )

        print(
            f"Successes: "
            f"{successful_episodes}/"
            f"{episodes}"
        )

        print(
            f"Final success rate: "
            f"{final_success_rate:.1f}%"
        )

        print(
            f"Training time: "
            f"{total_training_time:.1f}s"
        )

        print(
            "MLflow Run ID:",
            run.info.run_id
        )

        print(
            "===================================="
        )


if __name__ == "__main__":
    main()