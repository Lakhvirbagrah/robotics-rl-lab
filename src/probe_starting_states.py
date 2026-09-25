#!/usr/bin/env python3

import argparse
import math

import rospy
import yaml

from environments.turtlebot_env import TurtlebotEnv
from tasks.factory import create_task


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def deg_to_rad(degrees):
    return degrees * math.pi / 180.0


def classify_center_x(center_x):
    if center_x < 0.40:
        return "LEFT"

    if center_x > 0.60:
        return "RIGHT"

    return "CENTER"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    rospy.init_node(
        "starting_state_probe"
    )

    task = create_task(
        config
    )

    env = TurtlebotEnv(
        task,
        action_duration=config[
            "training"
        ].get(
            "action_duration",
            0.1
        )
    )

    reset_x = -2.46246
    reset_y = -5.46862

    baseline_yaw = 1.5708

    yaw_offsets_degrees = [
        -45,
        -35,
        -30,
        -25,
        -20,
        -15,
        -10,
        -5,
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        45
    ]

    print()
    print(
        "=========================================="
    )
    print(
        "STARTING STATE PROBE"
    )
    print(
        "=========================================="
    )
    print()

    print(
        "Looking for:"
    )
    print(
        "LEFT   -> center_x < 0.40"
    )
    print(
        "CENTER -> 0.40 <= center_x <= 0.60"
    )
    print(
        "RIGHT  -> center_x > 0.60"
    )
    print()

    for yaw_offset_deg in yaw_offsets_degrees:

        test_yaw = (
            baseline_yaw
            + deg_to_rad(
                yaw_offset_deg
            )
        )

        state = env.reset(
            x=reset_x,
            y=reset_y,
            yaw=test_yaw
        )

        if state is None:
            print(
                f"{yaw_offset_deg:+5.1f}° | "
                f"NO STATE"
            )
            continue

        center_x = float(
            state[0]
        )

        detected = int(
            state[1]
        )

        if detected == 0:

            classification = (
                "NOT DETECTED"
            )

        else:

            classification = (
                classify_center_x(
                    center_x
                )
            )

        print(
            f"Yaw offset "
            f"{yaw_offset_deg:+5.1f}° | "
            f"Absolute yaw "
            f"{test_yaw:.4f} | "
            f"center_x "
            f"{center_x:.3f} | "
            f"detected "
            f"{detected} | "
            f"{classification}"
        )

    env.robot.stop()

    print()
    print(
        "=========================================="
    )
    print(
        "PROBE FINISHED"
    )
    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()