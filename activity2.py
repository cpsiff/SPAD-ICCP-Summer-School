"""
Estimate distance to nearest object in a transient histogram.

See activity2.md for details.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

SPEED_OF_LIGHT = 299792458  # in meters per second
BIN_SIZE = 9.1e-11  # in seconds
# The TMF sensor seems to start looking for photons a bit before the laser pulse is fired.
# This offset accounts for the delay.
TOF_OFFSET = 1.1e-9  # in seconds


def estimate_distance(hist: np.ndarray) -> float:
    """
    Estimate the distance to the nearest object in a transient histogram from the TMF8820 sensor.

    Args:
        hist: (128,) numpy array of histogram counts

    Returns:
        distance: float, estimated distance to the nearest object in mm
    """
    return 0.0 # TODO: Implement your algorithm here


def eval_dist_on_dataset():
    with open("data/distance_data.json", "r") as f:
        data = json.load(f)

    true_dists = [datapoint["dist_to_plane"] * 1000 for datapoint in data]
    onboard_dists = [datapoint["distances"][0]["depths_1"][4] for datapoint in data]
    est_dists = [estimate_distance(datapoint["hists"][4]) for datapoint in data]

    plt.scatter(true_dists, onboard_dists, label="Onboard Distance")
    plt.scatter(true_dists, est_dists, label="Estimated Distance")
    max_dist = max(true_dists)
    plt.plot([0, max_dist], [0, max_dist], "k--", label="Ideal (slope=1)")
    plt.xlabel("True Distance (mm)")
    plt.ylabel("Your Algorithm (mm)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    eval_dist_on_dataset()
