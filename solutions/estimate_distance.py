import numpy as np

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
    peak_idx = np.argmax(hist)
    time_of_flight = peak_idx * BIN_SIZE - TOF_OFFSET
    distance = SPEED_OF_LIGHT * time_of_flight / 2
    return distance * 1000  # convert m to mm
