from typing import Union

import numpy as np
import taichi as ti
import torch
from einops import repeat, rearrange


def motion_projection_from_photon_cube(photon_cube, raster_trajectory, hot_pixel_mask):
    """

    :param photon_cube: Shape (h, w, time)
    :param raster_trajectory: Shape (num_dir, time, 3)
        represents rasterized trajectory
    :param hot_pixel_mask: Shape (h, w). 1 represents defunct pixels
    :return: motion projection
        Shape (h, w, num_dir)
    """
    h, w, t = photon_cube.shape
    photon_cube = photon_cube.contiguous()
    num_dir = raster_trajectory.shape[0]
    assert raster_trajectory.shape[1] == t

    if photon_cube.device.type == "cuda":
        ti.init(arch=ti.gpu, kernel_profiler=True)
    else:
        ti.init(arch=ti.cpu, kernel_profiler=True)

    motion_projection = ti.field(float, shape=(h, w, num_dir))
    normalizer_array = ti.field(float, shape=(h, w, num_dir))
    hot_pixel_mask = np.ascontiguousarray(hot_pixel_mask)

    @ti.kernel
    def _motion_proj_kernel(
        photon_cube: ti.types.ndarray(),
        raster_trajectory: ti.types.ndarray(),
        hot_pixel_mask: ti.types.ndarray(),
    ):
        for i, j, dir_idx in ti.ndrange(h, w, num_dir):
            # Loop through time
            for t_index in range(t):
                i_shift = raster_trajectory[dir_idx, t_index, 0]
                j_shift = raster_trajectory[dir_idx, t_index, 1]

                i_shifted = i + i_shift
                j_shifted = j + j_shift

                # Check validity of shift
                if (
                    (0 <= i_shifted <= h - 1)
                    and (0 <= j_shifted <= w - 1)
                    and (hot_pixel_mask[i_shifted, j_shifted] == 0)
                ):
                    motion_projection[i, j, dir_idx] += photon_cube[
                        i_shifted, j_shifted, t_index
                    ]
                    normalizer_array[i, j, dir_idx] += 1

            # Normalize
            if normalizer_array[i, j, dir_idx] > 0.1 * t:
                motion_projection[i, j, dir_idx] /= normalizer_array[i, j, dir_idx]

    _motion_proj_kernel(photon_cube, raster_trajectory, hot_pixel_mask)

    return motion_projection.to_torch(device=photon_cube.device)


def parabola(t, velocity_max: float, t_max: int):
    """
    Function that returns x= f(t).
    f is a parabola, such that:
        x'(0) = - velocity_max
        x'(t_max) = velocity_max
        x'(t_max/2) = 0
        x(0) = 0
    :param t: time_step
    :param velocity_max: Velocity at t=0,t_max
    :param t_max: Duration of parabolic trajectory
    :return:
    """
    a = velocity_max / t_max
    b = -velocity_max
    x = a * t**2 + b * t - (a * t_max**2 / 3 + b * t_max / 2)
    return x


def get_parabolic_direction(
    t_max: int,
    direction_vector: torch.Tensor = torch.tensor([0, 1]),
    velocity_max: float = 30,
):
    """
    Computed via rounding floating point numbers.
    Motion along x-axis (for now)

    :param t_max: Time steps
    :param direction_vector: orientation vector
    :param velocity_max:  (num_directions) describe maximum velocities (at t=0, t=t_max)
    :return: line_voxel_ll (num_directions, t, 3)
    """
    device = direction_vector.device
    t_ll = torch.arange(t_max, dtype=int).to(device)
    direction_vector = direction_vector.reshape(1, 2)

    parabola_point_ll = parabola(t_ll, velocity_max, t_max)
    parabola_point_ll = parabola_point_ll.reshape(-1, 1)
    i_j_ll = (parabola_point_ll * direction_vector).int()
    t_ll = t_ll[:, np.newaxis]
    return torch.cat((i_j_ll, t_ll), dim=-1)


def get_linear_direction(
    t_max: int,
    direction_vector: torch.Tensor = torch.tensor([0, 1]),
    velocity_max: Union[float, torch.Tensor] = 1,
):
    """
    Computed via rounding floating point numbers.
    Motion along x-axis (for now)

    :param t_max: Time steps
    :param direction_vector: orientation vector
    :param velocity_max:  (num_directions) describe maximum velocities (at t=0, t=t_max)
    :return: line_voxel_ll (num_directions, t, 3)
    """
    device = direction_vector.device

    squeeze_result = isinstance(velocity_max, float)
    if squeeze_result:
        velocity_max = torch.tensor([velocity_max]).to(device)

    num_dir = velocity_max.shape[0]
    linear_dir = torch.zeros(num_dir, t_max, 3).to(device)
    linear_dir[..., -1] = repeat(
        torch.arange(t_max).to(device), "t -> num_dir t", num_dir=num_dir
    )

    linear_dir[..., :2] = (
        rearrange(
            torch.linspace(-t_max / 2, t_max / 2, steps=t_max, device=device),
            "t -> 1 t 1",
        )
        * rearrange(velocity_max, "num_dir -> num_dir 1 1").to(device)
        * rearrange(direction_vector, "c -> 1 1 c").to(device)
    )

    if squeeze_result:
        linear_dir = linear_dir.squeeze(0)

    return linear_dir.int()
