"""
Loading photon cubes, DCR info
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import taichi as ti
import torch
from einops import rearrange
from loguru import logger


@ti.kernel
def interpolate_W(photon_cube: ti.types.ndarray(), cfa_mask: ti.types.ndarray()):
    """
    (Bilinear) interpolates "W" in a RGBW pattern

    :param photon_cube:
    :param cfa_mask: 1 where R/G/B, 0 otherwise
    :return:
    """
    h, w, t = photon_cube.shape
    for i, j, t_index in ti.ndrange(h, w, t):
        if cfa_mask[i, j] != 0:
            counter = 0
            value = 0

            if 0 <= i - 1 <= h - 1:
                counter += 1
                value += photon_cube[i - 1, j, t_index]

            if 0 <= i + 1 <= h - 1:
                counter += 1
                value += photon_cube[i - 1, j, t_index]

            if 0 <= j - 1 <= w - 1:
                counter += 1
                value += photon_cube[i, j - 1, t_index]

            if 0 <= j + 1 <= w - 1:
                counter += 1
                value += photon_cube[i, j + 1, t_index]

            prob = value / counter
            photon_cube[i, j, t_index] = ti.random() < prob


def colorSPAD_correct_func(photon_cube: torch.Tensor, colorSPAD_RGBW_CFA):
    # Crop and rearrange
    photon_cube_cropped = torch.zeros(
        254,
        496,
        photon_cube.shape[-1],
        dtype=photon_cube.dtype,
        device=photon_cube.device,
    )
    photon_cube_cropped[:254, :496] = photon_cube[2:, :496]
    photon_cube_cropped[:254, 252:256] = photon_cube[2:, 260:264]
    photon_cube_cropped[:254, 260:264] = photon_cube[2:, 252:256]
    photon_cube = photon_cube_cropped
    if colorSPAD_RGBW_CFA:
        cfa = cv2.imread(colorSPAD_RGBW_CFA)[2:, :496, ::-1].copy().astype(np.float32)
        cfa = torch.from_numpy(cfa).to(photon_cube.device)
        mask = (cfa.mean(axis=-1) < 255).to(torch.uint8)
        interpolate_W(photon_cube, mask)
    return photon_cube


def get_photon_cube(
    file: Union[str, Path],
    initial_time_step: int = 0,
    num_time_step: int = 1000,
    rotate_180: bool = False,
    flip_ud: bool = False,
    colorSPAD_col_correct: bool = False,
    crop_sensor: bool = False,
    colorSPAD_RGBW_CFA: Path = None,
    device: torch.device = torch.device("cpu"),
    **kwargs,
):
    """
    Loads photon-cube as a torch Tensor

    :param file: root folder containing .bin files
    :param initial_time_step: (inclusive)
    :param num_time_step: Extract photon cube b.w
        [initial_time_step: initial_time_step + num_time_step]

    :param rotate_180: flip h, w
    :param flip_ud: flip h

    :param colorSPAD_col_correct: correct column swap, corrupted cols.
    :param colorSPAD_RGBW_CFA:
        used to interpolate W to R, G, B
    :param device: device to load photon cube on
    :param kwargs: for compatibility with cfg parsing

    :return: photon cube (height, width, time)
    """
    if isinstance(file, str):
        file = Path(file)

    final_time_step = initial_time_step + num_time_step
    logger.info(
        f"Loading photon cube from {file}, Time steps {initial_time_step}:{final_time_step}"
    )

    memmap_cube = np.load(file, mmap_mode="r")
    photon_cube = memmap_cube[initial_time_step:final_time_step]
    photon_cube = rearrange(np.unpackbits(photon_cube, axis=-1), "t h w -> h w t")
    del memmap_cube

    photon_cube = torch.from_numpy(photon_cube).to(device)

    # colorSPAD has column swap issues and pixels that are always 1.
    # See RGBW pattern here
    # https://drive.google.com/file/d/1FQHLtpQcO0vGwLDqNmfVdQQErBUXWhqh/view?usp=share_link
    if colorSPAD_col_correct:
        if device.type == "cuda":
            ti.init(arch=ti.gpu, kernel_profiler=True)
        else:
            ti.init(arch=ti.cpu, kernel_profiler=True)
        photon_cube = colorSPAD_correct_func(photon_cube, colorSPAD_RGBW_CFA)

    if crop_sensor:
        photon_cube = photon_cube[2:, :496]

    # Rotate 180
    if rotate_180:
        photon_cube = torch.flip(photon_cube, [0, 1])

    # Flip UD
    if flip_ud:
        photon_cube = torch.flip(photon_cube, [0])

    return photon_cube


def get_hot_pixel_mask(rotate_180: bool = False, flip_ud: bool = False, **kwargs):
    """
    :param rotate_180: flip h, w
    :param flip_ud: flip h
    :return: Hot pixel mask
    """
    file_extension = Path(kwargs["mask_path"]).suffix[1:]

    if file_extension == "npy":
        hot_pixel_mask = np.load(kwargs["mask_path"]).astype(np.uint8)
    else:
        raise NotImplementedError(f"File extension {file_extension} not supported.")

    if rotate_180:
        hot_pixel_mask = hot_pixel_mask[::-1, ::-1]

    if flip_ud:
        hot_pixel_mask = hot_pixel_mask[::-1]

    return hot_pixel_mask
