from __future__ import annotations

import os

import cv2
import OpenEXR
import numpy as np
import numpy.typing as npt


def read_exr(path: str | os.PathLike) -> npt.NDArray:
    """Read an EXR file from disk and return it as a numpy array.

    Note:
        Imageio and cv2's cannot read an exr file when the data is stored in any
        other channel than RGB(A) but single channel EXRs often use the V channel.

    Args:
        path (str | os.PathLike): Path of EXR file.

    Returns:
        npt.NDArray: Array in HWC format
    """

    with OpenEXR.File(path) as f:
        if len(f.channels()) and list(f.channels().keys())[0] == "RGBA":
            return f.channels()["RGBA"].pixels
        if len(f.channels()) and list(f.channels().keys())[0] == "RGB":
            return f.channels()["RGB"].pixels
        return np.array([c.pixels for c in f.channels().values()])


def draw_rect_from_crop(
    im: npt.NDArray,
    crop: tuple[slice, ...],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 10,
) -> npt.NDArray:
    """Draw a bounding rectangle on an image based on a crop given as a tuple of slices.

    Warning:
        Image buffer will be modified.

    Args:
        im (npt.NDArray): Image on which to draw bounding rectangle, expects a uint8 numpy array in HWC format.
        crop (tuple[slice, ...]): Slices corresponding to the desired crop, often generated with `np.s_`.
        color (tuple[int, int, int], optional): Color of rectangle to draw. Defaults to red (255, 0, 0).
        thickness (int, optional): Width of rectangle lines in pixels. Defaults to 10.

    Returns:
        npt.NDArray: image with rectangle drawn onto it.
    """
    crop_w, crop_h, *_ = crop
    start = crop_h.start, crop_w.start
    stop = crop_h.stop, crop_w.stop
    return cv2.rectangle(im, start, stop, color, thickness)


class EmulatePhotonCube:
    def __init__(self, hdr_path: str | os.PathLike):
        """Emulate a photon cube from a static high dynamic range image.

        Args:
            hdr_path (str | os.PathLike): Path to the linear intensity image to sample from.
        """
        self.hdr = read_exr(hdr_path)[..., :3]
        self.detection_probability = 1.0 - np.exp(-self.hdr)

    def _sample_single_frame(self, seed: int = 123456) -> npt.NDArray:
        """Simulate the single photon camera imaging model by sampling
        from the underlying Bernoulli distribution (equiv. to uniform[0, 1] < p)

        Returns:
            npt.NDArray: Binary frame
        """
        # Set seed such that frames are reproducible
        np.random.seed(seed)
        bitplane = np.random.uniform(size=self.hdr.shape) < self.detection_probability
        return bitplane.astype(np.float32)

    def __getitem__(self, item: int | slice) -> npt.NDArray:
        """Get one or more binary frames.

        Args:
            item (int | slice): Index or range of bitplane to retrieve.

        Returns:
            npt.NDArray: Binary frame(s)
        """
        if isinstance(item, int):
            return self._sample_single_frame(seed=item)
        elif isinstance(item, slice):
            return np.array(
                [
                    self._sample_single_frame(seed=i)
                    for i in range(item.start, item.stop, item.step or 1)
                ]
            )
        raise IndexError(
            "Only simple indexing (using integers or a single slice) is allowed."
        )

    def sum_frames(self, start=0, end=100) -> npt.NDArray:
        """Sum many binary frames together, much faster than manually summing
        over frames (eg: sum(pcube[i] for i in range(10))).

        Note:
            The sum returned by this method and that of a manual sum
            will differ, but are statistically equivalent.

        Args:
            start (int, optional): Starting index. Defaults to 0.
            end (int, optional): Stopping index. Defaults to 100.

        Returns:
            npt.NDArray: Sum of binary frames
        """
        if end < start:
            raise ValueError(
                f"Cannot sum frames from {start} to {end}. Please use a valid range."
            )

        # Here we sample a Binomial distribution as it's equivalent to the sum of Bernoulli random variables.
        np.random.seed(hash((start, end)) % 10000000)
        return np.random.binomial(end - start, self.detection_probability)

