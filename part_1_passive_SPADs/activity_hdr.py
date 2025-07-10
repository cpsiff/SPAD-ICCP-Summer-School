import requests
from pathlib import Path
from loguru import logger
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def download_opencv_hdr_exposures(output_dir: Path | str) -> None:
    """
    Downloads and caches PNG images and list.txt from the OpenCV repository.

    :param output_dir: The directory where files will be saved.
    """
    api_url = "https://api.github.com/repos/opencv/opencv_extra/contents/testdata/cv/hdr/exposures?ref=4.x"
    output_path = Path(output_dir)
    logger.info(f"Fetching file list from: {api_url}")

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        files = response.json()
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensuring files are saved in '{output_path}/' directory...")

        for file_info in files:
            if file_info["type"] != "file":
                continue

            local_path = output_path / file_info["name"]
            is_target_file = (
                    local_path.suffix.lower() == ".png"
                    or local_path.name.lower() == "list.txt"
            )

            if is_target_file and not local_path.exists():
                logger.info(f"  -> Downloading {file_info['name']}...")
                file_response = requests.get(file_info["download_url"])
                file_response.raise_for_status()
                local_path.write_bytes(file_response.content)
            elif local_path.exists() and is_target_file:
                logger.info(f"'{local_path.name}' already cached. Skipping.")

        logger.success(f"Download and caching process complete for '{output_path}'.")

    except requests.exceptions.RequestException as e:
        logger.error(f"A network error occurred: {e}")


def create_hdr_image(
        image_dir: Path | str = "hdr_exposures",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Loads an exposure sequence and creates an HDR image.

    :param image_dir: Directory containing the images and list.txt.
    :return: A tuple containing the HDR image and exposure times, or (None, None) on failure.
    """
    logger.info(f"Starting HDR image creation from files in '{image_dir}'.")
    input_path = Path(image_dir)
    list_file = input_path / "list.txt"

    if not list_file.is_file():
        logger.error(f"Required 'list.txt' not found in '{input_path}'.")
        return None, None

    try:
        with open(list_file) as f:
            lines = f.readlines()

        images, times_list = [], []
        for i, line in enumerate(lines):
            tokens = line.split()
            if not tokens or not tokens[0].lower().endswith(".png"):
                continue

            img = cv.imread(str(input_path / tokens[0]))
            if img is not None:
                images.append(img)
                times_list.append(1 / float(tokens[1]))

        if len(images) < 2:
            logger.error("Not enough valid images found to create an HDR photo.")
            return None, None

        times = np.asarray(times_list, dtype=np.float32)
        logger.info(f"Loaded {len(images)} PNG images. Estimating CRF and merging...")

        calibrate = cv.createCalibrateDebevec()
        response = calibrate.process(images, times)
        merge_debevec = cv.createMergeDebevec()
        hdr = merge_debevec.process(images, times, response).clip(0, 1)

        logger.success("HDR image created and normalized successfully.")
        return hdr, times

    except (ValueError, IndexError) as e:
        logger.error(
            f"Failed to parse 'list.txt' on line {i + 1}: '{line.strip()}'. Please check the file format."
        )
        logger.debug(f"Details: {e}")
        return None, None


def plot_and_save_results(
        hdr_image: np.ndarray, exposure_times: np.ndarray, title: str, output_path: Path
):
    """
    Generates and saves a visualization of an HDR image, including tone-mapped,
    exposure-fused, and exposure bracketed LDR versions.

    :param hdr_image: The HDR image to visualize (values in [0, 1]).
    :param exposure_times: Array of exposure times used for bracketing.
    :param title: The main title for the plot.
    :param output_path: The path to save the output plot PDF.
    """
    logger.info(f"Generating plot: {title}")
    num_brackets = len(exposure_times)

    # --- Create LDR images for plotting ---
    # 1. Tone-mapped LDR
    tonemap = cv.createTonemapDrago(gamma=2.2)
    ldr_tonemapped = tonemap.process(hdr_image.astype(np.float32))

    # 2. Exposure Brackets (and collect for fusion)
    # This section is revised to produce more visually accurate LDR brackets
    # by applying auto-exposure scaling and gamma correction.
    ldr_brackets = []

    # Calculate a single scaling factor to normalize brightness for visualization
    mid_exposure_idx = len(exposure_times) // 2
    mid_exposure_time = exposure_times[mid_exposure_idx]
    mid_exposure_ldr_linear = hdr_image * mid_exposure_time

    # Use the median of non-zero pixels to find a robust scaling target
    median_val = np.median(mid_exposure_ldr_linear[mid_exposure_ldr_linear > 0])
    scaling_factor = 0.5 / median_val if median_val > 0 else 1.0

    for time in exposure_times:
        # Apply linear exposure scaling
        ldr_linear = hdr_image * time * scaling_factor
        # Apply gamma correction for display and convert to uint8
        ldr_gamma_corrected = np.clip(ldr_linear ** (1 / 2.2), 0, 1)
        ldr_bracket_uint8 = (ldr_gamma_corrected * 255).astype(np.uint8)
        ldr_brackets.append(ldr_bracket_uint8)

    # 3. Exposure-fused LDR
    merge_mertens = cv.createMergeMertens()
    ldr_fused = merge_mertens.process(ldr_brackets)

    # --- Plotting ---
    fig = plt.figure(figsize=(num_brackets * 4, 12))
    fig.suptitle(title, fontsize=20, y=0.98)

    # Plot tone-mapped image
    ax1 = plt.subplot2grid((3, num_brackets), (0, 0), colspan=num_brackets)
    ax1.imshow(cv.cvtColor(ldr_tonemapped, cv.COLOR_BGR2RGB))
    ax1.set_title("Tone-Mapped LDR (Drago)", fontsize=14)
    ax1.axis("off")

    # Plot exposure-fused image
    ax2 = plt.subplot2grid((3, num_brackets), (1, 0), colspan=num_brackets)
    ax2.imshow(cv.cvtColor(ldr_fused, cv.COLOR_BGR2RGB))
    ax2.set_title("Exposure-Fused LDR (Mertens)", fontsize=14)
    ax2.axis("off")

    # Plot exposure brackets
    for i, (img, time) in enumerate(zip(ldr_brackets, exposure_times)):
        ax = plt.subplot2grid((3, num_brackets), (2, i))
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(f"Exposure: {time:.4f}s")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")


def simulate_and_visualize_hdr_reconstruction(
        hdr_ground_truth: np.ndarray, times: np.ndarray, output_dir: Path | str
):
    """
    Simulates HDR reconstruction from binary frames and visualizes the results.

    :param hdr_ground_truth: The ground-truth HDR image.
    :param times: The original exposure times for visualization.
    :param output_dir: The directory to save visualization plots.
    """
    output_path = Path(output_dir)
    plot_and_save_results(
        hdr_ground_truth, times, "Ground Truth HDR", output_path / "0_ground_truth.pdf"
    )

    summed_binary_frames = np.zeros_like(hdr_ground_truth, dtype=np.float32)
    simulation_steps = [100, 1000, 10_000, 100_000] #, 1_000_000]

    current_n = 0
    for i, next_n in enumerate(simulation_steps):
        for _ in tqdm(
                range(next_n - current_n),
                desc=f"Simulating from N={current_n} to N={next_n}...",
        ):
            # Simulate a binary frame and add it to the accumulator
            binary_frame = (
                    np.random.uniform(size=hdr_ground_truth.shape) < hdr_ground_truth
            ).astype(np.float32)
            summed_binary_frames += binary_frame

        reconstructed_hdr = summed_binary_frames / next_n
        plot_and_save_results(
            reconstructed_hdr,
            times,
            f"Reconstructed HDR from N = {next_n:,} Frames",
            output_path / f"{i + 1}_reconstruction_N_{next_n}.pdf",
        )
        current_n = next_n

# TODO: HDR -> simulate -> bitpack and memmap
# func -> numpy file.
# scale
# sum, tonemapping, gamma 2.2., exposure fusion.
# matplotlib itself?
# noise blur
# 1. summing: HDR, noise blur
# 2. Summing with shifts, "motion projection": give them the RoI, vector, ask them to write the linear projection.
# Produce a motion sweep (mediapy show video). Find motion vectors yourself.
# (advanced / take-home description?/ remove.) parabolic.
# Implement the parabolic integration of at^2 + bt + c. Visualize.
# Deconvolve it using this PSF, this function.
# (backup) align-and-merge/pano
# (events?)

if __name__ == "__main__":
    data_directory = "part_1_passive_SPADs/data/exposure_sequence"

    download_opencv_hdr_exposures(output_dir=data_directory)
    hdr_image, exposure_times = create_hdr_image(image_dir=data_directory)

    if hdr_image is not None and exposure_times is not None:
        simulate_and_visualize_hdr_reconstruction(
            hdr_image, exposure_times, data_directory
        )
