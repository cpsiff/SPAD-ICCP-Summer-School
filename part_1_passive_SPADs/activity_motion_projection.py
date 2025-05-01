"""
Run as: streamlit run activity_motion_projection.py
"""
from pathlib import Path

import numpy as np
import streamlit as st
import torch.cuda
from einops import repeat, rearrange
from loguru import logger
from omegaconf import OmegaConf
from streamlit_image_coordinates import streamlit_image_coordinates
from torchvision.io import write_video
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

from data import get_photon_cube, get_hot_pixel_mask
from motion_projection import (
    get_linear_direction,
    motion_projection_from_photon_cube,
)
from ops import nearest_neighbor_inpaint
from utils import setup_camera_form, setup_sidebar


def _enable_motion_projection():
    st.session_state.run_motion_projection = True


def _clear_state():
    st.session_state.run_motion_projection = False


@st.cache_resource
def load_raft():
    # RAFT
    logger.info("Loading RAFT")
    raft_weights = Raft_Large_Weights.DEFAULT
    raft_model = raft_large(weights=raft_weights)
    raft_transforms = raft_weights.transforms()
    return raft_model, raft_transforms


def main():
    st.header("Motion Projections Using Optical Flow")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    raft_model, raft_transforms = load_raft()

    for key in ["run_motion_projection"]:
        if key not in st.session_state:
            setattr(st.session_state, key, False)

    # Load scene config
    scene_cfg = OmegaConf.load("scenes/falling_die.yaml")
    motion_projection_form_cfg = OmegaConf.load("motion_projection.yaml")

    print(OmegaConf.to_yaml(scene_cfg))
    scene_kwargs = setup_sidebar(scene_cfg)

    # Get photon cube
    photon_cube = get_photon_cube(
        **{**scene_cfg, **scene_kwargs}, dtype=torch.uint8, device=device
    )

    # Hot pixel mask
    if scene_cfg.get("hot_pixel"):
        hot_pixel_mask = get_hot_pixel_mask(
            **scene_cfg.hot_pixel,
            rotate_180=scene_cfg.get("rotate_180"),
            flip_lr=scene_cfg.get("flip_lr"),
            flip_ud=scene_cfg.get("flip_ud"),
        )
    else:
        hot_pixel_mask = None

    h, w, t = photon_cube.shape
    mean_frame = photon_cube.sum(dim=-1) / t

    # Output folder
    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True, parents=True)

    # Motion projection form
    (motion_form, motion_simulate_button, motion_kwargs) = setup_camera_form(
        motion_projection_form_cfg,
        "Motion Projection",
        math_expression=r"$\sum_t B_t(\mathbf{x} + \mathbf{r}(t))$, along 2-D trajectory $\mathbf{r}(t)$.",
        color="violet",
        on_click=_enable_motion_projection,
    )

    if st.session_state.run_motion_projection:
        long_exposure = mean_frame
        if isinstance(hot_pixel_mask, np.ndarray):
            long_exposure = nearest_neighbor_inpaint(long_exposure, hot_pixel_mask)
        long_exposure = (long_exposure * 255).to(torch.uint8)
        st.image(
            long_exposure.cpu().numpy(),
            caption="Long exposure",
            output_format="PNG",
            use_column_width=True,
        )

        middle_frame = (
            photon_cube[
                ...,
                t // 2
                - motion_kwargs["num_frames_averaged"] // 2 : t // 2
                + motion_kwargs["num_frames_averaged"] // 2,
            ].sum(axis=-1)
            / motion_kwargs["num_frames_averaged"]
        )
        if isinstance(hot_pixel_mask, np.ndarray):
            middle_frame = nearest_neighbor_inpaint(middle_frame, hot_pixel_mask)

        final_frame = (
            photon_cube[..., -motion_kwargs["num_frames_averaged"] :].sum(axis=-1)
            / motion_kwargs["num_frames_averaged"]
        )
        if isinstance(hot_pixel_mask, np.ndarray):
            final_frame = nearest_neighbor_inpaint(final_frame, hot_pixel_mask)

        # Convert to RGB
        middle_frame = repeat(middle_frame, "h w -> 1 c h w", c=3)
        final_frame = repeat(final_frame, "h w -> 1 c h w", c=3)

        with torch.inference_mode():
            middle_frame, final_frame = raft_transforms(middle_frame, final_frame)
            flow_pred = (
                raft_model(middle_frame, final_frame)[-1].squeeze(dim=0).detach()
            )

        # Show flow_pred
        flow_img = rearrange(flow_to_image(flow_pred), "c h w -> h w c")
        alpha = 0.7
        flow_img = alpha * flow_img + (1 - alpha) * repeat(
            long_exposure, "h w -> h w c", c=3
        )
        flow_img = flow_img.to(torch.uint8)
        st.subheader("Click on a point to bring in 'focus'!")
        st.caption(
            "<div style='text-align: left;'>"
            "Optical flow overlaid on long-exposure."
            "</div>",
            unsafe_allow_html=True,
        )
        value = streamlit_image_coordinates(flow_img.cpu().numpy(), key="numpy")

        if value:
            logger.info(f"Clicked on {value}")
            j_coord, i_coord = value["x"], value["y"]

            flow_at_pixel = torch.round(torch.flip(flow_pred[:, i_coord, j_coord], [0]))

            st.caption(
                "<div style='text-align: center;'>"
                f"Flow predicted at pixel {(i_coord, j_coord)}:"
                f" ({flow_at_pixel[0].item():.3g}, {flow_at_pixel[1].item():.3g})"
                f" pixels / {t} bit-planes"
                "</div>",
                unsafe_allow_html=True,
            )
            flow_direction = flow_at_pixel / torch.linalg.norm(flow_at_pixel)
            velocity = 2.1 * torch.linalg.norm(flow_at_pixel) / t
            line_voxel_ll = get_linear_direction(
                t,
                flow_direction,
                torch.linspace(0, velocity, steps=motion_kwargs["num_directions"]),
            )
            linear_projection_ll = motion_projection_from_photon_cube(
                photon_cube, line_voxel_ll, hot_pixel_mask
            )
            if isinstance(hot_pixel_mask, np.ndarray):
                linear_projection_ll = nearest_neighbor_inpaint(
                    linear_projection_ll, hot_pixel_mask
                )
            linear_projection_ll = repeat(
                linear_projection_ll, "h w num_dir -> num_dir h w c", c=3
            )

            video_path = output_folder / "motion_projection.mp4"
            logger.info(f"Writing motion projection video to {video_path}.")
            write_video(
                str(video_path),
                (linear_projection_ll * 255).to(torch.uint8).cpu(),
                fps=motion_kwargs.get("video_fps", 1),
            )
            st.subheader("**Motion Sweep**")
            st.video(str(video_path))

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
