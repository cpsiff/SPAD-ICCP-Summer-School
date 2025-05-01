import streamlit as st


def setup_sidebar(cfg):
    with st.sidebar, st.expander("Expand to edit photon-cube duration"):
        scene_kwargs = {
            "initial_time_step": st.number_input(
                "Initial Time Step",
                min_value=0,
                max_value=10_000_000,
                value=cfg.scene.initial_time_step,
                step=10,
            ),
            "num_time_step": st.number_input(
                "Num Time Step",
                min_value=10,
                max_value=200_000,
                value=cfg.scene.num_time_step,
                step=10,
            ),
        }
    return scene_kwargs


def setup_camera_form(
    camera_cfg,
    label: str,
    math_expression: str = "",
    color: str = "black",
    button_name: str = "Simulate",
    **submit_kwargs,
):
    form = st.form(f"{label} parameters")

    with form:
        # Event camera params
        st.subheader(f":{color}[{label}]")

        with st.expander(f"_{camera_cfg.description}_"):
            if math_expression:
                st.markdown(math_expression)

            camera_kwargs = {}

            for col, param_set in zip(
                st.columns(2), ["simulation_params", "viz_params"]
            ):
                with col:
                    for k, v in camera_cfg.get(param_set).items():
                        label = (
                            v.label
                            if hasattr(v, "label")
                            else k.replace("_", " ").capitalize()
                        )
                        camera_kwargs[k] = getattr(st, v.input_type)(
                            label, **v.input_kwargs
                        )

        submit_button = st.form_submit_button(button_name, **submit_kwargs)
    return form, submit_button, camera_kwargs
