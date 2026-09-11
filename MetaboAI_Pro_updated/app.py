if all_annotation_tracks:
    with st.expander("🎨 Annotation Colors", expanded=False):
        annotation_colors_live = {}

        for track_col, lookup_df, missing_label in all_annotation_tracks:
            kind = heatmap_module.classify_annotation_series(
                lookup_df[track_col]
            )

            st.markdown(f"**{track_col}**")

            if kind == "continuous":
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.caption("Color map")

                with col2:
                    cmap_choice = st.selectbox(
                        "Color map",
                        heatmap_module.CONTINUOUS_ANNOT_CMAPS,
                        key=f"heatmap_annot_cmap_{track_col}",
                        label_visibility="collapsed",
                    )

                annotation_colors_live[track_col] = cmap_choice

            else:
                values = sorted(
                    str(v)
                    for v in lookup_df[track_col].dropna().unique()
                )

                track_colors = {}

                if values:
                    color_cols = st.columns(len(values))

                    for j, val in enumerate(values):
                        default_hex = (
                            heatmap_module.GROUP_PALETTE[
                                j % len(heatmap_module.GROUP_PALETTE)
                            ]
                        )

                        with color_cols[j]:
                            st.caption(val)

                            track_colors[val] = st.color_picker(
                                f"Color for {val}",
                                value=default_hex,
                                key=f"heatmap_annot_color_{track_col}_{val}",
                                label_visibility="collapsed",
                            )

                annotation_colors_live[track_col] = track_colors

            st.divider()

        fc1, fc2, _ = st.columns([1, 1, 4])

        apply_clicked = fc1.button(
            "Apply Colors",
            type="primary",
            key="heatmap_apply_colors",
        )

        reset_clicked = fc2.button(
            "Reset Colors",
            key="heatmap_reset_colors",
        )

        if apply_clicked:
            st.session_state.heatmap_annotation_colors_applied = (
                annotation_colors_live
            )

        if reset_clicked:
            st.session_state.heatmap_colors_reset_pending = True
            st.session_state.heatmap_annotation_colors_applied = {}
            st.rerun()

annotation_colors = (
    st.session_state.get("heatmap_annotation_colors_applied") or {}
)
