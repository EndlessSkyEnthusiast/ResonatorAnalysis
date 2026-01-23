"""Streamlit app for browsing and comparing TiN XRD data stored in HDF5."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import medfilt, savgol_filter

DEFAULT_H5_PATH = (
    r"\\nas.ads.mwn.de\ga63raz\Desktop\SystOrdnerNachExperimenten\Res"
    r"\AllResonators\Time_temp\xrd_results.h5"
)


@dataclass(frozen=True)
class SampleCurve:
    angle_deg: np.ndarray
    intensity: np.ndarray


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return np.array([v.decode("utf-8", errors="ignore") for v in value])
    return value


@st.cache_resource(show_spinner=False)
def open_h5(path: str) -> h5py.File:
    return h5py.File(path, "r")


@st.cache_data(show_spinner=False)
def build_index(path: str) -> pd.DataFrame:
    with h5py.File(path, "r") as h5:
        if "samples" not in h5:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for sample_id in h5["samples"].keys():
            grp = h5["samples"][sample_id]
            attrs = {k: _decode_attr(v) for k, v in grp.attrs.items()}
            row: dict[str, Any] = {"sample_id": sample_id}
            row.update(attrs)
            rows.append(row)
    df = pd.DataFrame(rows)
    if "date_iso" in df.columns:
        df["date_iso"] = pd.to_datetime(df["date_iso"], errors="coerce")
    for col in ["temperature_C", "Ar_flow", "N2_flow", "pressure_ubar", "sputter_min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_curve(path: str, sample_id: str) -> Optional[SampleCurve]:
    with h5py.File(path, "r") as h5:
        grp = h5.get(f"samples/{sample_id}")
        if grp is None:
            return None
        if "angle_deg" not in grp or "intensity" not in grp:
            return None
        angle = np.asarray(grp["angle_deg"])  # type: ignore[index]
        intensity = np.asarray(grp["intensity"])  # type: ignore[index]
    return SampleCurve(angle_deg=angle, intensity=intensity)


@st.cache_data(show_spinner=False)
def load_detected_peaks(path: str, sample_id: str) -> pd.DataFrame:
    with h5py.File(path, "r") as h5:
        dataset = h5.get(f"samples/{sample_id}/peaks/detected")
        if dataset is None:
            return pd.DataFrame()
        data = np.asarray(dataset)
    return pd.DataFrame(data)


@st.cache_data(show_spinner=False)
def load_tin_fits(path: str, sample_id: str) -> pd.DataFrame:
    with h5py.File(path, "r") as h5:
        dataset = h5.get(f"samples/{sample_id}/fits/tin_peaks")
        if dataset is None:
            return pd.DataFrame()
        data = np.asarray(dataset)
    df = pd.DataFrame(data)
    if set(["h", "k", "l"]).issubset(df.columns):
        df["hkl"] = df[["h", "k", "l"]].astype(str).agg("".join, axis=1)
    return df


@st.cache_data(show_spinner=False)
def load_summary(path: str, sample_id: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with h5py.File(path, "r") as h5:
        grp = h5.get(f"samples/{sample_id}/summary")
        if grp is None:
            return summary
        if isinstance(grp, h5py.Group):
            for key, value in grp.attrs.items():
                summary[key] = _decode_attr(value)
            for key in grp.keys():
                summary[key] = _decode_attr(np.asarray(grp[key]))
        else:
            summary["summary"] = _decode_attr(np.asarray(grp))
    return summary


def normalize_curve(intensity: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        denom = np.nanmax(intensity)
        return intensity / denom if denom else intensity
    if mode == "area":
        area = np.trapz(intensity)
        return intensity / area if area else intensity
    return intensity


def apply_baseline(intensity: np.ndarray, mode: str, window: int = 101) -> np.ndarray:
    if mode == "rolling median":
        kernel = max(3, window | 1)
        baseline = medfilt(intensity, kernel_size=kernel)
        return intensity - baseline
    if mode == "polynomial":
        x = np.arange(intensity.size)
        coeff = np.polyfit(x, intensity, deg=3)
        baseline = np.polyval(coeff, x)
        return intensity - baseline
    return intensity


def smooth_curve(intensity: np.ndarray, window: int = 31, polyorder: int = 3) -> np.ndarray:
    window = max(5, window | 1)
    if intensity.size < window:
        return intensity
    return savgol_filter(intensity, window_length=window, polyorder=polyorder)


def filter_dataframe(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key not in df.columns:
            continue
        if value is None:
            continue
        if isinstance(value, tuple) and len(value) == 2:
            low, high = value
            mask &= df[key].between(low, high, inclusive="both")
        elif isinstance(value, list):
            if value:
                mask &= df[key].isin(value)
        elif isinstance(value, str) and value:
            mask &= df[key].astype(str).str.contains(value, case=False, na=False)
    return df[mask]


def format_sample_id_list(samples: Iterable[str]) -> str:
    return ", ".join(sorted(samples))


def _numeric_range(series: pd.Series) -> Optional[tuple[float, float]]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return None
    return float(numeric.min()), float(numeric.max())


def _range_slider(
    label: str,
    series: pd.Series,
    state_key: str,
) -> Optional[tuple[float, float]]:
    range_limits = _numeric_range(series)
    if range_limits is None:
        st.sidebar.write(f"{label}: no data available")
        return None
    min_value, max_value = range_limits
    limits_key = f"{state_key}::limits"
    previous_limits = st.session_state.get(limits_key, (min_value, max_value))
    default_range = st.session_state.get(state_key, (min_value, max_value))
    if not isinstance(default_range, (tuple, list)) or len(default_range) != 2:
        default_range = (min_value, max_value)
    if not isinstance(previous_limits, (tuple, list)) or len(previous_limits) != 2:
        previous_limits = (min_value, max_value)
    previous_min, previous_max = previous_limits
    default_low, default_high = float(default_range[0]), float(default_range[1])
    if min_value < previous_min and np.isclose(default_low, previous_min):
        default_low = min_value
    if max_value > previous_max and np.isclose(default_high, previous_max):
        default_high = max_value
    default_range = (
        max(min_value, float(default_low)),
        min(max_value, float(default_high)),
    )
    if default_range[0] > default_range[1]:
        default_range = (min_value, max_value)
    st.session_state[limits_key] = (min_value, max_value)
    if min_value == max_value:
        st.sidebar.write(f"{label}: {min_value}")
        st.session_state[state_key] = (min_value, max_value)
        return (min_value, max_value)
    return st.sidebar.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default_range,
        key=state_key,
    )


def init_session_state() -> None:
    st.session_state.setdefault("plot_samples", [])
    st.session_state.setdefault("sample_table", pd.DataFrame())
    st.session_state.setdefault("uploaded_path", "")


def update_plot_samples(samples: Iterable[str]) -> None:
    current = set(st.session_state.plot_samples)
    current.update(samples)
    st.session_state.plot_samples = sorted(current)


def remove_plot_sample(sample_id: str) -> None:
    st.session_state.plot_samples = [
        s for s in st.session_state.plot_samples if s != sample_id
    ]


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_fit_table(path: str, samples: list[str], metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_id in samples:
        fits = load_tin_fits(path, sample_id)
        if fits.empty:
            continue
        fits = fits.copy()
        fits["sample_id"] = sample_id
        rows.append(fits)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = df.merge(metadata, on="sample_id", how="left")
    if "r2" in df.columns:
        df["r2_flag"] = df["r2"] < 0.6
    return df


def build_summary_table(path: str, samples: list[str], metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_id in samples:
        summary = load_summary(path, sample_id)
        if not summary:
            continue
        summary_row = {"sample_id": sample_id}
        summary_row.update(summary)
        rows.append(summary_row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.merge(metadata, on="sample_id", how="left")


def render_sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Filters")
    filters: dict[str, Any] = {}

    if "temperature_C" in df.columns:
        selection = _range_slider(
            "temperature_C range",
            df["temperature_C"],
            "filter::temperature_C_range",
        )
        if selection is not None:
            filters["temperature_C"] = selection
    if "temp_label" in df.columns:
        temp_labels = sorted(df["temp_label"].dropna().unique())
        filters["temp_label"] = st.sidebar.multiselect("temp_label", temp_labels)

    for key in ["Ar_flow", "N2_flow"]:
        if key in df.columns:
            values = sorted(df[key].dropna().unique())
            filters[key] = st.sidebar.multiselect(key, values)

    for key in ["pressure_ubar", "sputter_min"]:
        if key in df.columns:
            selection = _range_slider(f"{key} range", df[key], f"filter::{key}_range")
            if selection is not None:
                filters[key] = selection
            exact_values = sorted(df[key].dropna().unique())
            filters[f"{key}_exact"] = st.sidebar.multiselect(
                f"{key} exact", exact_values
            )

    if "date_iso" in df.columns:
        date_min = df["date_iso"].min()
        date_max = df["date_iso"].max()
        if pd.notna(date_min) and pd.notna(date_max):
            date_range = st.sidebar.date_input(
                "date range",
                value=(date_min.date(), date_max.date()),
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                filters["date_iso"] = (
                    pd.to_datetime(date_range[0]),
                    pd.to_datetime(date_range[1]),
                )

    comment_search = ""
    if "comment" in df.columns:
        comment_search = st.sidebar.text_input("comment contains")
        comment_values = sorted(df["comment"].dropna().unique())
        filters["comment"] = st.sidebar.multiselect("comment exact", comment_values)

    filtered = filter_dataframe(df, filters)

    if "pressure_ubar_exact" in filters and filters["pressure_ubar_exact"]:
        filtered = filtered[filtered["pressure_ubar"].isin(filters["pressure_ubar_exact"])]
    if "sputter_min_exact" in filters and filters["sputter_min_exact"]:
        filtered = filtered[filtered["sputter_min"].isin(filters["sputter_min_exact"])]
    if comment_search:
        filtered = filtered[
            filtered["comment"].astype(str).str.contains(comment_search, case=False, na=False)
        ]

    return filtered, df


def render_sample_table(filtered: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.subheader("Sample table")
    if filtered.empty:
        st.sidebar.info("No samples match filters.")
        return filtered
    display_cols = [
        col
        for col in [
            "sample_id",
            "temperature_C",
            "temp_label",
            "Ar_flow",
            "N2_flow",
            "pressure_ubar",
            "sputter_min",
            "date_iso",
            "comment",
        ]
        if col in filtered.columns
    ]
    table = filtered[display_cols].copy()
    table.insert(0, "select", False)
    edited = st.sidebar.data_editor(
        table,
        hide_index=True,
        column_config={"select": st.column_config.CheckboxColumn(required=False)},
        height=300,
    )
    return edited


def render_plot_list_controls(edited_table: pd.DataFrame) -> None:
    st.sidebar.subheader("Plot list")
    add_selected = st.sidebar.button("Add selected to plot list")
    clear = st.sidebar.button("Clear plot list")
    add_input = st.sidebar.text_input("Add by sample_id")
    add_button = st.sidebar.button("Add sample_id")

    if add_selected and "select" in edited_table.columns:
        selected = edited_table.loc[edited_table["select"], "sample_id"].tolist()
        update_plot_samples(selected)
    if clear:
        st.session_state.plot_samples = []
    if add_button and add_input:
        update_plot_samples([add_input])

    if st.session_state.plot_samples:
        for sample_id in st.session_state.plot_samples:
            col1, col2 = st.sidebar.columns([4, 1])
            col1.write(sample_id)
            if col2.button("✖", key=f"remove_{sample_id}"):
                remove_plot_sample(sample_id)
                st.experimental_rerun()
    else:
        st.sidebar.caption("No samples in plot list.")


def plot_overlay(
    path: str,
    samples: list[str],
    show_peaks: bool,
    show_fits: bool,
    normalize: str,
    y_scale: str,
    baseline: str,
    smoothing: bool,
    waterfall: bool,
) -> Optional[go.Figure]:
    if not samples:
        st.info("Select samples to plot.")
        return None

    fig = go.Figure()
    offset = 0.0
    for sample_id in samples:
        curve = load_curve(path, sample_id)
        if curve is None:
            st.warning(f"Missing curve for {sample_id}.")
            continue
        intensity = normalize_curve(curve.intensity, normalize)
        intensity = apply_baseline(intensity, baseline)
        if smoothing:
            intensity = smooth_curve(intensity)
        if waterfall:
            intensity = intensity + offset
            offset += np.nanmax(intensity) * 0.1
        fig.add_trace(
            go.Scatter(
                x=curve.angle_deg,
                y=intensity,
                mode="lines",
                name=sample_id,
            )
        )
        if show_peaks:
            peaks = load_detected_peaks(path, sample_id)
            if not peaks.empty and "pos_deg" in peaks.columns:
                fig.add_trace(
                    go.Scatter(
                        x=peaks["pos_deg"],
                        y=np.interp(peaks["pos_deg"], curve.angle_deg, intensity),
                        mode="markers",
                        marker=dict(size=6),
                        name=f"{sample_id} peaks",
                    )
                )
        if show_fits:
            fits = load_tin_fits(path, sample_id)
            if not fits.empty and "xc_deg" in fits.columns:
                for _, row in fits.iterrows():
                    x_val = row.get("xc_deg")
                    hkl = row.get("hkl", "")
                    fig.add_trace(
                        go.Scatter(
                            x=[x_val, x_val],
                            y=[np.nanmin(intensity), np.nanmax(intensity)],
                            mode="lines",
                            line=dict(dash="dot"),
                            name=f"{sample_id} {hkl}",
                            showlegend=False,
                        )
                    )
    fig.update_layout(
        xaxis_title="2θ (deg)",
        yaxis_title="Intensity (a.u.)",
        yaxis_type=y_scale,
        legend_title="Samples",
        template="plotly_white",
    )
    return fig


def plot_fit_metrics(df: pd.DataFrame, x_axis: str, color: Optional[str]) -> list[go.Figure]:
    figures = []
    if df.empty:
        return figures
    for metric, title in [
        ("xc_deg", "Peak position (deg)"),
        ("FWHM_deg", "FWHM (deg)"),
        ("a_A", "Lattice parameter a (Å)"),
    ]:
        if metric not in df.columns:
            continue
        fig = px.scatter(
            df,
            x=x_axis,
            y=metric,
            color=color,
            symbol="hkl" if "hkl" in df.columns else None,
            hover_data=["sample_id"],
        )
        fig.update_layout(title=title, template="plotly_white")
        figures.append(fig)
    return figures


def plot_summary_metric(
    df: pd.DataFrame, metric: str, x_axis: str, color: Optional[str], trendline: bool
) -> Optional[go.Figure]:
    if df.empty or metric not in df.columns or x_axis not in df.columns:
        return None
    fig = px.scatter(df, x=x_axis, y=metric, color=color, hover_data=["sample_id"])
    if trendline:
        valid = df[[x_axis, metric]].dropna()
        if len(valid) >= 2:
            x_vals = pd.to_numeric(valid[x_axis], errors="coerce")
            y_vals = pd.to_numeric(valid[metric], errors="coerce")
            mask = x_vals.notna() & y_vals.notna()
            if mask.sum() >= 2:
                coeff = np.polyfit(x_vals[mask], y_vals[mask], deg=1)
                x_line = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 100)
                y_line = coeff[0] * x_line + coeff[1]
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="trend"))
    fig.update_layout(template="plotly_white")
    return fig


def correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu")
    fig.update_layout(title="Correlation heatmap", template="plotly_white")
    return fig


def main() -> None:
    st.set_page_config(page_title="XRD Plot", layout="wide")
    init_session_state()

    st.title("TiN XRD Explorer")

    st.sidebar.header("Data source")
    default_path = os.getenv("XRD_H5_PATH", DEFAULT_H5_PATH)
    path_input = st.sidebar.text_input("HDF5 path", value=default_path)
    uploaded = st.sidebar.file_uploader("Browse .h5", type=["h5", "hdf5"])

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded.getbuffer())
            st.session_state.uploaded_path = tmp.name
    data_path = st.session_state.uploaded_path or path_input

    if not os.path.exists(data_path):
        st.error("HDF5 file not found.")
        st.stop()

    _ = open_h5(data_path)
    df_index = build_index(data_path)
    if df_index.empty:
        st.warning("No samples found in HDF5.")
        st.stop()

    filtered, full_df = render_sidebar_filters(df_index)
    edited_table = render_sample_table(filtered)
    render_plot_list_controls(edited_table)

    st.sidebar.download_button(
        "Download filtered table CSV",
        dataframe_to_csv(filtered),
        file_name="xrd_samples_filtered.csv",
        mime="text/csv",
    )

    samples = st.session_state.plot_samples
    tabs = st.tabs(["XRD Overlay", "Peak Fit Comparison (TiN)", "Derived Summary"])

    with tabs[0]:
        st.subheader("XRD Overlay")
        col1, col2, col3 = st.columns(3)
        normalize = col1.selectbox("Normalize", ["none", "max", "area"], index=1)
        y_scale = col2.selectbox("Y scale", ["linear", "log"])
        baseline = col3.selectbox("Baseline subtraction", ["none", "rolling median", "polynomial"])
        col4, col5, col6 = st.columns(3)
        smoothing = col4.checkbox("Smoothing (Savitzky-Golay)")
        waterfall = col5.checkbox("Waterfall offset")
        show_peaks = col6.checkbox("Show detected peaks", value=True)
        show_fits = st.checkbox("Show fitted TiN peaks", value=True)
        x_min, x_max = 0.0, 90.0
        if samples:
            angles = []
            for sample_id in samples:
                curve = load_curve(data_path, sample_id)
                if curve is not None:
                    angles.append(curve.angle_deg)
            if angles:
                x_min = float(np.nanmin([np.nanmin(a) for a in angles]))
                x_max = float(np.nanmax([np.nanmax(a) for a in angles]))
        x_range = st.slider("2θ range", min_value=x_min, max_value=x_max, value=(x_min, x_max))

        fig = plot_overlay(
            data_path,
            samples,
            show_peaks=show_peaks,
            show_fits=show_fits,
            normalize=normalize,
            y_scale=y_scale,
            baseline=baseline,
            smoothing=smoothing,
            waterfall=waterfall,
        )
        if fig is not None:
            fig.update_xaxes(range=list(x_range))
            st.plotly_chart(fig, use_container_width=True)
            html = fig.to_html(include_plotlyjs="cdn")
            st.download_button(
                "Download overlay HTML",
                data=html,
                file_name="xrd_overlay.html",
                mime="text/html",
            )

    with tabs[1]:
        st.subheader("Peak Fit Comparison (TiN)")
        fit_df = build_fit_table(data_path, samples, full_df)
        if fit_df.empty:
            st.info("No TiN fits available for selected samples.")
        else:
            if "r2_flag" in fit_df.columns:
                styled = fit_df.style.apply(
                    lambda col: [
                        "background-color: #ffe6e6" if val else "" for val in col
                    ],
                    subset=["r2_flag"],
                )
                st.dataframe(styled)
            else:
                st.dataframe(fit_df)
            st.download_button(
                "Download fit table CSV",
                dataframe_to_csv(fit_df),
                file_name="xrd_fit_table.csv",
                mime="text/csv",
            )
            x_axis = st.selectbox(
                "X axis",
                [
                    "sample_id",
                    "temperature_C",
                    "date_iso",
                    "pressure_ubar",
                    "sputter_min",
                    "Ar_flow",
                    "N2_flow",
                ],
                key="fit_metrics_x_axis",
            )
            color_by = st.selectbox(
                "Color by",
                ["", "temperature_C", "pressure_ubar", "sputter_min", "Ar_flow", "N2_flow"],
                key="fit_metrics_color_by",
            )
            color_val = color_by or None
            for fig in plot_fit_metrics(fit_df, x_axis=x_axis, color=color_val):
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Derived Summary")
        summary_df = build_summary_table(data_path, samples, full_df)
        if summary_df.empty:
            st.info("No summary metrics available for selected samples.")
        else:
            st.caption("eps_WH and D_WH_nm are apparent/uncorrected values.")
            st.dataframe(summary_df)
            st.download_button(
                "Download summary table CSV",
                dataframe_to_csv(summary_df),
                file_name="xrd_summary_table.csv",
                mime="text/csv",
            )
            numeric_cols = summary_df.select_dtypes(include=[np.number]).columns.tolist()
            x_axis = st.selectbox(
                "X axis",
                options=summary_df.columns,
                index=0,
                key="summary_x_axis",
            )
            metric = st.selectbox("Metric", options=numeric_cols, key="summary_metric")
            color_by = st.selectbox(
                "Color by",
                options=["", "temperature_C", "pressure_ubar", "sputter_min", "Ar_flow", "N2_flow"],
                key="summary_color_by",
            )
            trendline = st.checkbox("Trendline", value=True)
            fig = plot_summary_metric(
                summary_df,
                metric=metric,
                x_axis=x_axis,
                color=color_by or None,
                trendline=trendline,
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            heatmap = correlation_heatmap(summary_df)
            if heatmap is not None:
                st.plotly_chart(heatmap, use_container_width=True)


if __name__ == "__main__":
    main()
