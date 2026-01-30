#!/usr/bin/env python3
"""Batch-analyze PANalytical/Empyrean XRD CSV files into a single HDF5 file.

Changelog (2025-02-xx)
- Added instrument broadening correction fields: fwhm_corr_deg, fwhm_corr_err_deg,
  fwhm_corr_rad, fwhm_corr_err_rad, and good_corr flags in fits.
- Switched peak model background to linear (y0 + m*(x-xc)); slope stored as m/m_err.
- Scherrer and Williamson-Hall now use instrument-corrected FWHM (beta_corr).
  Added uncorrected summary fields (D_scherrer_uncorr_median_nm, D_WH_uncorr_nm,
  WH_intercept_uncorr) for backward comparison.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

__version__ = "0.2.0"

# Instrument broadening (FWHM in 2θ, degrees).
INSTR_FWHM_DEG = 0.3
# Peak shape selection (currently only gaussian is implemented).
PEAK_SHAPE = "gaussian"

TINK_WINDOWS = {
    (1, 1, 1): (34.5, 38.5),
    (2, 0, 0): (40.5, 44.5),
    (2, 2, 0): (59.5, 63.5),
    (3, 1, 1): (72.0, 76.5),
    (2, 2, 2): (76.5, 79.5),
}


@dataclass
class FitResult:
    h: int
    k: int
    l: int
    xc_deg: float
    xc_err_deg: float
    sigma_deg: float
    sigma_err_deg: float
    fwhm_deg: float
    fwhm_err_deg: float
    fwhm_corr_deg: float
    fwhm_corr_err_deg: float
    fwhm_corr_rad: float
    fwhm_corr_err_rad: float
    amplitude: float
    amplitude_err: float
    y0: float
    y0_err: float
    m: float
    m_err: float
    area: float
    r2: float
    good: bool
    good_corr: bool
    d_A: float
    a_A: float
    window_lo: float
    window_hi: float


@dataclass
class DetectedPeak:
    pos_deg: float
    prominence: float
    height: float


class XRDParseError(RuntimeError):
    """Raised when an XRD CSV file cannot be parsed."""


def gaussian_with_background(
    x: np.ndarray,
    y0: float,
    m: float,
    a: float,
    xc: float,
    sigma: float,
) -> np.ndarray:
    return y0 + m * (x - xc) + a * np.exp(-0.5 * ((x - xc) / sigma) ** 2)


def compute_corrected_fwhm(
    fwhm_deg: float,
    fwhm_err_deg: float,
    instr_fwhm_deg: float,
) -> Tuple[float, float, float, float]:
    """Return corrected FWHM in deg/rad plus errors (NaN if invalid)."""
    if not np.isfinite(fwhm_deg) or fwhm_deg <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    fwhm_rad = math.radians(fwhm_deg)
    instr_rad = math.radians(instr_fwhm_deg)
    if fwhm_rad <= instr_rad:
        return float("nan"), float("nan"), float("nan"), float("nan")
    corr_rad = math.sqrt(max(fwhm_rad**2 - instr_rad**2, 0.0))
    corr_deg = math.degrees(corr_rad)
    if not np.isfinite(fwhm_err_deg) or fwhm_err_deg <= 0 or corr_rad == 0:
        return corr_deg, float("nan"), corr_rad, float("nan")
    fwhm_err_rad = math.radians(fwhm_err_deg)
    corr_err_rad = fwhm_rad / corr_rad * fwhm_err_rad
    corr_err_deg = math.degrees(corr_err_rad)
    return corr_deg, corr_err_deg, corr_rad, corr_err_rad


def parse_header_and_data(path: Path) -> Tuple[Dict[str, str], np.ndarray, np.ndarray]:
    """Parse an Empyrean CSV file into header metadata and scan arrays."""
    header: Dict[str, str] = {}
    scan_rows: List[Tuple[float, float]] = []
    in_header = False
    in_scan = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            token = row[0].strip()
            if token.startswith("[Measurement conditions]"):
                in_header = True
                in_scan = False
                continue
            if token.startswith("[Scan points]"):
                in_scan = True
                in_header = False
                continue
            if in_header:
                if len(row) >= 2:
                    key = row[0].strip()
                    value = row[1].strip()
                    if key:
                        header[key] = value
                continue
            if in_scan:
                if token.lower().startswith("angle"):
                    continue
                if len(row) >= 2:
                    try:
                        angle = float(row[0])
                        intensity = float(row[1])
                        scan_rows.append((angle, intensity))
                    except ValueError:
                        continue
    if not scan_rows:
        raise XRDParseError(f"No scan points found in {path}")
    data = np.array(scan_rows, dtype=float)
    return header, data[:, 0], data[:, 1]


def parse_filename_metadata(filename: str) -> Dict[str, object]:
    """Parse filename metadata from the expected naming convention."""
    stem = Path(filename).stem
    tokens = stem.split("_")
    if len(tokens) < 6:
        raise XRDParseError(f"Filename {filename} does not match expected format")
    temperature_token, ar_token, n2_token, pressure_token, sputter_token, date_token = tokens[:6]
    comment_token = "_".join(tokens[6:]) if len(tokens) > 6 else ""

    temp_label = temperature_token
    temperature_c = float("nan")
    if temperature_token.upper() != "RT":
        match = re.match(r"(-?\d+(?:\.\d+)?)C", temperature_token)
        if match:
            temperature_c = float(match.group(1))
    ar_flow = _parse_numeric_token(ar_token, "Ar")
    n2_flow = _parse_numeric_token(n2_token, "N2")
    pressure_ubar = _parse_pressure(pressure_token)
    sputter_min = _parse_numeric_token(sputter_token, "min")
    date_iso = _parse_date(date_token)
    comment = comment_token if comment_token.strip() else "None"

    return {
        "temp_label": temp_label,
        "temperature_C": temperature_c,
        "Ar_flow": ar_flow,
        "N2_flow": n2_flow,
        "pressure_ubar": pressure_ubar,
        "sputter_min": sputter_min,
        "date_iso": date_iso,
        "comment": comment,
    }


def _parse_numeric_token(token: str, suffix: str) -> float:
    match = re.match(rf"(-?\d+(?:\.\d+)?){re.escape(suffix)}", token)
    if not match:
        return float("nan")
    return float(match.group(1))


def _parse_pressure(token: str) -> float:
    cleaned = token.replace("ubar", "")
    cleaned = cleaned.replace("-", ".")
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def _parse_date(token: str) -> str:
    try:
        date = dt.datetime.strptime(token, "%Y%m%d").date()
        return date.isoformat()
    except ValueError:
        return ""


def _smooth_for_peaks(intensity: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    window = max(window, polyorder + 2)
    if window % 2 == 0:
        window += 1
    if intensity.size < window:
        return intensity
    return savgol_filter(intensity, window_length=window, polyorder=polyorder)


def fit_peak_window(
    angle: np.ndarray,
    intensity: np.ndarray,
    window: Tuple[float, float],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    mask = (angle >= window[0]) & (angle <= window[1])
    x = angle[mask]
    y = intensity[mask]
    if x.size < 5:
        return x, y, None
    if PEAK_SHAPE != "gaussian":
        raise ValueError(f"Unsupported PEAK_SHAPE={PEAK_SHAPE!r} (only 'gaussian' implemented).")
    y0_guess = float(np.min(y))
    a_guess = float(np.max(y) - y0_guess)
    xc_guess = float(x[np.argmax(y)])
    sigma_guess = 0.15
    slope_guess = float((y[-1] - y[0]) / (x[-1] - x[0])) if x.size > 1 else 0.0
    slope_limit = max(abs(slope_guess) * 5.0, abs(a_guess) / max(x.ptp(), 1e-3))
    y0_lower = float(np.min(y) - abs(a_guess))
    y0_upper = float(np.max(y) + abs(a_guess))
    bounds = (
        [y0_lower, -slope_limit, 0.0, window[0], 0.01],
        [y0_upper, slope_limit, np.inf, window[1], 1.5],
    )
    try:
        popt, pcov = curve_fit(
            gaussian_with_background,
            x,
            y,
            p0=[y0_guess, slope_guess, a_guess, xc_guess, sigma_guess],
            bounds=bounds,
            maxfev=10000,
        )
    except RuntimeError:
        return x, y, None
    return x, y, (popt, pcov)


def compute_r2(y: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1 - ss_res / ss_tot


def analyze_tin_peaks(
    angle: np.ndarray,
    intensity: np.ndarray,
    lambda_A: float,
) -> List[FitResult]:
    results: List[FitResult] = []
    for (h, k, l), window in TINK_WINDOWS.items():
        x, y, fit = fit_peak_window(angle, intensity, window)
        if fit is None or x.size == 0:
            results.append(
                FitResult(
                    h=h,
                    k=k,
                    l=l,
                    xc_deg=float("nan"),
                    xc_err_deg=float("nan"),
                    sigma_deg=float("nan"),
                    sigma_err_deg=float("nan"),
                    fwhm_deg=float("nan"),
                    fwhm_err_deg=float("nan"),
                    fwhm_corr_deg=float("nan"),
                    fwhm_corr_err_deg=float("nan"),
                    fwhm_corr_rad=float("nan"),
                    fwhm_corr_err_rad=float("nan"),
                    amplitude=float("nan"),
                    amplitude_err=float("nan"),
                    y0=float("nan"),
                    y0_err=float("nan"),
                    m=float("nan"),
                    m_err=float("nan"),
                    area=float("nan"),
                    r2=float("nan"),
                    good=False,
                    good_corr=False,
                    d_A=float("nan"),
                    a_A=float("nan"),
                    window_lo=window[0],
                    window_hi=window[1],
                )
            )
            continue
        popt, pcov = fit
        y_fit = gaussian_with_background(x, *popt)
        r2 = compute_r2(y, y_fit)
        y0, m, a, xc, sigma = popt
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(5, float("nan"))
        y0_err, m_err, a_err, xc_err, sigma_err = perr
        fwhm = 2.354820045 * sigma
        fwhm_err = 2.354820045 * sigma_err
        fwhm_corr_deg, fwhm_corr_err_deg, fwhm_corr_rad, fwhm_corr_err_rad = compute_corrected_fwhm(
            fwhm, fwhm_err, INSTR_FWHM_DEG
        )
        area = a * sigma * math.sqrt(2 * math.pi)
        theta_rad = math.radians(xc / 2.0)
        d_A = float("nan")
        a_A = float("nan")
        if lambda_A and math.sin(theta_rad) != 0:
            d_A = lambda_A / (2 * math.sin(theta_rad))
            a_A = d_A * math.sqrt(h * h + k * k + l * l)
        window_width = window[1] - window[0]
        edge_margin = 0.1 * window_width
        xc_in_window = window[0] + edge_margin < xc < window[1] - edge_margin
        sigma_lower, sigma_upper = 0.01, 1.5
        sigma_ok = sigma_lower * 1.05 < sigma < sigma_upper * 0.95
        residual_std = float(np.nanstd(y - y_fit)) if y.size else float("nan")
        amplitude_ok = a > 0 and (not np.isnan(residual_std) and residual_std > 0) and a > 3 * residual_std
        good = bool(r2 >= 0.6)
        corr_valid = np.isfinite(fwhm_corr_deg)
        good_corr = bool(good and xc_in_window and sigma_ok and amplitude_ok and corr_valid)
        results.append(
            FitResult(
                h=h,
                k=k,
                l=l,
                xc_deg=xc,
                xc_err_deg=xc_err,
                sigma_deg=sigma,
                sigma_err_deg=sigma_err,
                fwhm_deg=fwhm,
                fwhm_err_deg=fwhm_err,
                fwhm_corr_deg=fwhm_corr_deg,
                fwhm_corr_err_deg=fwhm_corr_err_deg,
                fwhm_corr_rad=fwhm_corr_rad,
                fwhm_corr_err_rad=fwhm_corr_err_rad,
                amplitude=a,
                amplitude_err=a_err,
                y0=y0,
                y0_err=y0_err,
                m=m,
                m_err=m_err,
                area=area,
                r2=r2,
                good=good,
                good_corr=good_corr,
                d_A=d_A,
                a_A=a_A,
                window_lo=window[0],
                window_hi=window[1],
            )
        )
    return results


def detect_other_peaks(
    angle: np.ndarray,
    intensity: np.ndarray,
    prominence: float,
) -> List[DetectedPeak]:
    peak_indices, properties = find_peaks(intensity, prominence=prominence)
    results: List[DetectedPeak] = []
    for i, idx in enumerate(peak_indices):
        results.append(
            DetectedPeak(
                pos_deg=float(angle[idx]),
                prominence=float(properties["prominences"][i]),
                height=float(intensity[idx]),
            )
        )
    return results


def summarize_peaks(fits: Sequence[FitResult], lambda_A: float) -> Dict[str, float]:
    good = [fit for fit in fits if fit.good]
    good_corr = [fit for fit in fits if fit.good_corr]
    a_values = [fit.a_A for fit in good if not math.isnan(fit.a_A)]
    a_mean = float(np.mean(a_values)) if a_values else float("nan")
    a_std = float(np.std(a_values)) if a_values else float("nan")
    n_good = float(len(good))
    n_good_corr = float(len(good_corr))

    ratios = {}
    area_by_hkl = {(fit.h, fit.k, fit.l): fit.area for fit in good}
    base = area_by_hkl.get((1, 1, 1), float("nan"))
    for hkl, label in [((2, 0, 0), "I200_I111"), ((2, 2, 0), "I220_I111"), ((3, 1, 1), "I311_I111"), ((2, 2, 2), "I222_I111")]:
        area = area_by_hkl.get(hkl, float("nan"))
        ratios[label] = area / base if base and not math.isnan(base) and not math.isnan(area) else float("nan")

    scherrer = []
    scherrer_uncorr = []
    lambda_nm = lambda_A * 0.1
    for fit in good_corr:
        theta = math.radians(fit.xc_deg / 2.0)
        beta = fit.fwhm_corr_rad
        if beta <= 0 or math.cos(theta) == 0 or not np.isfinite(beta):
            continue
        scherrer.append(0.9 * lambda_nm / (beta * math.cos(theta)))
    for fit in good:
        theta = math.radians(fit.xc_deg / 2.0)
        beta = math.radians(fit.fwhm_deg)
        if beta <= 0 or math.cos(theta) == 0:
            continue
        scherrer_uncorr.append(0.9 * lambda_nm / (beta * math.cos(theta)))
    d_scherrer_median = float(np.median(scherrer)) if scherrer else float("nan")
    d_scherrer_uncorr_median = float(np.median(scherrer_uncorr)) if scherrer_uncorr else float("nan")

    eps_wh = float("nan")
    intercept = float("nan")
    d_wh = float("nan")
    eps_wh_uncorr = float("nan")
    intercept_uncorr = float("nan")
    d_wh_uncorr = float("nan")
    if len(good_corr) >= 3:
        x_vals = []
        y_vals = []
        weights = []
        for fit in good_corr:
            theta = math.radians(fit.xc_deg / 2.0)
            beta = fit.fwhm_corr_rad
            if beta <= 0 or not np.isfinite(beta):
                continue
            x_vals.append(4 * math.sin(theta))
            y_val = beta * math.cos(theta)
            y_vals.append(y_val)
            sigma_beta = fit.fwhm_corr_err_rad
            if not np.isfinite(sigma_beta) or sigma_beta <= 0:
                weights.append(np.nan)
            else:
                sigma_y = sigma_beta * math.cos(theta)
                weights.append(1 / sigma_y**2 if sigma_y > 0 else np.nan)
        if len(x_vals) >= 3:
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            w_arr = np.array(weights)
            if np.all(~np.isfinite(w_arr)):
                slope, intercept = np.polyfit(x_arr, y_arr, 1)
            else:
                w_arr = np.where(np.isfinite(w_arr), w_arr, 0.0)
                slope, intercept = np.polyfit(x_arr, y_arr, 1, w=w_arr)
            eps_wh = float(slope)
            if intercept > 0:
                d_wh = 0.9 * lambda_nm / intercept
    if len(good) >= 3:
        x_vals = []
        y_vals = []
        for fit in good:
            theta = math.radians(fit.xc_deg / 2.0)
            beta = math.radians(fit.fwhm_deg)
            if beta <= 0:
                continue
            x_vals.append(4 * math.sin(theta))
            y_vals.append(beta * math.cos(theta))
        if len(x_vals) >= 3:
            slope, intercept_uncorr = np.polyfit(np.array(x_vals), np.array(y_vals), 1)
            eps_wh_uncorr = float(slope)
            if intercept_uncorr > 0:
                d_wh_uncorr = 0.9 * lambda_nm / intercept_uncorr

    summary = {
        "a_mean_A": a_mean,
        "a_std_A": a_std,
        "n_good": n_good,
        "n_good_corr": n_good_corr,
        "eps_WH": eps_wh,
        "D_WH_nm": d_wh,
        "D_scherrer_median_nm": d_scherrer_median,
        "WH_intercept": float(intercept),
        "eps_WH_uncorr": eps_wh_uncorr,
        "D_WH_uncorr_nm": d_wh_uncorr,
        "D_scherrer_uncorr_median_nm": d_scherrer_uncorr_median,
        "WH_intercept_uncorr": float(intercept_uncorr),
    }
    summary.update(ratios)
    return summary


def to_structured_array_fits(fits: Sequence[FitResult]) -> np.ndarray:
    dtype = [
        ("h", "i4"),
        ("k", "i4"),
        ("l", "i4"),
        ("xc_deg", "f8"),
        ("xc_err_deg", "f8"),
        ("sigma_deg", "f8"),
        ("sigma_err_deg", "f8"),
        ("FWHM_deg", "f8"),
        ("FWHM_err_deg", "f8"),
        ("fwhm_corr_deg", "f8"),
        ("fwhm_corr_err_deg", "f8"),
        ("fwhm_corr_rad", "f8"),
        ("fwhm_corr_err_rad", "f8"),
        ("A", "f8"),
        ("A_err", "f8"),
        ("y0", "f8"),
        ("y0_err", "f8"),
        ("m", "f8"),
        ("m_err", "f8"),
        ("area", "f8"),
        ("r2", "f8"),
        ("good", "i1"),
        ("good_corr", "i1"),
        ("d_A", "f8"),
        ("a_A", "f8"),
        ("window_lo", "f8"),
        ("window_hi", "f8"),
    ]
    array = np.zeros(len(fits), dtype=dtype)
    for i, fit in enumerate(fits):
        array[i] = (
            fit.h,
            fit.k,
            fit.l,
            fit.xc_deg,
            fit.xc_err_deg,
            fit.sigma_deg,
            fit.sigma_err_deg,
            fit.fwhm_deg,
            fit.fwhm_err_deg,
            fit.fwhm_corr_deg,
            fit.fwhm_corr_err_deg,
            fit.fwhm_corr_rad,
            fit.fwhm_corr_err_rad,
            fit.amplitude,
            fit.amplitude_err,
            fit.y0,
            fit.y0_err,
            fit.m,
            fit.m_err,
            fit.area,
            fit.r2,
            int(fit.good),
            int(fit.good_corr),
            fit.d_A,
            fit.a_A,
            fit.window_lo,
            fit.window_hi,
        )
    return array


def to_structured_array_peaks(peaks: Sequence[DetectedPeak]) -> np.ndarray:
    dtype = [("pos_deg", "f8"), ("prominence", "f8"), ("height", "f8")]
    array = np.zeros(len(peaks), dtype=dtype)
    for i, peak in enumerate(peaks):
        array[i] = (peak.pos_deg, peak.prominence, peak.height)
    return array


def write_sample_group(
    h5: h5py.File,
    sample_id: str,
    metadata: Dict[str, object],
    header: Dict[str, str],
    angle: np.ndarray,
    intensity: np.ndarray,
    fits: Sequence[FitResult],
    detected: Sequence[DetectedPeak],
    summary: Dict[str, float],
    filename: str,
) -> None:
    group = h5.create_group(f"samples/{sample_id}")
    for key, value in metadata.items():
        group.attrs[key] = value
    for key, value in header.items():
        group.attrs[f"header_{key}"] = value
    group.attrs["original_filename"] = filename
    group.attrs["analysis_timestamp"] = dt.datetime.utcnow().isoformat() + "Z"
    group.attrs["code_version"] = __version__

    group.create_dataset("angle_deg", data=angle.astype("f4"), compression="gzip")
    group.create_dataset("intensity", data=intensity.astype("f4"), compression="gzip")

    fits_group = group.create_group("fits")
    fits_group.create_dataset(
        "tin_peaks",
        data=to_structured_array_fits(fits),
        compression="gzip",
    )

    peaks_group = group.create_group("peaks")
    peaks_group.create_dataset(
        "detected",
        data=to_structured_array_peaks(detected),
        compression="gzip",
    )

    summary_group = group.create_group("summary")
    for key, value in summary.items():
        summary_group.create_dataset(key, data=value)
    summary_group.attrs["note"] = (
        "Scherrer and Williamson-Hall sizes use instrument-corrected FWHM; "
        "uncorrected metrics are stored with *_uncorr names."
    )


def iter_csv_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob("*.csv")
    else:
        yield from root.glob("*.csv")


def analyze_files(
    root: Path,
    out_path: Path,
    recursive: bool,
    prominence: float,
    smooth_window: int,
    smooth_polyorder: int,
) -> None:
    logging.info("Writing results to %s", out_path)
    sample_ids: Dict[str, int] = {}
    with h5py.File(out_path, "w") as h5:
        for path in iter_csv_files(root, recursive):
            try:
                header, angle, intensity = parse_header_and_data(path)
                metadata = parse_filename_metadata(path.name)
            except Exception as exc:
                logging.error("Skipping %s: %s", path, exc)
                continue

            lambda_A = float(header.get("K-Alpha1 wavelength", "nan"))
            smooth_intensity = _smooth_for_peaks(intensity, smooth_window, smooth_polyorder)
            fits = analyze_tin_peaks(angle, intensity, lambda_A)
            detected = detect_other_peaks(angle, smooth_intensity, prominence)
            summary = summarize_peaks(fits, lambda_A)

            sample_id = Path(path).stem
            if sample_id in sample_ids:
                sample_ids[sample_id] += 1
                sample_id = f"{sample_id}_{sample_ids[sample_id]}"
            else:
                sample_ids[sample_id] = 0

            write_sample_group(
                h5,
                sample_id,
                metadata,
                header,
                angle,
                intensity,
                fits,
                detected,
                summary,
                path.name,
            )


def _example_csv(path: Path, peak_center: float, intensity_scale: float) -> None:
    angle = np.linspace(30, 80, 1500)
    noise = np.random.default_rng(0).normal(0, 5, size=angle.size)
    intensity = 50 + intensity_scale * np.exp(-0.5 * ((angle - peak_center) / 0.3) ** 2) + noise
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[Measurement conditions]\n")
        handle.write("Anode material,Cu\n")
        handle.write("K-Alpha1 wavelength,1.5406\n")
        handle.write("K-Alpha2 wavelength,1.5444\n")
        handle.write("Ratio K-Alpha2/K-Alpha1,0.5\n")
        handle.write("Monochromator used,YES\n")
        handle.write("[Scan points]\n")
        handle.write("Angle,Intensity\n")
        for ang, inten in zip(angle, intensity):
            handle.write(f"{ang:.4f},{inten:.4f}\n")


def run_self_check() -> int:
    """Generate two example CSV files and run analysis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        file1 = root / "20C_25Ar_1N2_2-2ubar_50min_20251215.csv"
        file2 = root / "550C_25Ar_1N2_2-2ubar_50min_20251231_note.csv"
        _example_csv(file1, 36.7, 500)
        _example_csv(file2, 42.5, 400)
        out_path = root / "xrd_results.h5"
        analyze_files(root, out_path, recursive=False, prominence=50.0, smooth_window=11, smooth_polyorder=3)
        if not out_path.exists():
            logging.error("Self-check failed: output file missing")
            return 1
    logging.info("Self-check completed successfully")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch analyze PANalytical/Empyrean XRD CSV files.")
    parser.add_argument(
        "--root",
        default=r"\\nas.ads.mwn.de\ga63raz\Desktop\SystOrdnerNachExperimenten\Res\AllResonators\Time_temp",
        help="Root directory containing CSV files.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output HDF5 path. Defaults to <root>/xrd_results.h5",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for CSV files.")
    parser.add_argument("--prominence", type=float, default=100.0, help="Peak prominence for detection.")
    parser.add_argument("--smooth-window", type=int, default=11, help="Savitzky-Golay window length.")
    parser.add_argument("--smooth-polyorder", type=int, default=3, help="Savitzky-Golay polynomial order.")
    parser.add_argument("--self-check", action="store_true", help="Run self-check with generated examples.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.self_check:
        return run_self_check()
    root = Path(args.root)
    out_path = Path(args.out) if args.out else root / "xrd_results.h5"
    analyze_files(
        root=root,
        out_path=out_path,
        recursive=args.recursive,
        prominence=args.prominence,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
