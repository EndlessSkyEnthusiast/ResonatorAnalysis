#!/usr/bin/env python3
"""Extract resonator fit parameters from PDF files into an HDF5 structure."""

from __future__ import annotations

import re
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import h5py
import numpy as np
from PyPDF2 import PdfReader

BASE_PATH = Path(
    r"\\nas.ads.mwn.de\ga63raz\Desktop\SystOrdnerNachExperimenten\Res\AllResonators\Time_temp"
)
OUTPUT_PATH = BASE_PATH / "AlleResonatorDaten.h5"
HBAR = 1.054571817e-34

FIT_KEYS = ("fr", "Ql", "Qc", "Qi", "Qi_err")
TEMP_REGEX = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(m?K)", re.IGNORECASE)
POWER_REGEX = re.compile(r"(-?\d+)")
RESONATOR_REGEX = re.compile(r"^Res\d+", re.IGNORECASE)
DATE_REGEX = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class FitParams:
    fr: float
    Ql: float
    Qc: float
    Qi: float
    Qi_err: float

    @classmethod
    def from_text(cls, text: str) -> "FitParams":
        values: Dict[str, float] = {}
        for key in FIT_KEYS:
            match = re.search(rf"\b{re.escape(key)}:\s*([0-9.+-eE]+)", text)
            if match:
                values[key] = float(match.group(1))
        missing = [key for key in FIT_KEYS if key not in values]
        if missing:
            raise ValueError(f"Missing fit params in PDF text: {', '.join(missing)}")
        return cls(**values)  # type: ignore[arg-type]


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_temperature(folder_name: str) -> Optional[str]:
    match = TEMP_REGEX.search(folder_name)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "mk":
        value /= 1000.0
    return f"{value:.4f}"


def parse_power_dbm(pdf_name: str) -> Optional[int]:
    match = POWER_REGEX.match(pdf_name)
    if not match:
        return None
    return int(match.group(1))


def iter_pdf_files(root: Path) -> Iterable[Path]:
    yield from root.glob("*.pdf")


def compute_n_photon(fit: FitParams, power_dbm: int) -> float:
    if fit.fr <= 0 or fit.Ql <= 0 or fit.Qc <= 0:
        return np.nan
    power_w = 10 ** ((power_dbm - 30) / 10)
    omega = 2 * math.pi * fit.fr
    return power_w * (fit.Ql**2 / fit.Qc) / (HBAR * omega**2)


def _parse_float_token(token: str, suffixes: Iterable[str]) -> Optional[float]:
    token_lower = token.lower()
    match_len: Optional[int] = None
    for suffix in suffixes:
        suffix_lower = suffix.lower()
        if token_lower.endswith(suffix_lower):
            match_len = len(suffix_lower)
            break
    if match_len is None:
        return None
    raw = token[: -match_len]
    raw = raw.replace(",", ".")
    raw = re.sub(r"(?<=\d)-(?=\d)", ".", raw)
    try:
        return float(raw)
    except ValueError:
        return None


def _extract_params(tokens: Iterable[str]) -> tuple[Dict[str, str], list[str], bool]:
    params: Dict[str, str] = {}
    comments: list[str] = []
    study = False
    for token in tokens:
        if token == "Study":
            study = True
            continue
        temp_c = _parse_float_token(token, ("C",))
        if temp_c is not None:
            params["temperature_c"] = f"{temp_c:.3f}"
            continue
        ar = _parse_float_token(token, ("Ar",))
        if ar is not None:
            params["argon_sccm"] = f"{ar:.3f}"
            continue
        n2 = _parse_float_token(token, ("N2",))
        if n2 is not None:
            params["nitrogen_sccm"] = f"{n2:.3f}"
            continue
        pressure = _parse_float_token(token, ("ubar", "mubar"))
        if pressure is not None:
            params["pressure_ubar"] = f"{pressure:.3f}"
            continue
        sputter = _parse_float_token(token, ("min",))
        if sputter is not None:
            params["sputter_min"] = f"{sputter:.3f}"
            continue
        if DATE_REGEX.match(token):
            params["date"] = token
            continue
        comments.append(token)
    return params, comments, study


def _chip_group_name(folder_path: Path) -> str:
    if "Other" in folder_path.parts:
        return f"Other/{folder_path.name}"
    return folder_path.name


def _iter_resonator_dirs(container: Path) -> Iterable[Path]:
    for entry in container.iterdir():
        if entry.is_dir() and RESONATOR_REGEX.match(entry.name):
            yield entry


def _iter_sweep_resonators(sweep_folder: Path) -> Iterable[Path]:
    for entry in sweep_folder.iterdir():
        if entry.is_dir():
            yield entry


def _list_unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _parse_measurement_temperature(folder_name: str) -> Optional[float]:
    match = TEMP_REGEX.search(folder_name)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "mk":
        value /= 1000.0
    return value


def _has_required_params(params: Dict[str, str]) -> bool:
    required = {
        "temperature_c",
        "argon_sccm",
        "nitrogen_sccm",
        "pressure_ubar",
        "sputter_min",
        "date",
    }
    return required.issubset(params.keys())


def _has_resonator_data(chip_path: Path) -> bool:
    sweep_path = chip_path / "full power sweep"
    if sweep_path.exists():
        if any(entry.is_dir() for entry in sweep_path.iterdir()):
            return True
    for entry in chip_path.iterdir():
        if not entry.is_dir():
            continue
        if RESONATOR_REGEX.match(entry.name):
            return True
        if _parse_measurement_temperature(entry.name) is not None:
            sweep_entry = entry / "full power sweep"
            if sweep_entry.exists() and any(
                child.is_dir() for child in sweep_entry.iterdir()
            ):
                return True
            if any(
                child.is_dir() and RESONATOR_REGEX.match(child.name)
                for child in entry.iterdir()
            ):
                return True
    return False



def store_fit_params(group: h5py.Group, fit: FitParams, source_pdf: Path, power_dbm: int) -> None:
    group.attrs["power_dbm"] = power_dbm
    group.attrs["source_pdf"] = str(source_pdf)
    for key in FIT_KEYS:
        group.attrs[key] = getattr(fit, key)
    group.attrs["n_photon"] = compute_n_photon(fit, power_dbm)


def _write_chip_metadata(group: h5py.Group, metadata: Dict[str, str]) -> None:
    for key, value in metadata.items():
        if key not in group.attrs:
            group.attrs[key] = value


def _process_resonators(
    resonator_dirs: Iterable[Path],
    temp_group: h5py.Group,
    experiment_label: str,
    unphysical_resonators: set[str],
) -> None:
    for resonator_path in resonator_dirs:
        if not resonator_path.is_dir():
            continue
        existing_resonator_group = temp_group.get(resonator_path.name)
        if existing_resonator_group is not None:
            continue
        resonator_group = temp_group.create_group(resonator_path.name)
        for pdf_path in iter_pdf_files(resonator_path):
            power_dbm = parse_power_dbm(pdf_path.name)
            if power_dbm is None:
                continue
            pdf_text = extract_pdf_text(pdf_path)
            fit = FitParams.from_text(pdf_text)
            if fit.Qi <= 0 or fit.Ql <= 0 or fit.Qc <= 0:
                resonator_id = f"{experiment_label}|{resonator_path.name}"
                unphysical_resonators.add(resonator_id)
            power_group = resonator_group.require_group(str(power_dbm))
            store_fit_params(power_group, fit, pdf_path, power_dbm)


def _process_chip_folder(
    chip_path: Path,
    chip_group: h5py.Group,
    unphysical_resonators: set[str],
) -> None:
    sweep_path = chip_path / "full power sweep"
    if sweep_path.exists():
        temp_group = chip_group.require_group("0.1000")
        temp_group.attrs.setdefault("measurement_temperature_k", 0.1)
        resonator_dirs = _iter_sweep_resonators(sweep_path)
        _process_resonators(
            resonator_dirs,
            temp_group,
            f"{chip_group.name}|0.1000",
            unphysical_resonators,
        )
        return

    temp_folders: list[tuple[Path, float]] = []
    for path in chip_path.iterdir():
        if not path.is_dir():
            continue
        temperature_k = _parse_measurement_temperature(path.name)
        if temperature_k is None:
            continue
        temp_folders.append((path, temperature_k))
    for temp_path, temperature_k in temp_folders:
        temp_key = f"{temperature_k:.4f}"
        temp_group = chip_group.require_group(temp_key)
        temp_group.attrs.setdefault("measurement_temperature_k", temperature_k)
        sweep_path = temp_path / "full power sweep"
        resonator_dirs = []
        if sweep_path.exists():
            resonator_dirs.extend(_iter_sweep_resonators(sweep_path))
        resonator_dirs.extend(_iter_resonator_dirs(temp_path))
        unique_resonators = _list_unique_paths(resonator_dirs)
        _process_resonators(
            unique_resonators,
            temp_group,
            f"{chip_group.name}|{temp_key}",
            unphysical_resonators,
        )


def process_experiment(base_path: Path, output_path: Path) -> None:
    experiments = [p for p in base_path.iterdir() if p.is_dir()]
    if not experiments:
        raise FileNotFoundError(f"No experiment folders found in {base_path}")

    unphysical_resonators: set[str] = set()
    unphysical_path = output_path.parent / "unphysical_resonators.json"
    if unphysical_path.exists():
        try:
            existing = json.loads(unphysical_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                unphysical_resonators.update(
                    entry for entry in existing if isinstance(entry, str)
                )
        except json.JSONDecodeError:
            pass

    with h5py.File(output_path, "a") as h5file:
        for experiment_path in experiments:
            if experiment_path.name == "Other":
                for chip_path in experiment_path.iterdir():
                    if not chip_path.is_dir():
                        continue
                    params, comments, study = _extract_params(chip_path.name.split("_"))
                    if study:
                        for subfolder in chip_path.iterdir():
                            if not subfolder.is_dir():
                                continue
                            sub_params, sub_comments, _ = _extract_params(
                                subfolder.name.split("_")
                            )
                            combined = {**params, **sub_params}
                            combined_comments = comments + sub_comments
                            if not _has_required_params(combined):
                                continue
                            chip_group = h5file.require_group(_chip_group_name(subfolder))
                            combined["category"] = "Other"
                            combined["study"] = "True"
                            combined["study_parent"] = chip_path.name
                            if combined_comments:
                                combined["comments"] = "_".join(combined_comments)
                            _write_chip_metadata(chip_group, combined)
                            _process_chip_folder(subfolder, chip_group, unphysical_resonators)
                            h5file.flush()
                    else:
                        if not _has_required_params(params):
                            continue
                        chip_group = h5file.require_group(_chip_group_name(chip_path))
                        params["category"] = "Other"
                        if comments:
                            params["comments"] = "_".join(comments)
                        _write_chip_metadata(chip_group, params)
                        _process_chip_folder(chip_path, chip_group, unphysical_resonators)
                        h5file.flush()
                continue
            params, comments, study = _extract_params(experiment_path.name.split("_"))
            if study:
                for subfolder in experiment_path.iterdir():
                    if not subfolder.is_dir():
                        continue
                    sub_params, sub_comments, _ = _extract_params(subfolder.name.split("_"))
                    combined = {**params, **sub_params}
                    combined_comments = comments + sub_comments
                    if not _has_required_params(combined) and not _has_resonator_data(subfolder):
                        continue
                    chip_group = h5file.require_group(_chip_group_name(subfolder))
                    combined["study"] = "True"
                    combined["study_parent"] = experiment_path.name
                    if combined_comments:
                        combined["comments"] = "_".join(combined_comments)
                    _write_chip_metadata(chip_group, combined)
                    _process_chip_folder(subfolder, chip_group, unphysical_resonators)
                    h5file.flush()
            else:
                subfolders = [p for p in experiment_path.iterdir() if p.is_dir()]
                has_study_subfolders = False
                for subfolder in subfolders:
                    sub_params, sub_comments, sub_study = _extract_params(
                        subfolder.name.split("_")
                    )
                    if not sub_study:
                        continue
                    has_study_subfolders = True
                    combined = {**params, **sub_params}
                    combined_comments = comments + sub_comments
                    if not _has_required_params(combined) and not _has_resonator_data(subfolder):
                        continue
                    chip_group = h5file.require_group(_chip_group_name(subfolder))
                    combined["study"] = "True"
                    combined["study_parent"] = experiment_path.name
                    if combined_comments:
                        combined["comments"] = "_".join(combined_comments)
                    _write_chip_metadata(chip_group, combined)
                    _process_chip_folder(subfolder, chip_group, unphysical_resonators)
                    h5file.flush()
                if has_study_subfolders:
                    continue
                if not _has_required_params(params) and not _has_resonator_data(experiment_path):
                    continue
                chip_group = h5file.require_group(_chip_group_name(experiment_path))
                if comments:
                    params["comments"] = "_".join(comments)
                _write_chip_metadata(chip_group, params)
                _process_chip_folder(experiment_path, chip_group, unphysical_resonators)
                h5file.flush()

    unphysical_path.write_text(
        json.dumps(sorted(unphysical_resonators), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    process_experiment(BASE_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
