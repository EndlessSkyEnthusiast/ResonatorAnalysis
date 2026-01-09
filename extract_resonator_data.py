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
    r"\\nas.ads.mwn.de\ga63raz\Desktop\AllResonators\25Ar1N2-2ubarTimedependentTempdependent"
)
OUTPUT_PATH = Path("resonators.h5")
HBAR = 1.054571817e-34

FIT_KEYS = ("fr", "Ql", "Qc", "Qi", "Qi_err")
TEMP_REGEX = re.compile(r"([0-9]+(?:\.[0-9]+)?)K", re.IGNORECASE)
POWER_REGEX = re.compile(r"(-?\d+)")


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
    return match.group(1)


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



def store_fit_params(group: h5py.Group, fit: FitParams, source_pdf: Path, power_dbm: int) -> None:
    group.attrs["power_dbm"] = power_dbm
    group.attrs["source_pdf"] = str(source_pdf)
    for key in FIT_KEYS:
        group.attrs[key] = getattr(fit, key)
    group.attrs["n_photon"] = compute_n_photon(fit, power_dbm)


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
            experiment_group = h5file.require_group(experiment_path.name)
            for chip_path in experiment_path.iterdir():
                if not chip_path.is_dir():
                    continue
                chip_group = experiment_group.require_group(chip_path.name)
                for temp_path in chip_path.iterdir():
                    if not temp_path.is_dir():
                        continue
                    temperature = parse_temperature(temp_path.name)
                    if temperature is None:
                        continue
                    temp_group = chip_group.require_group(temperature)
                    sweep_path = temp_path / "full power sweep"
                    if not sweep_path.exists():
                        continue
                    for resonator_path in sweep_path.iterdir():
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
                                resonator_id = (
                                    f"{experiment_path.name}|{chip_path.name}|"
                                    f"{temperature}|{resonator_path.name}"
                                )
                                unphysical_resonators.add(resonator_id)
                            power_group = resonator_group.require_group(str(power_dbm))
                            store_fit_params(power_group, fit, pdf_path, power_dbm)

    unphysical_path.write_text(
        json.dumps(sorted(unphysical_resonators), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    process_experiment(BASE_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
