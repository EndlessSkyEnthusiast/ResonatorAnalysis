#!/usr/bin/env python3
"""Extract resonator fit parameters from PDF files into an HDF5 structure."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import h5py
from PyPDF2 import PdfReader

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


def store_fit_params(group: h5py.Group, fit: FitParams, source_pdf: Path, power_dbm: int) -> None:
    group.attrs["power_dbm"] = power_dbm
    group.attrs["source_pdf"] = str(source_pdf)
    for key in FIT_KEYS:
        group.attrs[key] = getattr(fit, key)


def process_experiment(base_path: Path, output_path: Path) -> None:
    experiments = [p for p in base_path.iterdir() if p.is_dir()]
    if not experiments:
        raise FileNotFoundError(f"No experiment folders found in {base_path}")

    with h5py.File(output_path, "w") as h5file:
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
                        resonator_group = temp_group.require_group(resonator_path.name)
                        for pdf_path in iter_pdf_files(resonator_path):
                            power_dbm = parse_power_dbm(pdf_path.name)
                            if power_dbm is None:
                                continue
                            pdf_text = extract_pdf_text(pdf_path)
                            fit = FitParams.from_text(pdf_text)
                            power_group = resonator_group.require_group(str(power_dbm))
                            store_fit_params(power_group, fit, pdf_path, power_dbm)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract resonator fit parameters into an HDF5 structure.",
    )
    parser.add_argument(
        "base_path",
        type=Path,
        help="Base directory containing experiment folders (e.g. 550C, Other, RT).",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output HDF5 file path.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    process_experiment(args.base_path, args.output_path)


if __name__ == "__main__":
    main()
