"""
cohort.py: the modeling cohort written by data_process, and names for its classes.

data_process writes, for each modeled study, data/gdm/filtered/<portal>/<type>/<study_id>/
labels_train.csv and labels_test.csv (patientId, label 0/1): the patients with clinical data
and every configured omics modality. Read by scripts/cohort_table.py (cohort characteristics,
Table 2 of the manuscript) and by post_analysis (KM curves by the modeled class).

Class names come from processor.labels.<type>.class_names in pipeline_config.yaml
({0: FHWT, 1: DAWT} or "0=FHWT,1=DAWT"). Without it they are taken from the label map: the
single value mapped to a class, or "Low-grade" / "High-grade" when every value of the class
says so, else "label <k>".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

FILTERED_ROOT = Path("data/gdm/filtered")
FETCH_ROOT = Path("data/fetch/cbioportal_api_request")


def slug(cancer_type: str) -> str:
    """Folder name of a cancer type ("Wilms Tumor" -> "wilms_tumor"), as in every step."""
    return str(cancer_type).replace(" ", "_").lower()


def filtered_dir(portal: str, cancer_type: str, study_id: str, root: Path = FILTERED_ROOT) -> Path:
    return Path(root) / portal / slug(cancer_type) / study_id


def extract_dir(portal: str, cancer_type: str, study_id: str, root: Path = FETCH_ROOT) -> Path:
    return Path(root) / portal / slug(cancer_type) / study_id


def load_labels(portal: str, cancer_type: str, study_id: str, root: Path = FILTERED_ROOT) -> pd.DataFrame:
    """Modeling cohort: one row per patient with patientId, label (int) and split (train / test)."""
    d = filtered_dir(portal, cancer_type, study_id, root)
    parts = []
    for split in ("train", "test"):
        f = d / f"labels_{split}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found: run data_process first")
        y = pd.read_csv(f, index_col=0)
        parts.append(pd.DataFrame({"patientId": y.index.astype(str).str.strip(),
                                   "label": pd.to_numeric(y.iloc[:, 0]).astype(int).to_numpy(),
                                   "split": split}))
    return pd.concat(parts, ignore_index=True)


def _parse_pairs(value) -> Dict[str, int]:
    """Label map ("FHWT=0,DAWT=1" or a YAML mapping value -> 0/1) as {value: class}."""
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k).strip(): int(str(v).strip()) for k, v in value.items()}
    out = {}
    for tok in str(value).split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = int(v.strip())
    return out


def _grade_name(values: List[str]) -> str | None:
    for word, name in (("low", "Low-grade"), ("high", "High-grade")):
        if values and all(re.search(rf"\b{word}[- ]grade\b", v, re.I) for v in values):
            return name
    return None


def class_names(cfg: dict, cancer_type: str) -> Dict[int, str]:
    """{0: name, 1: name} for a cancer type (see the module docstring)."""
    lc = (((cfg or {}).get("processor") or {}).get("labels") or {}).get(cancer_type) or {}
    given = lc.get("class_names")
    if given:
        if isinstance(given, dict):
            return {int(k): str(v).strip() for k, v in given.items()}
        pairs = [tok.split("=", 1) for tok in str(given).split(",") if "=" in tok]   # "0=FHWT,1=DAWT"
        return {int(k): v.strip() for k, v in pairs}
    by_class: Dict[int, List[str]] = {0: [], 1: []}
    for value, k in _parse_pairs(lc.get("map")).items():
        by_class.setdefault(k, []).append(value)
    names = {}
    for k, values in by_class.items():
        names[k] = _grade_name(values) or (values[0] if len(values) == 1 else f"label {k}")
    return names
