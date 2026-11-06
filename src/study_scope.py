"""
study_scope.py - Which studies each pipeline step uses.

data_fetch and data_extract use every study in cBio_studyId / pedcBio_studyId. The steps that
build the modeling data and models (data_process, data_model, graph_construct, graph_ai_model,
post_analysis, benchmark) can be limited to a subset per cancer type with

    modeling_studyId:
      "Wilms Tumor": "wt_target_2018_pub"
      "Glioma": "pbta_all"            # brain_cptac_2020 is fetched and extracted, but not modeled

A value is a study ID or a list of them, optionally as "portal:study_id". Without the key (or for
a cancer type not listed), every configured study is used, as before.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


def pipeline_cfg(path: Optional[str] = None) -> dict:
    """config/pipeline_config.yaml (or $CONFIG_PATH), for steps that read another config file."""
    p = path or os.getenv("CONFIG_PATH", "config/pipeline_config.yaml")
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _as_list(v) -> List[str]:
    if v is None or v == "":
        return []
    return [str(x).strip() for x in (v if isinstance(v, (list, tuple)) else [v])]


def configured_studies(cfg: dict, ct: str) -> List[Tuple[str, str]]:
    """(portal, study_id) for every study configured for a cancer type (cBio first, then PedcBio)."""
    out: List[Tuple[str, str]] = []
    for key, portal in (("cBio_studyId", "cbio"), ("pedcBio_studyId", "pedcbio")):
        out += [(portal, s) for s in _as_list((cfg.get(key) or {}).get(ct))]
    return out


def modeling_selection(cfg: dict, ct: str) -> Optional[List[str]]:
    """The modeling_studyId entries of a cancer type, or None when it is not restricted."""
    sel = _as_list((cfg.get("modeling_studyId") or {}).get(ct))
    return sel or None


def is_modeled(cfg: dict, ct: str, portal: str, study_id: str) -> bool:
    sel = modeling_selection(cfg, ct)
    return sel is None or study_id in sel or f"{portal}:{study_id}" in sel


def modeling_studies(cfg: dict, ct: str,
                     studies: Optional[Iterable[Tuple[str, str]]] = None) -> List[Tuple[str, str]]:
    """
    The (portal, study_id) pairs to model for a cancer type: `studies` (default: all configured)
    filtered by modeling_studyId. A modeling_studyId entry that is not configured is an error,
    so a typo does not silently drop the cancer type.
    """
    studies = list(studies) if studies is not None else configured_studies(cfg, ct)
    sel = modeling_selection(cfg, ct)
    if sel is None:
        return studies
    known = {s for _, s in studies} | {f"{p}:{s}" for p, s in studies}
    unknown = [x for x in sel if x not in known]
    if unknown:
        raise ValueError(f"modeling_studyId[{ct!r}] = {unknown} is not in cBio_studyId / pedcBio_studyId "
                         f"(configured: {', '.join(f'{p}:{s}' for p, s in studies) or 'none'})")
    return [(p, s) for p, s in studies if is_modeled(cfg, ct, p, s)]
