# src/data_model.py
"""
data_model.py — Map cBio clinical to a GDC-aligned data model + fast DuckDB access

What it does
------------
1) Loads GDC dictionary schemas (YAML) from a local checkout, normalizes refs/anchors,
   and builds validators (strict or lite).
2) Projects cBio clinical CSVs to GDC-like node tables:
   - case           (submitter_id, project_id, disease_type, primary_site⋯)
   - sample         (submitter_id, preservation_method, specimen_type, tissue_type, tumor_descriptor⋯)
   - demographic    (submitter_id, sex_at_birth, vital_status, ethnicity, race)
   - diagnosis      (case_submitter_id, tumor_stage⋯)   [lenient; optional validation]
   - convenience link tables: links_sample_case, links_demographic_case
   - idmap (subject_id <-> sample_id) for graph joins
3) Registers feature stores (mrna/mirna/methylation.zarr) in a minimal 'file' node when present.
4) Creates TEMP VIEWs in DuckDB over the Parquets for instant cohorting/joins.

Inputs
------
- Clinical CSVs written by extractor:
    data/fetch/cbioportal_api_request/<Cancer_Type>/
      clinical_patients.csv, clinical_samples.csv
- Feature stores / indexes (optional; from extractor):
    data/gdm/features/*.zarr
    data/gdm/indexes/*_{rows,cols}.parquet
- Config: config/gdc_mapping.yaml
    project_id: WILMS_TUMOR_CBIO      # stamped on 'case' and 'file'
    schemas_dir: src/gdcdictionary/schemas
    gdm_root: data/gdm
    gdc_mapping:
      disease_type_map, primary_site_map, icdo_topography_map, icdo_morph_to_site_map, oncotree_to_site
      sample_suffix_map, preservation_map, preservation_default

Outputs (under gdm_root)
------------------------
clinical/
  clinical_subjects.parquet
  clinical_samples.parquet
nodes/
  case.parquet
  sample.parquet
  demographic.parquet
  diagnosis.parquet           # optional but written if derivable
  nodes.parquet               # convenience catalog of node ids
indexes/
  idmap.parquet
  links_sample_case.parquet
  links_demographic_case.parquet
files/
  file.parquet                # minimal 'file' node rows for features
  file_full.parquet           # richer analytics view with sizes/md5
features/
  mrna.zarr, mirna.zarr, methylation.zarr  # (if present; not created here)

DuckDB TEMP VIEWs (non-reserved names)
--------------------------------------
subjects            → clinical/clinical_subjects.parquet
samples             → clinical/clinical_samples.parquet
gdc_case            → nodes/case.parquet
gdc_demographic     → nodes/demographic.parquet
gdc_diagnosis       → nodes/diagnosis.parquet
links_sample_case   → indexes/links_sample_case.parquet
links_demographic_case → indexes/links_demographic_case.parquet
idmap, nodes, files, and feature index views if present

Validation modes
----------------
- Lite (default): relax required, allow additionalProperties. Great for iterative pipelines.
- Strict: set --strict or DM_GDC_STRICT=1 to enforce dictionary requirements.
  You may need to expand your mappings (e.g., preservation enums) to pass strict.

Environment
-----------
DM_CONFIG           : alternative config path (else CONFIG_PATH or default config/gdc_mapping.yaml)
CONFIG_PATH         : pipeline config (project_id, mappings, …)
DM_GDC_STRICT       : 1/0 to force strict/lite (CLI --strict overrides)
DM_DUCKDB_MODE      : memory | ro | rw   (default: memory)
DM_DUCKDB_PATH      : file path for persistent DB when mode=rw
DM_DUCKDB_RETRIES   : int retries on lock when mode=rw (default 0)
DM_DUCKDB_RETRY_SLEEP : seconds between retries (default 0.5)

CLI / Pipeline
--------------
- Recommended (pipeline):
    python -m src.main --call data_model
  (Runs `build_data_model()` for all cancer types listed in config and then attaches views.)

- Ad-hoc demo:
    python -m src.data_model \
      --schemas-dir src/gdcdictionary/schemas \
      --gdm-root data/gdm \
      --cbio-dir data/fetch/cbioportal_api_request/Wilms_Tumor \
      --project-id WILMS_TUMOR_CBIO

Ad-hoc SQL (via pipeline data_query)
------------------------------------
# Join case + diagnosis + demographic via link table; get a cohort’s sample_ids
PIPELINE_SQL=$'WITH cohort AS (
  SELECT c.submitter_id AS subject_id
  FROM gdc_case c
  LEFT JOIN gdc_diagnosis d
    ON d.case_submitter_id = c.submitter_id
  LEFT JOIN links_demographic_case ldc
    ON ldc.case_submitter_id = c.submitter_id
  LEFT JOIN gdc_demographic demo
    ON demo.submitter_id = ldc.demographic_submitter_id
  WHERE upper(coalesce(d.tumor_stage, '''')) LIKE ''III%''
    AND lower(coalesce(demo.sex_at_birth, '''')) = ''female''
)
SELECT DISTINCT i.sample_id
FROM cohort c
JOIN idmap i USING (subject_id)
ORDER BY 1' \
python -m src.main --call data_query

Notes & tips
------------
- We avoid the reserved word "case" for view names (use gdc_case).
- When writing the 'file' node for features, we compute md5 and byte sizes; state defaults to "validated".
- Schema normalization injects common anchors (uuid/state/project_id) and strips external refs so the validator
  works offline from your local YAML checkout.
"""

from __future__ import annotations
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional, Union
import json
import yaml
import duckdb
import pandas as pd
import numpy as np

# JSON Schema (local, no internet). RefResolver is deprecated but works;
# you can swap to "referencing" later if you prefer.
import copy
from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7, DRAFT4
from datetime import datetime, timezone
import uuid
# ==============================================================================
# Utilities. & Helpers
# ==============================================================================

def _plain(vv):
    # normalize pandas/NumPy types to plain Python
    import numpy as np  # local import to avoid hard dep here
    if vv is None:
        return None
    if vv is pd.NA:
        return None
    if isinstance(vv, float) and pd.isna(vv):
        return None
    if isinstance(vv, pd.Timestamp):
        return vv.isoformat()
    if isinstance(vv, np.generic):
        return vv.item()
    if isinstance(vv, dict):
        return {k: _plain(x) for k, x in vv.items()}
    if isinstance(vv, (list, tuple)):
        return [_plain(x) for x in vv]
    return vv

def _md5_file(path: Path, chunk=1024*1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _dir_size_and_manifest_md5(dir_path: Path) -> tuple[int, str]:
    """
    For a directory (e.g., Zarr), compute total size and a stable MD5:
    md5 of a text manifest whose lines are "<relpath>\t<file_md5>\t<size>"
    sorted by relpath.
    """
    entries = []
    total = 0
    for root, _, files in os.walk(dir_path):
        for fn in files:
            p = Path(root) / fn
            rel = str(p.relative_to(dir_path))
            md5 = _md5_file(p)
            sz = p.stat().st_size
            total += sz
            entries.append(f"{rel}\t{md5}\t{sz}")
    entries.sort()
    manifest = "\n".join(entries).encode("utf-8")
    return total, hashlib.md5(manifest).hexdigest()

def _stat_and_md5(path: Path) -> tuple[int, str]:
    if path.is_file():
        return path.stat().st_size, _md5_file(path)
    if path.is_dir():
        return _dir_size_and_manifest_md5(path)
    raise FileNotFoundError(path)

def _gdc_file_state() -> str:
    # conservative, recognized state for demo content
    return "validated"

def _stable_uuid5(name: str, ns: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    """Deterministic UUID from a stable string (e.g., submitter_id)."""
    return str(uuid.uuid5(ns, name))

def _subject_from_sample(sample_id: str) -> str:
    # TARGET-50-CAAAAC-01 -> TARGET-50-CAAAAC
    return sample_id.rsplit("-", 1)[0]

def _env_flag(name: str, default: bool=False) -> bool:
    v = os.getenv(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","yes","on")

def _pick_first_notnull(*series_list):
    for s in series_list:
        if s is not None:
            v = pd.Series(s)
            if v.notna().any():
                return v
    return None

def _normalize_c_code(x: pd.Series) -> pd.Series:
    return x.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)

def _parse_morph_code(x: pd.Series) -> pd.Series:
    # "8960/3" -> "8960"
    return x.astype(str).str.extract(r"^(\d{4})").iloc[:, 0]

def _samples_subject_ids(samples_df: pd.DataFrame) -> pd.Series:
    """
    Return subject IDs from samples_df.
    Priority:
      1) 'subject_id' column if present
      2) derive from 'sampleId' by stripping a trailing '-<digits>' (e.g., '-01', '-02')
    """
    if "subject_id" in samples_df.columns:
        return samples_df["subject_id"].astype(str)

    if "sampleId" in samples_df.columns:
        # Robust derivation: remove final dash-number suffix if present
        s = samples_df["sampleId"].astype(str)
        # e.g., "TARGET-50-CAAAAC-01" -> "TARGET-50-CAAAAC"
        return s.str.replace(r"-\d+$", "", regex=True)

    # Helpful error to surface schema mismatch
    raise KeyError(
        "samples_df must contain either 'subject_id' or 'sampleId'. "
        f"Available columns: {list(samples_df.columns)}"
    )

def _patients_subject_ids(patients_df: pd.DataFrame) -> pd.Series:
    """Accept either 'patientId' (cBio) or 'submitter_id' (post-rename)."""
    if "patientId" in patients_df.columns:
        return patients_df["patientId"].astype(str)
    if "submitter_id" in patients_df.columns:
        return patients_df["submitter_id"].astype(str)
    raise KeyError(
        "patients_df must contain 'patientId' or 'submitter_id'. "
        f"Available columns: {list(patients_df.columns)}"
    )

def _derive_primary_site(patients_df: pd.DataFrame,
                         samples_df: pd.DataFrame,
                         cfg: dict) -> pd.Series:
    gm = (cfg or {}).get("gdc_mapping", {})
    topo_map  = gm.get("icdo_topography_map", {})
    morph_map = gm.get("icdo_morph_to_site_map", {})
    onco_map  = gm.get("oncotree_to_site", {})
    ctype_map = gm.get("primary_site_map", {})

    pid = _patients_subject_ids(patients_df)
    out = pd.Series(index=pid, dtype=object)

    # 1) Prefer sample topography (ICD_0_3_T)
    samp = samples_df.copy()
    samp["__subject_id"] = _samples_subject_ids(samp)
    if "ICD_0_3_T" in samp.columns:
        s_topo = _normalize_c_code(samp["ICD_0_3_T"])
        top_by_subj = (
            samp.assign(C=s_topo)
                .dropna(subset=["C"])
                .groupby("__subject_id")["C"]
                .agg(lambda x: x.value_counts().idxmax())
        )
        out = out.combine_first(top_by_subj.map(topo_map))

    # 2) Patient-level topography (if present)
    for col in ("ICD_0_3_T", "ICD_O_3_T", "ICD_O_3_TOPO"):
        if col in patients_df.columns:
            mapped = _normalize_c_code(patients_df[col]).map(topo_map)
            out = out.where(out.notna(), pd.Series(mapped.values, index=pid))

    # 3) OncoTree fallback (from samples)
    if "ONCOTREE_CODE" in samp.columns:
        onco_first = samp.dropna(subset=["ONCOTREE_CODE"]).groupby("__subject_id")["ONCOTREE_CODE"].agg("first")
        out = out.combine_first(onco_first.map(onco_map))

    # 4) Morphology fallback (patients ICD_O_3_SITE like "8960/3")
    if "ICD_O_3_SITE" in patients_df.columns:
        morph = _parse_morph_code(patients_df["ICD_O_3_SITE"])
        out = out.where(out.notna(), pd.Series(morph.map(morph_map).values, index=pid))

    # 5) Cancer-type fallback
    ctype = _pick_first_notnull(patients_df.get("CANCER_TYPE_DETAILED"), patients_df.get("CANCER_TYPE"))
    if ctype is not None:
        out = out.where(out.notna(), pd.Series(ctype.astype(str).map(ctype_map).values, index=pid))

    return out.fillna("Unknown")

def _derive_disease_type(
    patients_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    cfg: dict
) -> pd.Series:
    """
    Prefer CANCER_TYPE_DETAILED / CANCER_TYPE from samples_df (rolled up to subject),
    else fall back to patients_df if present, else leave 'Unknown'.
    Then map via gdc_mapping.disease_type_map (if provided).
    """
    gm = (cfg or {}).get("gdc_mapping", {})
    dtype_map = gm.get("disease_type_map", {})

    # 1) Gather per-subject cancer type from samples if available
    subj_from_sample = _samples_subject_ids(samples_df).rename("__subject_id")
    s = pd.Series(index=pd.Index([], name="__subject_id", dtype="object"), dtype=object)

    # choose detailed then basic
    for col in ("CANCER_TYPE_DETAILED", "CANCER_TYPE"):
        if col in samples_df.columns:
            tmp = (
                samples_df
                .assign(__subject_id=_samples_subject_ids(samples_df))
                .dropna(subset=[col])
                .groupby("__subject_id")[col].agg("first")
            )
            if not tmp.empty:
                s = tmp.astype(str)
                break

    # 2) Fallback: try patients_df (rare in cBio clinical_patients)
    if s.empty:
        src = _pick_first_notnull(patients_df.get("CANCER_TYPE_DETAILED"),
                                  patients_df.get("CANCER_TYPE"))
        if src is not None:
            subj = _patients_subject_ids(patients_df)
            s = pd.Series(src.values, index=subj, dtype=object)

    # 3) Build output aligned to patients_df order
    out_index = _patients_subject_ids(patients_df)
    if s.empty:
        out = pd.Series(["Unknown"] * len(patients_df), index=out_index, dtype=object)
    else:
        out = s.reindex(out_index).fillna("Unknown").astype(str)

    # 4) Apply mapping
    return out.map(dtype_map).fillna(out)

def _derive_sex_at_birth(patients_df: pd.DataFrame) -> pd.Series:
    # prefer SEX, fallback to GENDER
    src = None
    for col in ("SEX", "GENDER", "gender"):
        if col in patients_df.columns:
            src = patients_df[col]
            break
    if src is None:
        return pd.Series(["unknown"] * len(patients_df), index=patients_df.index, dtype=object)
    s = src.astype(str).str.strip().str.lower()
    out = pd.Series("unknown", index=s.index, dtype=object)
    out = out.mask(s.str.startswith("f"), "female")
    out = out.mask(s.str.startswith("m"), "male")
    out = out.mask(s.str.contains("not reported"), "not reported")
    out = out.mask(s.str.contains("unknown"), "unknown")
    return out.fillna("unknown")

def _derive_vital_status(patients_df: pd.DataFrame) -> pd.Series:
    # cBioPortal OS_STATUS examples: "1:DECEASED", "0:LIVING"
    s = patients_df.get("OS_STATUS")
    if s is None:
        return pd.Series(["Unknown"] * len(patients_df), index=patients_df.index, dtype=object)
    u = s.astype(str).str.upper()
    out = pd.Series("Unknown", index=u.index, dtype=object)
    out = out.mask(u.str.contains("DECEASED", na=False), "Dead")
    out = out.mask(u.str.contains("LIVING",   na=False), "Alive")
    # allow Unknown/Not Reported per GDC enum
    return out.fillna("Unknown")

def _norm_ethnicity(x: pd.Series) -> pd.Series:
    """
    Map cBio 'ETHNICITY' to GDC demographic.ethnicity.
    Target enums (typical GDC): 
      {'hispanic or latino','not hispanic or latino','unknown','not reported'}
    """
    if x is None:
        return pd.Series(["unknown"] * 0, dtype=object)
    s = x.astype(str).str.strip().str.lower()

    out = pd.Series("not reported", index=s.index, dtype=object)
    out = out.mask(s.str.contains(r"\bhispanic\b", regex=True), "hispanic or latino")
    out = out.mask(s.str.contains(r"not\s*hispanic|non[-\s]*hispanic", regex=True), "not hispanic or latino")
    out = out.mask(s.str.contains("unknown|undetermined|missing", regex=True), "unknown")
    out = out.mask(s.str.contains("not reported|n/?a|prefer not", regex=True), "not reported")
    return out.fillna("unknown")

def _norm_race(x: pd.Series) -> pd.Series:
    """
    Map cBio 'RACE' to GDC demographic.race.
    Target enums (typical GDC):
      {'white','black or african american','asian',
       'american indian or alaska native','native hawaiian or other pacific islander',
       'other','unknown','not reported'}
    """
    if x is None:
        return pd.Series(["unknown"] * 0, dtype=object)
    s = x.astype(str).str.strip().str.lower()

    # handle multi-valued entries like "white;asian" by taking first token
    s = s.str.split(r"[;,/]| and ", regex=True).str[0].str.strip()

    out = pd.Series("not reported", index=s.index, dtype=object)
    out = out.mask(s.str.contains(r"\bwhite\b"), "white")
    out = out.mask(s.str.contains(r"black|african"), "black or african american")
    out = out.mask(s.str.contains(r"\basian\b"), "asian")
    out = out.mask(s.str.contains(r"american indian|alaska"), "american indian or alaska native")
    out = out.mask(s.str.contains(r"hawaiian|pacific"), "native hawaiian or other pacific islander")
    out = out.mask(s.str.contains(r"two or more|multiple|multiracial|biracial"), "other")
    out = out.mask(s.str.contains(r"other"), "other")
    out = out.mask(s.str.contains(r"unknown|undetermined|missing"), "unknown")
    out = out.mask(s.str.contains(r"not reported|n/?a|prefer not"), "not reported")
    return out.fillna("unknown")

def _sample_suffix(series: pd.Series) -> pd.Series:
    # "TARGET-50-CAAAAC-01" -> "01"
    return series.astype(str).str.extract(r"-([0-9]{2})$", expand=True).iloc[:, 0]

def _derive_preservation_method(samples_df: pd.DataFrame, cfg: dict) -> pd.Series:
    gm = (cfg or {}).get("gdc_mapping", {})
    pmap = {str(k).lower(): v for k, v in gm.get("preservation_map", {}).items()}
    default = gm.get("preservation_default", "Unknown")
    col = next((c for c in ("PRESERVATION_METHOD","preservation_method","PRESERVATION","SAMPLE_PRESERVATION")
                if c in samples_df.columns), None)
    if col is None:
        return pd.Series(default, index=samples_df.index, dtype=object)
    return (samples_df[col].astype(str).str.strip().str.lower()
            .map(pmap).fillna(default).astype(object))

def _derive_sample_fields(samples_df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (specimen_type, tissue_type, tumor_descriptor), aligned to your schema enums.
    """
    gm = (cfg or {}).get("gdc_mapping", {})
    raw = gm.get("sample_suffix_map", {}) or {}
    sfx_map = {str(k).zfill(2): (v or {}) for k, v in raw.items()}

    sfx = _sample_suffix(samples_df["sampleId"]).fillna("").astype(str).str.zfill(2)
    overrides = pd.DataFrame([sfx_map.get(k, {}) for k in sfx.tolist()], index=samples_df.index)

    # Safe defaults for pediatric solid primaries
    specimen_type    = pd.Series("Solid Tissue",  index=samples_df.index, dtype=object)   # material
    tissue_type      = pd.Series("Tumor",         index=samples_df.index, dtype=object)   # context
    tumor_descriptor = pd.Series("Primary",       index=samples_df.index, dtype=object)   # disease descriptor

    if not overrides.empty:
        if "specimen_type" in overrides:    specimen_type    = overrides["specimen_type"].combine_first(specimen_type).astype(object)
        if "tissue_type" in overrides:      tissue_type      = overrides["tissue_type"].combine_first(tissue_type).astype(object)
        if "tumor_descriptor" in overrides: tumor_descriptor = overrides["tumor_descriptor"].combine_first(tumor_descriptor).astype(object)

    return specimen_type, tissue_type, tumor_descriptor

def _clinical_gender_from_sex(x: Optional[pd.Series]) -> pd.Series:
    if x is None:
        return pd.Series(["unspecified"] * 0, dtype=object)
    s = x.astype(str).str.strip().str.lower()
    out = pd.Series("unspecified", index=s.index, dtype=object)
    out = out.mask(s.str.startswith("f"), "female")
    out = out.mask(s.str.startswith("m"), "male")
    out = out.mask(s.str.contains("unknown"), "unknown")
    return out

def _clinical_ethnicity_2enums(x: Optional[pd.Series]) -> pd.Series:
    # clinical.yaml allows only {hispanic or latino, not hispanic or latino}
    if x is None:
        return pd.Series(["not hispanic or latino"] * 0, dtype=object)
    s = x.astype(str).str.strip().str.lower()
    out = pd.Series("not hispanic or latino", index=s.index, dtype=object)
    out = out.mask(s.str.contains(r"\bhispanic\b"), "hispanic or latino")
    return out

def _clinical_vital_status_from_os_status(x: Optional[pd.Series]) -> pd.Series:
    # "1:DECEASED" -> "dead"; "0:LIVING" -> "alive"
    if x is None:
        return pd.Series(["alive"] * 0, dtype=object)  # safe default
    u = x.astype(str).str.upper()
    out = pd.Series("alive", index=u.index, dtype=object)
    out = out.mask(u.str.contains("DECEASED", na=False), "dead")
    out = out.mask(u.str.contains("LIVING",   na=False), "alive")
    # you can add 'lost to follow-up' if you have a signal
    return out

def _clinical_days_to_death(os_days: Optional[pd.Series], os_status: Optional[pd.Series]) -> pd.Series:
    if os_days is None or os_status is None:
        return pd.Series([None] * 0, dtype=object)
    dead = os_status.astype(str).str.upper().str.contains("DECEASED", na=False)
    days = pd.to_numeric(os_days, errors="coerce")
    out = pd.Series([None] * len(days), index=days.index, dtype=object)
    out[dead] = days[dead]
    return out

# ==============================================================================
# Metadata helpers
# ==============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _write_json(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ==============================================================================
# GDC Schema Registry & Validation
# ==============================================================================

class GDCSchemaRegistry:
    """
    Loads GDC dictionary YAMLs and returns *normalized* node schemas ready for validation.
    Normalization steps:
      - Resolve odd $ref forms (lists -> string; external "_terms.yaml" refs -> permissive type).
      - Inject commonly-referenced anchors (uuid/state/datetime/project_id/...) from metaschema/_definitions.
      - Reconcile 'required' vs 'properties' (ensure required keys exist in properties).
    Exposes:
      - node_schema(node): normalized node schema (dict)
      - allowed_properties(node): set of property names considered valid for the node
      - required_fields_for(node): list of required keys
      - enums_for(node): {prop: [enum values]} map
      - validator(node, mode, observed_keys): a jsonschema validator
      - validate_df(node, df, mode, observed_keys): returns [(row_index, message), ...]
    Modes:
      - strict: use schema as-is
      - lite:   ignore 'required', set additionalProperties=True (for project-internal pipelines)
    """

    def __init__(self, schemas_dir: str | Path):
        self.schemas_dir = Path(schemas_dir)
        # core side files
        self._metaschema   = self._load_yaml("metaschema.yaml") or {}
        self._definitions  = self._load_yaml("_definitions.yaml") or {}
        self._terms        = self._load_yaml("_terms.yaml") or {}
        self._terms_enum   = self._load_yaml("_terms_enum.yaml") or {}
        # per-node cache (post-normalized)
        self._node_cache: Dict[str, Dict[str, Any]] = {}

    # ---------- file loading ----------
    def _load_yaml(self, name: str) -> dict | list | None:
        p = self.schemas_dir / name
        if not p.exists():
            return None
        with open(p, "r") as f:
            return yaml.safe_load(f)

    def _load_node_schema(self, node: str) -> dict:
        """Raw node schema dict from <schemas_dir>/<node>.yaml"""
        raw = self._load_yaml(f"{node}.yaml")
        if not isinstance(raw, dict):
            raise FileNotFoundError(
                f"Schema for node '{node}' not found or not a dict at {self.schemas_dir}/{node}.yaml"
            )
        return copy.deepcopy(raw)

    # ---------- anchor helpers ----------
    def _collect_local_anchor_refs(self, obj: Any) -> set[str]:
        """Find names referenced as $ref: '#/name' (or list of such) anywhere in a schema dict."""
        out: set[str] = set()
        if isinstance(obj, dict):
            ref = obj.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                out.add(ref[2:].split("/", 1)[0])
            elif isinstance(ref, list):
                for r in ref:
                    if isinstance(r, str) and r.startswith("#/"):
                        out.add(r[2:].split("/", 1)[0])
            for v in obj.values():
                out |= self._collect_local_anchor_refs(v)
        elif isinstance(obj, list):
            for v in obj:
                out |= self._collect_local_anchor_refs(v)
        return out

    def _lookup_anchor(self, name: str):
        for src in (self._metaschema, self._definitions):
            if not isinstance(src, dict):
                continue
            if name in src:
                return self._normalize_anchor_obj(src[name])
            for k in ("definitions", "$defs"):
                if k in src and isinstance(src[k], dict) and name in src[k]:
                    return self._normalize_anchor_obj(src[k][name])
        return None

    def _inject_common_anchors(self, s: dict) -> dict:
        """Copy missing anchors (uuid/state/datetime/project_id/…) into the node schema if referenced locally."""
        needed = self._collect_local_anchor_refs(s)
        if not needed:
            return s
        for name in needed:
            if name in s:
                continue  # already present in node
            found = self._lookup_anchor(name)
            if found is not None:
                s[name] = found
        return s

    # ---------- $ref massaging ----------
    def _massage_refs(self, obj: Any, parent_key: Optional[str] = None) -> None:
        """
        Normalize $ref forms in-place to avoid jsonschema resolver issues:

        - If '$ref' is a list, keep the first string element.
        - If a dict under 'properties' is literally {'$ref': <...>}, DROP the ref and
        keep an empty properties map ({}). This matches how some GDC nodes embed
        `_definitions.yaml#/ubiquitous_properties`.
        - For external refs (strings starting with '_', e.g. '_definitions.yaml#/to_one'):
        replace them with a permissive schema. By default we use {} (accept-any).
        For well-known link anchors ('to_one', 'to_many') we coerce to a minimal
        object schema to keep things a bit saner in strict runs.
        - Recurse into nested dicts and lists.
        """
        if isinstance(obj, dict):
            # Special: if we're sitting directly under "properties" and the entire
            # value is just {"$ref": ...}, keep an empty map for properties.
            if parent_key == "properties" and "$ref" in obj and len(obj) == 1:
                obj.clear()
                return
            if "$ref" in obj:
                ref = obj["$ref"]
                # Case: $ref provided as a list -> keep the first string
                if isinstance(ref, list):
                    first_str = next((r for r in ref if isinstance(r, str)), None)
                    if first_str is None:
                        obj.pop("$ref", None)
                    else:
                        obj["$ref"] = first_str
                    ref = obj.get("$ref")
                # Case: external refs to _definitions.yaml / _terms.yaml, etc.
                if isinstance(ref, str) and ref.startswith("_"):
                    # Be permissive by default
                    obj.pop("$ref", None)
                    # Small, safe heuristics for common link anchors:
                    #   _definitions.yaml#/to_one
                    #   _definitions.yaml#/to_many
                    # These are objects in GDC; keep it minimally typed.
                    lower = ref.lower()
                    if "to_one" in lower or "to_many" in lower:
                        obj.setdefault("type", "object")
                        # we intentionally do NOT restrict properties/required here
                        # to avoid over-constraining without full ref resolution
                    else:
                        # Accept-any (empty dict) is the safest fallback
                        # (do nothing; the ref was already popped)
                        pass
            # Recurse with knowledge of the current key
            for k, v in list(obj.items()):
                self._massage_refs(v, parent_key=k)

        elif isinstance(obj, list):
            for v in obj:
                self._massage_refs(v, parent_key=parent_key)

    def _sanitize_schema(self, s: dict) -> None:
        """
        Coerce positions that must be schemas into dicts.
        - properties: each value must be a dict
        - items / additionalProperties / not: must be dict if present
        - allOf/anyOf/oneOf: elements must be dicts
        """
        if not isinstance(s, dict):
            return

        # Apply sanitization logic to ALL keys that map names to schemas, not just 'properties'.
        for map_key in ("properties", "definitions", "$defs", "patternProperties"):
            schema_map = s.get(map_key)
            if isinstance(schema_map, dict):
                for k, v in list(schema_map.items()):
                    if isinstance(v, str):
                        schema_map[k] = {"type": "string"}
                    elif isinstance(v, list):
                    # rare/invalid; coerce to anyOf of string
                        schema_map[k] = {"anyOf": [{"type": "string"}]}
                    elif not isinstance(v, (dict, bool)): # bools (True/False) are valid schemas
                        schema_map[k] = {"type": "string"}

        for key in ("items", "additionalProperties", "not"):
            if key in s and not isinstance(s[key], dict):
                s[key] = {"type": "string"}

        for key in ("items", "additionalProperties", "not"):
            if key in s:
                val = s[key]
                if key == "items" and isinstance(val, list):
                    # This is tuple validation. Skip it here;
                    # it will be handled by the schema-list loop below.
                    continue
                # For 'not', 'additionalProperties', or 'items' (when not a list),
                # the value must be a valid schema (dict or bool).
                if not isinstance(val, (dict, bool)):
                    # Coerce any invalid value (like a string) to a basic schema.
                    s[key] = {"type": "string"}

        for key in ("allOf", "anyOf", "oneOf", "items"):
            arr = s.get(key)
            if isinstance(arr, list):
                new = []
                for el in arr:
                    if isinstance(el, dict):
                        new.append(el)
                    else:
                        new.append({"type": "string"})
                s[key] = new

        # Recurse
        for k, v in list(s.items()):
            if isinstance(v, dict):
                self._sanitize_schema(v)
            elif isinstance(v, list):
                for el in v:
                    if isinstance(el, dict):
                        self._sanitize_schema(el)

    # ---------- required vs properties ----------
    def _ensure_required_have_props(self, s: dict) -> dict:
        """
        If a key appears in `required` but not under `properties`, make it a property.
        Prefer referencing an in-schema anchor or a metaschema/_definitions anchor.
        Fallback to {'type':'string'}.
        """
        props = s.setdefault("properties", {})
        req = s.get("required", []) or []
        for k in req:
            if k in props:
                continue
            # ALWAYS prefer the global definition (from _definitions.yaml), as the local def (s[k]) is often a junk string.
            anchor = self._lookup_anchor(k) # Finds AND normalizes the REAL schema
            if anchor is not None:
                s[k] = anchor   # Inject the REAL, normalized schema (overwriting local junk string)
                props[k] = {"$ref": f"#/{k}"}
            elif k in s:
                # Fallback: No global def, trust and normalize the local def
                s[k] = self._normalize_anchor_obj(s[k])
                props[k] = {"$ref": f"#/{k}"}
            else:
                # Fallback: No def found anywhere
                props[k] = {"type": "string"}
        return s

    # ---------- public API ----------
    def _ensure_system_have_props(self, s: dict) -> dict:
        """
        Ensure all systemProperties (e.g., 'project_id', 'state', 'created_datetime')
        also exist under 'properties' so strict validation won't reject them when present.
        """
        props = s.setdefault("properties", {})
        sysprops = s.get("systemProperties", []) or []
        for k in sysprops:
            if k in props:
                continue
            # ALWAYS prefer the global definition
            anchor = self._lookup_anchor(k) # Finds AND normalizes the REAL schema
            if anchor is not None:
                s[k] = anchor   # Inject the REAL, normalized schema
                props[k] = {"$ref": f"#/{k}"}
            elif k in s:
                # Fallback: No global def, trust and normalize the local def
                s[k] = self._normalize_anchor_obj(s[k])
                props[k] = {"$ref": f"#/{k}"}
            else:
                # Fallback: No def found anywhere
                props[k] = {"type": "string"}
        return s

    def node_schema(self, node: str) -> dict:
        """
        Return a normalized node schema ready to use in validation:
        - local anchors injected
        - odd $ref normalized
        - required fields guaranteed to exist under 'properties'
        """
        if node not in self._node_cache:
            s = self._load_node_schema(node)
            self._massage_refs(s)
            s = self._inject_common_anchors(s)
            s = self._ensure_required_have_props(s)
            s = self._ensure_system_have_props(s)   
            self._massage_refs(s)  
            self._sanitize_schema(s)                  
            self._node_cache[node] = s
        return self._node_cache[node]

    def allowed_properties(self, node: str) -> set[str]:
        s = self.node_schema(node)
        props = set((s.get("properties") or {}).keys())
        props |= set(s.get("systemProperties") or [])
        props |= set(s.get("required") or [])
        return props

    def required_fields_for(self, node: str) -> List[str]:
        s = self.node_schema(node)
        req = s.get("required", []) or []
        # ensure unique strings, stable order
        return list(dict.fromkeys([str(x) for x in req]))

    def enums_for(self, node: str) -> Dict[str, List[Any]]:
        """
        Return {prop: enum_list} for properties that declare an 'enum'.
        """
        s = self.node_schema(node)
        out: Dict[str, List[Any]] = {}
        props = s.get("properties") or {}
        for k, v in props.items():
            if isinstance(v, dict) and "enum" in v and isinstance(v["enum"], list):
                out[k] = v["enum"]
        return out

    # ---------- validation ----------
    def _schema_for_mode(
        self,
        node: str,
        mode: str = "lite",
        observed_keys: Optional[Iterable[str]] = None,
    ) -> dict:
        """
        Return a *copy* of the node schema tweaked for the requested mode:
          - strict: as-is (normalized)
          - lite:   required=[], additionalProperties=True
        If observed_keys is provided, restrict 'properties' to those keys
        plus any required & systemProperties (to avoid accidental drops).
        """
        base = copy.deepcopy(self.node_schema(node))

        if mode not in ("strict", "lite"):
            raise ValueError(f"Unknown validation mode: {mode}")

        if mode == "lite":
            # relax global constraints
            base["additionalProperties"] = True
            base["required"] = []

        if observed_keys is not None:
            observed = set(map(str, observed_keys))
            required = set(map(str, base.get("required", []) or []))
            sysprops = set(map(str, base.get("systemProperties", []) or []))
            keep = observed | required | sysprops
            props = base.get("properties", {}) or {}
            new_props = {k: v for k, v in props.items() if k in keep}
            base["properties"] = new_props

        return base

    def validator(
        self,
        node: str,
        mode: str = "lite",
        observed_keys: Optional[Iterable[str]] = None,
    ) -> Draft7Validator:
        """
        Build a jsonschema Draft7 validator for the given node / mode.
        We avoid using RefResolver to sidestep deprecation warnings; references are
        normalized into the schema by node_schema().
        """
        schema = self._schema_for_mode(node, mode=mode, observed_keys=observed_keys)
        return Draft7Validator(schema)

    def validate_df(
        self,
        node: str,
        df: pd.DataFrame,
        mode: str = "lite",
        observed_keys: Optional[Iterable[str]] = None,
    ) -> List[Tuple[int, str]]:
        """
        Validate a DataFrame row-by-row against the node schema (per mode).
        Returns a list of (row_index, error_message). Empty list means OK.

        Behavior:
          - If observed_keys is provided, we pass only those keys (+ required/sysprops)
            to the validator (so extra columns in df won't trigger additionalProperties).
          - In strict mode without observed_keys, you should have pre-pruned df to allowed
            properties, otherwise additionalProperties may fire depending on the schema.
        """
        v = self.validator(node, mode=mode, observed_keys=observed_keys)

        # build column subset per observed_keys if provided
        if observed_keys is not None:
            required = set(self.required_fields_for(node))
            sysprops = set(self.node_schema(node).get("systemProperties") or [])
            cols = [c for c in df.columns if c in (set(observed_keys) | required | sysprops)]
            work = df.loc[:, cols]
        else:
            work = df

        errs: List[Tuple[int, str]] = []
        # iterate preserving original index values
        for i, rec in work.fillna(value=pd.NA).to_dict(orient="index").items():
            tmp = {k: _plain(vv) for k, vv in rec.items()}
            clean = {k: v for k, v in tmp.items() if v is not None}  # drop null keys
            for e in v.iter_errors(clean):
                errs.append((i, e.message))
        return errs

    def _normalize_anchor_obj(self, obj) -> dict:
        """Return a schema dict for an anchor, never a bare string/list."""
        import copy
        if isinstance(obj, dict):
            a = copy.deepcopy(obj)
            self._massage_refs(a)  # strip external refs etc.
            return a
        if isinstance(obj, list):
            # choose first mapping if present, else fall back
            for it in obj:
                if isinstance(it, dict):
                    return self._normalize_anchor_obj(it)
            return {"type": "string"}
        # bare strings / numbers -> permissive fallback
        return {"type": "string"}

# ==============================================================================
# Projections: cBio/CCDI tables -> GDC-like node tables
# ==============================================================================

def cbio_to_case(patients_csv_df: pd.DataFrame,
                 samples_csv_df: pd.DataFrame,
                 project_id: str,
                 strict: bool,
                 cfg: dict) -> pd.DataFrame:
    dfp = patients_csv_df.rename(columns={"patientId": "submitter_id"}).copy()

    out = pd.DataFrame({
        "submitter_id": dfp["submitter_id"].astype(str),
        "project_id": project_id,
    })
    # Even in lite mode, populate these if we can (schema-required in strict)
    try:
        out["disease_type"] = _derive_disease_type(patients_csv_df, samples_csv_df, cfg).values
    except Exception:
        if strict:
            raise
    try:
        prim = _derive_primary_site(patients_csv_df, samples_csv_df, cfg)
        out["primary_site"] = prim.reindex(dfp["submitter_id"].astype(str)).fillna("Unknown").values
    except Exception:
        if strict:
            raise

    return out.drop_duplicates("submitter_id").dropna(subset=["submitter_id"])

def cbio_to_demographic(patients_csv_df: pd.DataFrame,
                        strict: bool = True) -> pd.DataFrame:
    """
    Build a GDC-compliant demographic node table from cBio patient clinical.
    Returns columns suitable for strict validation: submitter_id, sex_at_birth,
    vital_status, ethnicity, race. Keeps case linkage in 'case_submitter_id'
    (drop that column before validating).
    """
    # subject IDs
    subj = _patients_subject_ids(patients_csv_df)  # patientId or submitter_id

    # required fields
    demo = pd.DataFrame({
        "submitter_id": (subj + "-DEMO") if strict else subj,
        "sex_at_birth": _derive_sex_at_birth(patients_csv_df),
        "vital_status": _derive_vital_status(patients_csv_df),
    })

    # ethnicity / race if present
    demo["ethnicity"] = _norm_ethnicity(patients_csv_df.get("ETHNICITY"))
    demo["race"]      = _norm_race(patients_csv_df.get("RACE"))

    # keep link to case for your index layer (do NOT validate this column)
    demo["case_submitter_id"] = subj
    return demo.dropna(subset=["submitter_id"]).drop_duplicates("submitter_id")

def cbio_to_diagnosis(patients_csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal diagnosis table with tumor_stage (maps from CLINICAL_STAGE best-effort).
    """
    df = patients_csv_df.rename(columns={"patientId": "case_submitter_id"})
    if "CLINICAL_STAGE" in df.columns:
        tumor_stage = df["CLINICAL_STAGE"].astype(str).str.lower()
    else:
        tumor_stage = pd.Series(index=df.index, dtype=object)
    out = pd.DataFrame({
        "case_submitter_id": df["case_submitter_id"].astype(str),
        "tumor_stage": tumor_stage,
    })
    return out.drop_duplicates("case_submitter_id").dropna(subset=["case_submitter_id"])

def cbio_to_sample(samples_csv_df: pd.DataFrame,
                   strict: bool,
                   cfg: dict) -> pd.DataFrame:
    sid = samples_csv_df["sampleId"].astype(str)

    specimen_type, tissue_type, tumor_desc = _derive_sample_fields(samples_csv_df, cfg)
    preserv = _derive_preservation_method(samples_csv_df, cfg)

    node = pd.DataFrame({
        "submitter_id": sid,
        "preservation_method": preserv,     # REQUIRED
        "specimen_type": specimen_type,     # REQUIRED (material: 'Solid Tissue', 'Peripheral Whole Blood', …)
        "tissue_type": tissue_type,         # REQUIRED (context: 'Tumor', 'Normal', …)
        "tumor_descriptor": tumor_desc,     # REQUIRED ('Primary','Recurrence','Metastatic','Unknown', …)
    })

    # Keep case link for your graph layer (NOT part of node schema)
    node["case_submitter_id"] = _samples_subject_ids(samples_csv_df)
    return node.dropna(subset=["submitter_id"]).drop_duplicates("submitter_id")

def cbio_to_clinical(patients_csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    GDC-like 'clinical' node (non-submittable) per schemas/clinical.yaml.
    Required: id, submitter_id, cases (to_one link to case).
    We keep `case_submitter_id` only for writing link parquet; drop before validation.
    """
    df = patients_csv_df.rename(columns={"patientId": "case_submitter_id"}).copy()
    case_id = df["case_submitter_id"].astype(str)

    # Numeric sources
    age_days = pd.to_numeric(df.get("AGE_IN_DAYS"), errors="coerce")
    os_days  = pd.to_numeric(df.get("OS_DAYS"), errors="coerce")
    os_stat  = df.get("OS_STATUS")

    # days_to_death only when deceased
    if os_stat is not None:
        dead_mask = os_stat.astype(str).str.upper().str.contains("DECEASED", na=False)
    else:
        dead_mask = pd.Series(False, index=df.index)

    days_to_death = pd.Series([None] * len(df), index=df.index, dtype=object)
    days_to_death[dead_mask] = os_days[dead_mask]

    clinical = pd.DataFrame({
        # required
        "submitter_id": case_id,
        "age_at_diagnosis": age_days,  
        "days_to_death": days_to_death,
        "gender": _clinical_gender_from_sex(df.get("SEX")),
        "ethnicity": _clinical_ethnicity_2enums(df.get("ETHNICITY")),
        "race": _norm_race(df.get("RACE")).replace({"unknown": "not reported"}),
        "vital_status": _clinical_vital_status_from_os_status(df.get("OS_STATUS")),
    })
    # required 'id' (deterministic per submitter_id)
    clinical["id"] = clinical["submitter_id"].apply(_stable_uuid5)
    # required 'cases' (to_one link). Use submitter_id reference.
    clinical["cases"] = case_id.map(lambda sid: {"submitter_id": sid})
    # keep explicit link for indexes (not part of the node schema)
    clinical["case_submitter_id"] = case_id
    return clinical

def build_idmap_from_samples(samples_csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    subject_id <-> sample_id mapping, plus node ids for graphs.
    """
    df = samples_csv_df.rename(columns={"sampleId": "sample_id"})
    if "subject_id" not in df.columns:
        df["subject_id"] = df["sample_id"].astype(str).map(_subject_from_sample)
    out = df[["subject_id", "sample_id"]].dropna().drop_duplicates()
    out["node_id_subject"] = "patient:" + out["subject_id"].astype(str)
    out["node_id_sample"] = "sample:" + out["sample_id"].astype(str)
    return out

def files_for_zarr(features_dir: Path, project_id: str) -> pd.DataFrame:
    """
    Build a minimal 'file' node table for Zarr feature stores in data/gdm/features/.
    One row per store (mrna/mirna/methylation/scrna/scatac).
    """
    features = []
    mapping = {
        "mrna": ("Transcriptome Profiling", "Gene Expression", "RNA-Seq"),
        "mirna": ("Transcriptome Profiling", "miRNA Expression", "miRNA-Seq"),
        "methylation": ("DNA Methylation", "Methylation Beta Value", "Methylation Array"),
        "scrna": ("Transcriptome Profiling", "Gene Expression", "Single-Cell RNA-Seq"),
        "scatac": ("Epigenomics", "Chromatin Accessibility", "Single-Cell ATAC-Seq"),
    }
    for name, (cat, dtype, strat) in mapping.items():
        z = features_dir / f"{name}.zarr"
        if z.exists():
            features.append({
                "submitter_id": f"{project_id}:{name}.zarr",
                "data_category": cat,
                "data_type": dtype,
                "experimental_strategy": strat,
                "platform": "local-zarr",
                "file_name": f"{name}.zarr",
                "project_id": project_id
            })
    return pd.DataFrame(features)

# ==============================================================================
# DuckDB open + DataModel façade
# ==============================================================================

def open_duckdb(path: Optional[Union[str, Path]] = None,
                mode: str = "memory",
                max_retries: int = 0,
                retry_wait_s: float = 0.5) -> duckdb.DuckDBPyConnection:
    """
    Open DuckDB with simple lock-handling:
      - mode="memory": ephemeral, no file, no locks
      - mode="ro": read-only file (many readers allowed)
      - mode="rw": read-write file (one writer; optional retry on lock)
    """
    path = str(path) if path else None

    if mode == "memory" or not path:
        return duckdb.connect()

    if mode == "ro":
        return duckdb.connect(path, read_only=True)

    if mode != "rw":
        raise ValueError("duckdb_mode must be one of: 'memory', 'ro', 'rw'")

    # rw with optional retry on 'Conflicting lock'
    import time
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return duckdb.connect(path)
        except duckdb.IOException as e:
            msg = str(e)
            last_err = e
            if "Conflicting lock" in msg and attempt < max_retries:
                time.sleep(retry_wait_s)
                continue
            # Any other IOException, or we're out of retries -> raise
            raise
    # If we exhausted retries on lock contention:
    raise last_err if last_err else duckdb.IOException("Failed to open DuckDB (unknown error)")

class DataModel:
    """
    Orchestrates:
      * validation against local GDC schemas
      * writing GDC-shaped Parquet tables under data/gdm/
      * attaching DuckDB TEMP VIEWs for fast cohorting/joins
    """

    def __init__(self,
                 schemas_dir: Union[str, Path] = "src/gdcdictionary/schemas",
                 gdm_root: Optional[Union[str, Path]] = "data/gdm",
                 duckdb_path: Optional[Union[str, Path]] = None,
                 duckdb_mode: str = "memory",
                 strict: Optional[bool] = None,
                 cfg_path: Union[str, Path] = "config/gdc_mapping.yaml",
                 cfg: Optional[dict] = None,
                 ):
        # --- strict/lite switch (CLI param wins, else env, else default False) ---
        self.strict = strict if strict is not None else _env_flag("DM_GDC_STRICT", default=False)

        # --- load config (dict wins; else file; env DM_CONFIG can override path) ---
        cfg_file = Path(os.getenv("DM_CONFIG", str(cfg_path)))
        if cfg is not None:
            self._cfg = cfg
            self._cfg_source = "<dict>"
        else:
            try:
                with open(cfg_file, "r") as f:
                    self._cfg = yaml.safe_load(f) or {}
                self._cfg_source = str(cfg_file)
            except FileNotFoundError:
                # proceed with empty config; downstream code uses safe defaults
                self._cfg = {}
                self._cfg_source = None

        # ensure gdc_mapping sections exist so readers can do .get(...) safely
        gm = self._cfg.setdefault("gdc_mapping", {})
        for key in (
            "disease_type_map",
            "primary_site_map",
            "icdo_topography_map",
            "icdo_morph_to_site_map",
            "oncotree_to_site",
        ):
            gm.setdefault(key, {})

        # If YAML provides gdm_root, let it override the default unless an explicit
        # argument was passed (i.e., only use config when caller left gdm_root as None)
        if gdm_root is None:
            gdm_root = self._cfg.get("gdm_root", "data/gdm")

        # --- schema registry & filesystem layout ---
        self.schemas = GDCSchemaRegistry(schemas_dir)
        self.schemas_dir = Path(schemas_dir) 
        self.gdm_root = Path(gdm_root)
        (self.gdm_root / "clinical").mkdir(parents=True, exist_ok=True)
        (self.gdm_root / "indexes").mkdir(parents=True, exist_ok=True)
        (self.gdm_root / "nodes").mkdir(parents=True, exist_ok=True)
        (self.gdm_root / "files").mkdir(parents=True, exist_ok=True)
        (self.gdm_root / "features").mkdir(parents=True, exist_ok=True)

        # --- DuckDB connection (with simple retry knobs via env) ---
        retries = int(os.getenv("DM_DUCKDB_RETRIES", "0"))
        sleep_s = float(os.getenv("DM_DUCKDB_RETRY_SLEEP", "0.5"))
        self.con = open_duckdb(
            duckdb_path,
            duckdb_mode,
            max_retries=retries,
            retry_wait_s=sleep_s,
        )

    # ---------- validation helper ----------
    def validate_node(self, node: str, df: pd.DataFrame, required_cols: Optional[List[str]] = None):
        if self.strict:
            check_df = df  # validate everything in strict mode
        else:
            if required_cols:
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing required columns for node '{node}': {missing}")
                check_df = df[required_cols]
            else:
                check_df = df

        mode = "strict" if self.strict else "lite"
        obs = None if self.strict else df.columns.tolist()
        errs = self.schemas.validate_df(node, check_df, mode=mode, observed_keys=obs)
        if errs:
            head = "\n".join(f"  row {i}: {msg}" for i, msg in errs[:10])
            raise ValueError(f"Validation errors for node '{node}' (showing up to 10):\n{head}")

    # ---------- write node tables ----------
    def write_case_bundle(self,
                      patients_csv_df: pd.DataFrame,
                      samples_csv_df: pd.DataFrame,
                      project_id: str):
        # ---------- Build nodes ----------
        case = cbio_to_case(patients_csv_df, samples_csv_df, project_id,
                            strict=self.strict, cfg=self._cfg)
        demo = cbio_to_demographic(patients_csv_df, strict=self.strict)
        diag = cbio_to_diagnosis(patients_csv_df)
        samp = cbio_to_sample(samples_csv_df, strict=self.strict, cfg=self._cfg)
        clin = cbio_to_clinical(patients_csv_df)
        idmap = build_idmap_from_samples(samples_csv_df)

        # ---------- Validate (lite by default; strict if DM_GDC_STRICT=1 / --strict) ----------
        self.validate_node("case", case)

        demo_for_validate = demo.drop(columns=["case_submitter_id"], errors="ignore")
        if not demo_for_validate.empty:
            self.validate_node("demographic", demo_for_validate)

        samp_for_validate = samp.drop(columns=["case_submitter_id"], errors="ignore")
        if not samp_for_validate.empty:
            self.validate_node("sample", samp_for_validate)

        clin_for_validate = clin.drop(columns=["case_submitter_id"], errors="ignore")
        if not clin_for_validate.empty:
            self.validate_node("clinical", clin_for_validate)

        # ---------- Persist legacy clinical mirrors ----------
        (self.gdm_root / "clinical").mkdir(parents=True, exist_ok=True)
        case.to_parquet(self.gdm_root / "clinical/clinical_subjects.parquet",
                        compression="zstd", index=False)
        samp.to_parquet(self.gdm_root / "clinical/clinical_samples.parquet",
                        compression="zstd", index=False)

        # ---------- Persist node tables ----------
        nroot = self.gdm_root / "nodes"
        nroot.mkdir(parents=True, exist_ok=True)

        case.to_parquet(nroot / "case.parquet", compression="zstd", index=False)

        if not demo_for_validate.empty:
            demo_for_validate.to_parquet(nroot / "demographic.parquet", compression="zstd", index=False)
            # link: demographic -> case
            links_demo = demo[["submitter_id", "case_submitter_id"]].rename(
                columns={"submitter_id": "demographic_submitter_id"}
            )
            (self.gdm_root / "indexes").mkdir(parents=True, exist_ok=True)
            links_demo.to_parquet(self.gdm_root / "indexes/links_demographic_case.parquet",
                                compression="zstd", index=False)

        if not samp_for_validate.empty:
            samp_for_validate.to_parquet(nroot / "sample.parquet", compression="zstd", index=False)
            # link: sample -> case
            links_samp = samp[["submitter_id", "case_submitter_id"]].rename(
                columns={"submitter_id": "sample_submitter_id"}
            )
            links_samp.to_parquet(self.gdm_root / "indexes/links_sample_case.parquet",
                                compression="zstd", index=False)

        if not clin_for_validate.empty:
            clin_for_validate.to_parquet(nroot / "clinical.parquet", compression="zstd", index=False)
            # link: clinical -> case
            links_clin = clin[["submitter_id", "case_submitter_id"]].rename(
                columns={"submitter_id": "clinical_submitter_id"}
            )
            links_clin.to_parquet(self.gdm_root / "indexes/links_clinical_case.parquet",
                                compression="zstd", index=False)

        # ---------- Indexes ----------
        (self.gdm_root / "indexes").mkdir(parents=True, exist_ok=True)
        idmap.to_parquet(self.gdm_root / "indexes/idmap.parquet",
                        compression="zstd", index=False)

        # Optional: diagnosis node table (kept lenient)
        if isinstance(diag, pd.DataFrame) and not diag.empty:
            diag.to_parquet(nroot / "diagnosis.parquet", compression="zstd", index=False)

        # Nodes catalog (lightweight, optional)
        nodes = pd.concat([
            pd.DataFrame({
                "node_id": "patient:" + case["submitter_id"].astype(str),
                "node_type": "patient",
                "label": case["submitter_id"].astype(str)
            }),
            pd.DataFrame({
                "node_id": "sample:" + samp["submitter_id"].astype(str),
                "node_type": "sample",
                "label": samp["submitter_id"].astype(str)
            })
        ], ignore_index=True)
        nodes.to_parquet(self.gdm_root / "nodes/nodes.parquet",
                        compression="zstd", index=False)

        # ---------- Return frames for metadata/reporting ----------
        return {
            "case": case,
            "demographic": demo_for_validate,
            "sample": samp_for_validate,
            "diagnosis": diag if isinstance(diag, pd.DataFrame) else pd.DataFrame(),
            "clinical": clin_for_validate,
            "idmap": idmap,
        }

    def write_files_for_features(self, project_id: str):
        feats = files_for_zarr(self.gdm_root / "features", project_id)
        if feats.empty:
            return pd.DataFrame()

        files_root = self.gdm_root / "features"

        # Ensure file_name; derive abs path for size/hash
        if "file_name" not in feats.columns:
            if "path" in feats.columns:
                feats = feats.rename(columns={"path": "file_name"})
            else:
                raise ValueError("features listing must include 'file_name' or 'path'")

        feats["__abs_path"] = feats["file_name"].apply(lambda s: (files_root / str(s)).resolve())

        # Required 'file' fields we can compute
        sizes_md5 = feats["__abs_path"].apply(_stat_and_md5)  # helpers from earlier
        feats["file_size"] = sizes_md5.apply(lambda t: t[0])
        feats["md5sum"] = sizes_md5.apply(lambda t: t[1])
        feats["state"] = _gdc_file_state()  # e.g., "validated"

        # If schema requires submitter_id, add it; otherwise we won't include it
        req = self.schemas.required_fields_for("file")
        if "submitter_id" in req and "submitter_id" not in feats.columns:
            feats["submitter_id"] = feats["file_name"].astype(str).map(lambda s: f"{project_id}:{s}")

        # Build strict-safe projection (only allowed props)
        allowed = self.schemas.allowed_properties("file")
        file_node = feats.loc[:, [c for c in feats.columns if c in allowed]].copy()

        # Sanity: ensure all required columns exist now
        missing = [c for c in req if c not in file_node.columns]
        if missing:
            raise ValueError(f"'file' node missing required columns per schema: {missing}. "
                            f"Allowed props = {sorted(allowed)}")

        # Strict validation of minimal node
        self.validate_node("file", file_node, req)

        # Persist strict node + rich analytics view
        out_dir = self.gdm_root / "files"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_node.to_parquet(out_dir / "file.parquet", compression="zstd", index=False)
        feats.drop(columns=["__abs_path"], errors="ignore") \
            .to_parquet(out_dir / "file_full.parquet", compression="zstd", index=False)
        return file_node

     # ---------- non-raising validation (for reports) ----------
    def validate_df_report(
        self,
        node: str,
        df: pd.DataFrame,
        required_cols: Optional[List[str]] = None,
    ) -> dict:
        """
        Run validation and ALWAYS return a structured report dict:
          {"node":..., "rows": N, "errors": [{"row_index": i, "message": msg}, ...]}
        Does not raise. Respects strict/lite modes same as validate_node().
        """
        if self.strict:
            check_df = df
            mode = "strict"
            obs = None
        else:
            if required_cols:
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    # Report the missing fields as "fatal"
                    return {
                        "node": node, "rows": len(df),
                        "errors": [{"row_index": None, "message": f"Missing required columns: {missing}"}]
                    }
                check_df = df[required_cols]
            else:
                check_df = df
            mode = "lite"
            obs = df.columns.tolist()

        errs = self.schemas.validate_df(node, check_df, mode=mode, observed_keys=obs)
        return {
            "node": node,
            "rows": int(len(df)),
            "errors": [{"row_index": int(i) if i is not None else None, "message": str(msg)} for i, msg in errs]
        }

    # ---------- examples & metadata bundle ----------
    def dump_example_rows(
        self,
        out_dir: Path,
        node_frames: dict[str, pd.DataFrame],
        cancer_type: str = "default"
    ) -> None:
        """
        Collect one example row per node and write them all into a single JSON file,
        named <CancerType>_examples.json.
        """
        examples: dict[str, dict] = {}
        for node, df in node_frames.items():
            if df is None or df.empty:
                continue
            rec = df.iloc[0].to_dict()
            rec = {k: _plain(v) for k, v in rec.items()}
            examples[node] = rec

        if examples:
            _ensure_dir(out_dir)
            # slug: letters, numbers, dash/underscore only
            safe_ct = (
                str(cancer_type)
                .strip()
                .replace(" ", "_")
            )
            safe_ct = "".join(ch for ch in safe_ct if ch.isalnum() or ch in ("-", "_"))
            _write_json(out_dir / f"{safe_ct}_examples.json", examples)

    def emit_metadata_bundle(
        self,
        project_id: str,
        nodes_in_scope: list[str],
        validation_reports: list[dict],
        provenance: dict,
        out_root: Optional[Path] = None,
    ) -> None:
        """
        Write the 4 metadata files into data/metadata/ (or provided root).
        """
        out_root = out_root or (self.gdm_root.parent / "metadata")
        _ensure_dir(out_root)

        # 1) dictionary "version" snapshot (path + nodes + mode + timestamp)
        dict_snapshot = {
            "schemas_dir": str(self.schemas_dir) if hasattr(self, "schemas_dir") else None,
            "nodes_validated": nodes_in_scope,
            "strict_mode": bool(self.strict),
            "timestamp": _now_iso(),
            "project_id": project_id,
        }
        # best-effort: record a few enums for transparency
        enums = {}
        for n in nodes_in_scope:
            try:
                enums[n] = self.schemas.enums_for(n)
            except Exception:
                pass
        dict_snapshot["selected_enums"] = enums
        _write_json(out_root / "gdcdictionary_version.json", dict_snapshot)

        # 2) provenance.json (caller-provided + environment)
        prov = dict(provenance or {})
        prov.update({
            "timestamp": _now_iso(),
            "project_id": project_id,
            "python": sys.version,
            "pandas": pd.__version__,
            "duckdb": duckdb.__version__,
        })
        _write_json(out_root / "provenance.json", prov)

        # 3) validation_report.json
        agg = {
            "project_id": project_id,
            "timestamp": _now_iso(),
            "strict_mode": bool(self.strict),
            "reports": validation_reports,
            "totals": {
                "nodes": len(validation_reports),
                "rows": sum(r.get("rows", 0) for r in validation_reports),
                "errors": sum(len(r.get("errors", [])) for r in validation_reports),
            }
        }
        _write_json(out_root / "validation_report.json", agg)

        # 4) validation_report.md (human summary)
        lines = []
        lines.append(f"# Validation Report — {project_id}")
        lines.append(f"- Timestamp: {agg['timestamp']}")
        lines.append(f"- Strict mode: {agg['strict_mode']}")
        lines.append("")
        for r in validation_reports:
            node = r["node"]
            rows = r["rows"]
            errn = len(r["errors"])
            lines.append(f"## {node}")
            lines.append(f"- Rows: {rows}")
            lines.append(f"- Errors: {errn}")
            if errn:
                preview = r["errors"][:10]
                lines.append("  - First errors:")
                for e in preview:
                    lines.append(f"    - row {e['row_index']}: {e['message']}")
            lines.append("")
        _write_text(out_root / "validation_report.md", "\n".join(lines))
    
    # ---------- DuckDB views ----------
    def attach_views(self):
        """
        Register Parquet sources as TEMP VIEWs (no prepared params in CREATE VIEW).
        """
        reg = {
            "subjects":        self.gdm_root / "clinical/clinical_subjects.parquet",
            "samples":         self.gdm_root / "clinical/clinical_samples.parquet",
            "gdc_case":        self.gdm_root / "nodes/case.parquet",
            "links_demographic_case": self.gdm_root / "indexes/links_demographic_case.parquet",
            "gdc_demographic": self.gdm_root / "nodes/demographic.parquet",
            "gdc_diagnosis":   self.gdm_root / "nodes/diagnosis.parquet",
            "idmap":           self.gdm_root / "indexes/idmap.parquet",
            "nodes":           self.gdm_root / "nodes/nodes.parquet",
            "files":           self.gdm_root / "files/file.parquet",
            "mrna_rows":       self.gdm_root / "indexes/mrna_rows.parquet",
            "mrna_cols":       self.gdm_root / "indexes/mrna_cols.parquet",
            "mirna_rows":      self.gdm_root / "indexes/mirna_rows.parquet",
            "mirna_cols":      self.gdm_root / "indexes/mirna_cols.parquet",
            "methyl_rows":     self.gdm_root / "indexes/methyl_rows.parquet",
            "methyl_cols":     self.gdm_root / "indexes/methyl_cols.parquet",
            "gdc_clinical":   self.gdm_root / "nodes/clinical.parquet",
            "links_clinical_case": self.gdm_root / "indexes/links_clinical_case.parquet",
        }

        for view_name, p in reg.items():
            p = Path(p)
            if p.exists():
                path_sql = p.as_posix().replace("'", "''")  # escape single quotes
                self.con.execute(
                    f"CREATE OR REPLACE TEMP VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{path_sql}')"
                )

    def sql(self, query: str) -> pd.DataFrame:
        return self.con.execute(query).fetch_df()

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---config-aware entrypoints for the pipeline ---

def _cfg_load(path: Optional[str] = None) -> dict:
    cfg_path = path or os.getenv("CONFIG_PATH", "config/gdc_mapping.yaml")
    try:
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def _ct_dir(root: Path, ct: str) -> Path:
    return root / ct.replace(" ", "_")

def build_data_model(
    project_id: Optional[str] = None,
    schemas_dir: Optional[str] = None,
    gdm_root: Optional[str] = None,
    strict: Optional[bool] = None,
    ) -> None:
    cfg = _cfg_load()
    schemas_dir = schemas_dir or cfg.get("schemas_dir", "src/gdcdictionary/schemas")
    gdm_root = gdm_root or cfg.get("gdm_root", "data/gdm")
    project_id = project_id or cfg.get("project_id", "wt_target_gdc")
    strict = strict if strict is not None else _env_flag("DM_GDC_STRICT", default=False)

    cbio_root = Path("data") / "fetch" / "cbioportal_api_request"
    cancer_types = cfg.get("cancer_type_interest", []) or []
    print(cancer_types)

    with DataModel(
        schemas_dir=schemas_dir,
        gdm_root=gdm_root,
        strict=strict,
    ) as dm:
        any_written = False
        all_reports = []
        provenance_sources = []
        file_node_df = pd.DataFrame()  # <-- define once, outside loop

        for ct in cancer_types:
            cbio_dir = _ct_dir(cbio_root, ct)
            pat_csv = cbio_dir / "clinical_patients.csv"
            sam_csv = cbio_dir / "clinical_samples.csv"
            if not pat_csv.exists() or not sam_csv.exists():
                print(f"[data_model] SKIP {ct}: missing {pat_csv} or {sam_csv}")
                continue

            patients = pd.read_csv(pat_csv)
            samples = pd.read_csv(sam_csv)

            node_frames = dm.write_case_bundle(patients, samples, project_id=project_id)
            any_written = True

            # per-node validation reports (non-raising)
            rep_case = dm.validate_df_report("case", node_frames["case"], dm.schemas.required_fields_for("case"))
            rep_demo = dm.validate_df_report("demographic", node_frames["demographic"],
                                             dm.schemas.required_fields_for("demographic")) if not node_frames["demographic"].empty else {"node":"demographic","rows":0,"errors":[]}
            rep_samp = dm.validate_df_report("sample", node_frames["sample"],
                                             dm.schemas.required_fields_for("sample")) if not node_frames["sample"].empty else {"node":"sample","rows":0,"errors":[]}
            rep_diag = {"node":"diagnosis","rows":int(len(node_frames["diagnosis"])), "errors": []}  # lenient
            rep_clin = dm.validate_df_report("clinical", node_frames["clinical"],
                                 dm.schemas.required_fields_for("clinical")) \
                       if not node_frames["clinical"].empty else {"node":"clinical","rows":0,"errors":[]}
            all_reports.extend([rep_case, rep_demo, rep_samp, rep_diag, rep_clin])

            provenance_sources.append({
                "cancer_type": ct,
                "cbioportal_dir": str(cbio_dir),
                "clinical_patients_csv": str(pat_csv),
                "clinical_samples_csv": str(sam_csv),
                "row_counts": {
                    "patients": int(len(patients)),
                    "samples": int(len(samples))
                }
            })

            # ---- dump consolidated human-readable examples per cancer ----
            dm.dump_example_rows(
                dm.gdm_root.parent / "metadata",
                {
                    "case":        node_frames["case"],
                    "demographic": node_frames["demographic"],
                    "sample":      node_frames["sample"],
                    "diagnosis":   node_frames["diagnosis"],
                    "clinical":    node_frames["clinical"],
                    "file":        pd.DataFrame(),  # file examples added after loop (project-level)
                },
                cancer_type=ct
            )

        # ---- after processing all cancers ----
        if not any_written:
            print("[data_model] nothing to write (no clinical CSVs found).")
            return

        # Register feature stores into 'file' node (once)
        file_node_df = dm.write_files_for_features(project_id=project_id)
        if not file_node_df.empty:
            rep_file = dm.validate_df_report("file", file_node_df, dm.schemas.required_fields_for("file"))
            all_reports.append(rep_file)

        # Attach views once
        dm.attach_views()
        print("[data_model] GDM tables written and views attached.")

        # ---- emit metadata bundle (once) ----
        nodes_in_scope = ["case", "demographic", "sample", "diagnosis"]
        if not file_node_df.empty:
            nodes_in_scope.append("file")

        provenance = {
            "sources": provenance_sources,
            "features_root": str(dm.gdm_root / "features"),
            "notes": "Clinical/omics fetched from cBioPortal + CCDI; transformed to GDC-shaped nodes."
        }
        dm.emit_metadata_bundle(
            project_id=project_id,
            nodes_in_scope=nodes_in_scope,
            validation_reports=all_reports,
            provenance=provenance,
            out_root=dm.gdm_root.parent / "metadata",  # -> data/metadata/
        )

        # ---- optional: write a project-level features exemplar ----
        if not file_node_df.empty:
            dm.dump_example_rows(
                dm.gdm_root.parent / "metadata",
                {"file": file_node_df},
                cancer_type=f"{project_id}_features"
            )

def run_duckdb_query(sql: str, out: Optional[str] = None) -> None:
    """
    Run an ad-hoc DuckDB query against GDM Parquets (views must exist or will be attached).
    If 'out' is provided, write a CSV there; always print a small preview to stdout.
    """
    if not sql or not sql.strip():
        raise ValueError("No SQL provided.")
    cfg = _cfg_load()
    gdm_root = cfg.get("gdm_root", "data/gdm")

    with DataModel(
        schemas_dir=cfg.get("schemas_dir", "src/gdcdictionary/schemas"),
        gdm_root=gdm_root,
        strict=_env_flag("DM_GDC_STRICT", default=False),
    ) as dm:
        dm.attach_views()
        df = dm.sql(sql)
        # print a tiny preview
        try:
            print(df.head(20).to_string(index=False))
        except Exception:
            print(df.head(20))
        if out:
            outp = Path(out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outp, index=False)
            print(f"[data_query] wrote {outp}")


if __name__ == "__main__":
    """
    Example usage (run inside Docker container):

    python -m src.data_model \
    --schemas-dir src/gdcdictionary/schemas \
    --gdm-root data/gdm \
    --cbio-dir data/fetch/cbioportal_api_request/Wilms_Tumor \
    --project-id wt_target_gdc

    Environment overrides:
      DM_DUCKDB_MODE = memory | ro | rw   (default: memory)
      DM_DUCKDB_PATH = /data/gdm/gdm.duckdb
    """
    import argparse

    parser = argparse.ArgumentParser(description="GDC-aligned Data Model demo (gdm/).")
    parser.add_argument("--schemas-dir", default="src/gdcdictionary/schemas", help="Path to GDC YAML schemas.")
    parser.add_argument("--gdm-root", default="data/gdm", help="Output root for GDM Parquet/Zarr.")
    parser.add_argument("--cbio-dir", default="data/fetch/cbioportal_api_request/Wilms_Tumor",
                        help="Folder with clinical_patients.csv and clinical_samples.csv.")
    parser.add_argument("--project-id", default="wt_target_gdc", help="Project ID to stamp on 'case' and 'file'.")
    parser.add_argument("--strict", dest="strict", action="store_true", help="Force strict GDC validation.")
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=None)  # None -> defer to env
    args = parser.parse_args()

    duckdb_mode = os.getenv("DM_DUCKDB_MODE", "memory")     # memory|ro|rw
    duckdb_path = os.getenv("DM_DUCKDB_PATH", None)         # e.g., /data/gdm/gdm.duckdb
    cli_strict = args.strict if args.strict is not None else _env_flag("DM_GDC_STRICT", default=False)

    # Load raw cBio clinical CSVs (or synthesize tiny frames if missing)
    pat_csv = Path(args.cbio_dir) / "clinical_patients.csv"
    sam_csv = Path(args.cbio_dir) / "clinical_samples.csv"

    if pat_csv.exists():
        patients = pd.read_csv(pat_csv)
    else:
        print(f"[demo] {pat_csv} not found; creating a tiny synthetic patients table.")
        patients = pd.DataFrame({
            "patientId": ["TARGET-50-CAAAAC", "TARGET-50-CAAAAH"],
            "SEX": ["Female", "Female"],
            "CLINICAL_STAGE": ["III", "III"],
            "ETHNICITY": ["Not Hispanic or Latino", "Not reported"],
            "RACE": ["White", "White"]
        })

    if sam_csv.exists():
        samples = pd.read_csv(sam_csv)
    else:
        print(f"[demo] {sam_csv} not found; creating a tiny synthetic samples table.")
        samples = pd.DataFrame({
            "sampleId": ["TARGET-50-CAAAAC-01", "TARGET-50-CAAAAH-01"],
            "CANCER_TYPE": ["Wilms Tumor", "Wilms Tumor"],
            "HISTOLOGY_CLASSIFICATION_IN_PRIMARY_TUMOR": ["FHWT", "DAWT"]
        })

    # Derive a human-readable cancer type from the folder name (e.g., "Wilms_Tumor" -> "Wilms Tumor")
    cancer_type = Path(args.cbio_dir).name.replace("_", " ")

    with DataModel(
        schemas_dir=args.schemas_dir,
        gdm_root=args.gdm_root,
        duckdb_path=duckdb_path,
        duckdb_mode=duckdb_mode,
        strict=cli_strict,
    ) as dm:
        # Build node bundles
        node_frames = dm.write_case_bundle(patients, samples, project_id=args.project_id)
        file_node_df = dm.write_files_for_features(project_id=args.project_id)
        dm.attach_views()

        # Build validation reports
        reports = [
            dm.validate_df_report("case", node_frames["case"], dm.schemas.required_fields_for("case")),
            dm.validate_df_report("demographic", node_frames["demographic"],
                                  dm.schemas.required_fields_for("demographic")) if not node_frames["demographic"].empty else {"node":"demographic","rows":0,"errors":[]},
            dm.validate_df_report("sample", node_frames["sample"],
                                  dm.schemas.required_fields_for("sample")) if not node_frames["sample"].empty else {"node":"sample","rows":0,"errors":[]},
            {"node":"diagnosis","rows":int(len(node_frames["diagnosis"])), "errors":[]}
        ]
        if not file_node_df.empty:
            reports.append(dm.validate_df_report("file", file_node_df, dm.schemas.required_fields_for("file")))

        # Provenance bundle for this demo run
        provenance = {
            "sources": [{
                "cbioportal_dir": str(args.cbio_dir),
                "clinical_patients_csv": str(pat_csv),
                "clinical_samples_csv": str(sam_csv),
                "row_counts": {
                    "patients": int(len(patients)),
                    "samples": int(len(samples))
                }
            }],
            "notes": "Demo run with cBioPortal + CCDI extracts"
        }

        nodes_in_scope = ["case","demographic","sample","diagnosis"]
        if not file_node_df.empty:
            nodes_in_scope.append("file")

        out_meta = Path(args.gdm_root).parent / "metadata"
        dm.emit_metadata_bundle(
            project_id=args.project_id,
            nodes_in_scope=nodes_in_scope,
            validation_reports=reports,
            provenance=provenance,
            out_root=out_meta
        )

        # ---- Per-cancer example file (prefix) ----
        dm.dump_example_rows(
            out_meta,
            {
                "case": node_frames["case"],
                "demographic": node_frames["demographic"],
                "sample": node_frames["sample"],
                "diagnosis": node_frames["diagnosis"],
                "clinical": node_frames["clinical"],
            },
            cancer_type=cancer_type
        )

        # ---- Optional: project-level features exemplar ----
        if not file_node_df.empty:
            dm.dump_example_rows(
                out_meta,
                {"file": file_node_df},
                cancer_type=f"{args.project_id}_features"
            )

        # ---- Print summary ----
        print("\n[demo] Metadata written to:", out_meta)
        for f in sorted(out_meta.glob("*")):
            print(" -", f.name)

        # Show quick preview of validation_report.md
        vr_md = out_meta / "validation_report.md"
        if vr_md.exists():
            print("\n[demo] validation_report.md (first 20 lines):")
            with open(vr_md, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    print(line.rstrip())
                    if i >= 19:
                        break

        # Show any *_examples.json files as a sanity check
        for exfile in sorted(out_meta.glob("*_examples.json")):
            print(f"\n[demo] {exfile.name}:")
            print(open(exfile, "r", encoding="utf-8").read())