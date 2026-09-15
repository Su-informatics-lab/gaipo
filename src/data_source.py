# src/data_source.py
"""
data_source.py — Harvest CCDI Federation subjects & samples by cancer-type terms

What it does
------------
1) Builds active cancer-type vocabularies from config:
     - default_type_terms + custom_type_terms
     - restricts to cancer_type_interest
     - compiles word-boundary regex per active type
2) Pages CCDI Federation endpoints to harvest:
     - /subject-diagnosis  → subject_id + diagnoses (per source)
     - /sample-diagnosis   → sample_id + subject_id + diagnoses (per source)
3) Classifies entities to cancer types via compiled regexes.
4) Writes per-type CSVs and an id map parquet for downstream steps.

Key inputs (from config/pipeline_config.yaml)
---------------------------------------------
services:
  ccdi_sample_diagnosis: https://federation.ccdi.cancer.gov/api/v1/sample-diagnosis
  ccdi_subject_diagnosis: https://federation.ccdi.cancer.gov/api/v1/subject-diagnosis
cancer_type_interest: ["Wilms Tumor", ...]
default_type_terms: { "Wilms Tumor": ["wilms", "nephroblastoma", ...], ... }
custom_type_terms: {}  # optional additions/overrides
target_source: null | "PCDC" | "KidsFirst" | ...  # filter to a single federation source
per_page: 100
timeout: 60
retry_status: [429, 500, 502, 503, 504]
studyId:
  "Wilms Tumor": "wt_target_2018_pub"    # used to annotate idmap.parquet

Outputs (relative to repo root)
-------------------------------
data/fetch/ccdi_api_request/
  subject_ids/<cancer_type>.csv   # columns: source,subject_id,diagnoses
  sample_ids/<cancer_type>.csv    # columns: source,sample_id,subject_id,diagnoses
data/gdm/indexes/
  idmap.parquet                   # columns: source,sample_id,subject_id,diagnoses,study_id,
                                  #          node_id_subject,node_id_sample

Environment
-----------
CONFIG_PATH      : path to pipeline_config.yaml (default: config/pipeline_config.yaml)
VERBOSE (yaml)   : verbose logs when True

CLI / Pipeline
--------------
- Recommended: via main driver
    python -m src.main --call data_fetch
- Ad-hoc (module has a __main__ that calls data_fetch()):
    python -m src.data_source

Notes & tips
------------
- Classification is regex-based on the 'associated_diagnoses' values returned by the API.
- Set `target_source` in config to restrict to one federation backend (e.g., "PCDC") if you want a clean cohort.
- The helper writes/updates idmap.parquet idempotently and de-dups by sample_id.
- Rate limiting: on 429/5xx, the client backs off with simple 2**i sleeps up to 5 tries.
"""
import os
import requests
import time
import csv
import re
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import yaml

# ─── Paths & Config ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

cfg_env = os.getenv("CONFIG_PATH")
cfg_path = Path(cfg_env) if cfg_env else ROOT / "config" / "pipeline_config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Service endpoints
BASE_SUBJECT = cfg["services"]["ccdi_subject_diagnosis"].rstrip("/")
BASE_SAMPLE  = cfg["services"]["ccdi_sample_diagnosis"].rstrip("/")

# HTTP settings
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "gaipo/0.1 (+data_source.py)"})
TIMEOUT = cfg.get("timeout", 120)
RETRY_STATUS = set(cfg.get("retry_status", []))

# Pipeline knobs
PER_PAGE = cfg.get("per_page", 100)
VERBOSE = cfg.get("verbose", True)
CONSIDER_OVERLAP = cfg.get("consider_overlap", False)

# Cancer‐type definitions
default_type_terms = cfg["default_type_terms"]
custom_type_terms  = cfg.get("custom_type_terms") or {}
cancer_type_interest = cfg["cancer_type_interest"]
TARGET_SOURCE = cfg.get("target_source")
STUDY_IDS     = cfg.get("studyId", {})  # e.g., {"Wilms Tumor": "wt_target_2018_pub"}

# Output directories
SUBJECT_IDS_DIR = ROOT / "data" / "fetch" / "ccdi_api_request" / "subject_ids"
SAMPLE_IDS_DIR  = ROOT / "data" / "fetch" / "ccdi_api_request" / "sample_ids"
for p in (SUBJECT_IDS_DIR, SAMPLE_IDS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Parquet file output path
GDM_ROOT  = ROOT / "data" / "gdm"
GDM_INDEX = GDM_ROOT / "indexes"
GDM_INDEX.mkdir(parents=True, exist_ok=True)
GDM_ROOT.mkdir(parents=True, exist_ok=True)


# ─── Build Active Types & Regexes ──────────────────────────────────────────────

# Merge default + custom
combined_type_terms = {**default_type_terms, **custom_type_terms}

# Validate interest list
for t in cancer_type_interest:
    if t not in combined_type_terms:
        raise KeyError(f"Interest {t!r} not in defined types")

# Restrict to active
active_type_terms = {
    t: sorted(combined_type_terms[t])
    for t in cancer_type_interest
}

active_type_regexes = {
    t: re.compile(
        r"\b(?:" + "|".join(re.escape(term) for term in active_type_terms[t]) + r")\b",
        re.I
    )
    for t in active_type_terms
}

# All unique search terms
search_terms = [
    term
    for terms in active_type_terms.values()
    for term in terms
]


# ─── Utilities to robustly flatten CCDI value objects ──────────────────────────

def _flatten_scalar(obj):
    """
    Return a robust string for a single value-like thing.

    Handles common CCDI shapes, e.g.
      {"value": "Alive"} →
      {"value": None, "ancestors": ["unharmonized.vital_status"]} →
      {"name": "White"} →
      {"value": {"days": 123}} →
      {"number": 123, "unit": "days"} →
    """
    if obj is None:
        return ""

    # simple scalars
    if isinstance(obj, (str, int, float, bool)):
        return str(obj)

    # dicts – try common keys in order
    if isinstance(obj, dict):
        # direct value/name/id/text
        for k in ("value", "name", "text", "label", "id"):
            if k in obj and obj[k] is not None:
                s = _flatten_scalar(obj[k])
                if s:
                    return s

        # numeric-ish structures
        for k in ("number", "amount", "val"):
            if k in obj and obj[k] is not None:
                return str(obj[k])

        # time-ish structures
        for k in ("days", "months", "years"):
            if k in obj and obj[k] is not None:
                return str(obj[k])

        # otherwise: pick first non-empty scalar we find
        for v in obj.values():
            s = _flatten_scalar(v)
            if s:
                return s

        # if truly nothing usable, return empty
        return ""

    # lists/tuples – join their scalars
    if isinstance(obj, (list, tuple)):
        return _flatten_list(obj)

    # fallback stable JSON (should be rare)
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(obj)


def _flatten_list(lst, sep="|"):
    """Flatten a list of possibly nested value-like things into a sep-joined string."""
    out = []
    for x in (lst or []):
        if isinstance(x, (list, tuple)):
            out.append(_flatten_list(x, sep=sep))
        else:
            out.append(_flatten_scalar(x))
    # remove empties but keep order
    out = [s for s in out if s]
    return sep.join(out)


def as_str(v):
    """Flatten dict/list/scalar to a compact string."""
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return _flatten_scalar(v)
    if isinstance(v, list):
        return _flatten_list(v)
    if isinstance(v, dict):
        # If it's a dict with 'value' / 'name' curation, extract; else JSON
        return _flatten_scalar(v)
    return _flatten_scalar(v)


def md_get(md, key):
    """Extract metadata[key] → flattened string."""
    if not isinstance(md, dict):
        return ""
    return as_str(md.get(key))


def md_get_set_of_values(md, key):
    """
    For things like 'associated_diagnoses': list of dicts with 'value'.
    Returns a set of strings.
    """
    vals = set()
    if not isinstance(md, dict):
        return vals
    raw = md.get(key)
    if isinstance(raw, list):
        for item in raw:
            s = as_str(item)
            if s:
                vals.add(s)
    elif raw is not None:
        s = as_str(raw)
        if s:
            vals.add(s)
    return vals

def md_get_value(md, key, join_sep="|"):
    """
    Extract a field from metadata favoring the '.value' member.
    - dict with 'value' -> return str(value) or "" if None
    - list -> join each item's extracted value with join_sep
    - scalar -> str(scalar)
    Else -> ""
    """
    if not isinstance(md, dict):
        return ""
    obj = md.get(key)

    def _value_of(x):
        if x is None:
            return ""
        if isinstance(x, (str, int, float, bool)):
            return str(x)
        if isinstance(x, dict):
            # Prefer the explicit 'value' slot; if it's None, treat as empty string
            if "value" in x:
                v = x.get("value")
                return "" if v is None else str(v)
            # Fall back to common shapes
            if "name" in x:
                return "" if x.get("name") is None else str(x.get("name"))
            if "id" in x:
                # id could be nested
                idv = x.get("id")
                if isinstance(idv, dict) and "name" in idv:
                    return "" if idv.get("name") is None else str(idv.get("name"))
                return "" if idv is None else str(idv)
            return ""
        if isinstance(x, list):
            return join_sep.join([_value_of(y) for y in x if _value_of(y)])
        return ""
    if isinstance(obj, list):
        parts = [_value_of(it) for it in obj]
        parts = [p for p in parts if p]
        return join_sep.join(parts)
    return _value_of(obj)

# ─── Generic paging helper ─────────────────────────────────────────────────────

def _get_page(base, term, page, per_page=PER_PAGE):
    params = {"search": term, "page": page, "per_page": per_page}
    for i in range(5):
        r = SESSION.get(base, params=params, timeout=TIMEOUT)
        if r.status_code in RETRY_STATUS:
            time.sleep(2**i)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


# ─── Subject pipeline ──────────────────────────────────────────────────────────

SUBJECT_OUT_COLUMNS = [
    "source",
    "subject_id",
    "diagnosis",               # from associated_diagnoses
    "depositions",
    "sex",
    "race",
    "ethnicity",
    "identifiers",
    "vital_status",
    "age_at_vital_status",
]

def harvest_subjects_for_term(term):
    page = 1
    per_src = {}
    done = set()
    while True:
        blocks = _get_page(BASE_SUBJECT, term, page)
        all_done = True
        for b in blocks:
            src = b.get("source")
            if not src or b.get("errors"):
                continue
            if TARGET_SOURCE and src != TARGET_SOURCE:
                continue

            total = b["summary"]["counts"]["all"]
            data  = b["data"]
            info  = per_src.setdefault(
                src,
                {"total": total, "seen": 0, "subjects": {}}
            )
            info["seen"] += len(data)
            if info["seen"] < total:
                all_done = False
            else:
                done.add(src)

            for subj in data:
                sid = subj["id"]["name"]
                md  = subj.get("metadata") or {}

                # diagnosis set from associated_diagnoses
                diag_set = md_get_set_of_values(md, "associated_diagnoses")

                # other fields
                row = info["subjects"].setdefault(
                    sid,
                    {
                        "diagnosis": set(),
                        "depositions": "",
                        "sex": "",
                        "race": "",
                        "ethnicity": "",
                        "identifiers": "",
                        "vital_status": "",
                        "age_at_vital_status": "",
                    }
                )
                row["diagnosis"].update(diag_set)
                row["depositions"]        = row["depositions"] or md_get(md, "depositions")
                row["sex"]                = row["sex"] or md_get(md, "sex")
                row["race"]               = row["race"] or md_get(md, "race")
                row["ethnicity"]          = row["ethnicity"] or md_get(md, "ethnicity")
                row["identifiers"]        = row["identifiers"] or md_get(md, "identifiers")
                row["vital_status"]        = row["vital_status"] or md_get_value(md, "vital_status")
                row["age_at_vital_status"] = row["age_at_vital_status"] or md_get_value(md, "age_at_vital_status")

        if VERBOSE:
            print(f"[SUBJ:{term}] p{page}: {len(done)}/{len(per_src)}")
        if all_done:
            break
        page += 1
    return per_src


def merge_subjects(results):
    merged = {}
    for _, per_src in results:
        for src, info in per_src.items():
            dst = merged.setdefault(src, {})
            for sid, row in info["subjects"].items():
                mr = dst.setdefault(
                    sid,
                    {
                        "diagnosis": set(),
                        "depositions": "",
                        "sex": "",
                        "race": "",
                        "ethnicity": "",
                        "identifiers": "",
                        "vital_status": "",
                        "age_at_vital_status": "",
                    }
                )
                mr["diagnosis"].update(row["diagnosis"])
                for k in ("depositions","sex","race","ethnicity","identifiers","vital_status","age_at_vital_status"):
                    if not mr[k] and row[k]:
                        mr[k] = row[k]
    return merged


def classify_entities(merged_map, regexes):
    out = {}
    for src, emap in merged_map.items():
        o2 = {}
        for eid, row in emap.items():
            diags = row["diagnosis"]
            mt = {t for t, r in regexes.items() if any(r.search(v) for v in diags)}
            if mt:
                o2[eid] = {
                    "diagnoses": set(diags),
                    "matched_types": mt,
                    **{k: row[k] for k in ("depositions","sex","race","ethnicity","identifiers","vital_status","age_at_vital_status")}
                }
        out[src] = o2
    return out


def filter_entities(matches, include):
    inc = set(include)
    rows = []
    for src, emap in matches.items():
        for eid, info in emap.items():
            if inc.issubset(info["matched_types"]):
                rows.append((
                    src,
                    eid,
                    info["diagnoses"],
                    info["depositions"],
                    info["sex"],
                    info["race"],
                    info["ethnicity"],
                    info["identifiers"],
                    info["vital_status"],
                    info["age_at_vital_status"],
                ))
    return rows


def export_subjects(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUBJECT_OUT_COLUMNS)
        for (src, sid, diags, dep, sex, race, eth, ids, vs, a_vs) in rows:
            w.writerow([
                src,
                sid,
                "|".join(sorted(diags)),
                dep,
                sex,
                race,
                eth,
                ids,
                vs,
                a_vs,
            ])
    print(f"Wrote {len(rows):,} rows to {out_path}")


# ─── Sample pipeline ───────────────────────────────────────────────────────────

SAMPLE_OUT_COLUMNS = [
    "source",
    "sample_id",
    "subject_id",
    "diagnosis",
    "depositions",
    "age_at_diagnosis",
    "anatomical_sites",
    "disease_phase",
    "library_selection_method",
    "tissue_type",
    "tumor_classification",
    "tumor_tissue_morphology",
    "age_at_collection",
    "library_strategy",
    "library_source_material",
    "preservation_method",
    "tumor_grade",
    "specimen_molecular_analyte_type",
    "identifiers",
]

def harvest_samples_for_term(term):
    page = 1
    per_src = {}
    done = set()
    while True:
        blocks = _get_page(BASE_SAMPLE, term, page)
        all_done = True
        for b in blocks:
            src = b.get("source")
            if not src or b.get("errors"):
                continue
            if TARGET_SOURCE and src != TARGET_SOURCE:
                continue

            total = b["summary"]["counts"]["all"]
            data  = b["data"]
            info  = per_src.setdefault(
                src,
                {"total": total, "seen": 0, "samples": {}}
            )
            info["seen"] += len(data)
            if info["seen"] < total:
                all_done = False
            else:
                done.add(src)

            for samp in data:
                samp_id = samp["id"]["name"]
                subj_id = samp["subject"]["name"]
                md      = samp.get("metadata") or {}

                # primary diagnosis value (string)
                diag_str = md_get(md, "diagnosis")

                entry = info["samples"].setdefault(
                    samp_id,
                    {
                        "subject_id": subj_id,
                        "diagnoses": set(),  # for classification (set of strings)
                        "depositions": "",
                        "age_at_diagnosis": "",
                        "anatomical_sites": "",
                        "disease_phase": "",
                        "library_selection_method": "",
                        "tissue_type": "",
                        "tumor_classification": "",
                        "tumor_tissue_morphology": "",
                        "age_at_collection": "",
                        "library_strategy": "",
                        "library_source_material": "",
                        "preservation_method": "",
                        "tumor_grade": "",
                        "specimen_molecular_analyte_type": "",
                        "identifiers": "",
                    }
                )

                if diag_str:
                    entry["diagnoses"].add(diag_str)

                # fill first non-empty for the rest
                entry["depositions"]                    = entry["depositions"] or md_get(md, "depositions")
                entry["age_at_diagnosis"]               = entry["age_at_diagnosis"] or md_get(md, "age_at_diagnosis")
                entry["anatomical_sites"]               = entry["anatomical_sites"] or md_get(md, "anatomical_sites")
                entry["disease_phase"]                  = entry["disease_phase"] or md_get(md, "disease_phase")
                entry["library_selection_method"]       = entry["library_selection_method"] or md_get(md, "library_selection_method")
                entry["tissue_type"]                    = entry["tissue_type"] or md_get(md, "tissue_type")
                entry["tumor_classification"]           = entry["tumor_classification"] or md_get(md, "tumor_classification")
                entry["tumor_tissue_morphology"]        = entry["tumor_tissue_morphology"] or md_get(md, "tumor_tissue_morphology")
                entry["age_at_collection"]              = entry["age_at_collection"] or md_get(md, "age_at_collection")
                entry["library_strategy"]               = entry["library_strategy"] or md_get(md, "library_strategy")
                entry["library_source_material"]        = entry["library_source_material"] or md_get(md, "library_source_material")
                entry["preservation_method"]            = entry["preservation_method"] or md_get(md, "preservation_method")
                entry["tumor_grade"]                    = entry["tumor_grade"] or md_get(md, "tumor_grade")
                entry["specimen_molecular_analyte_type"]= entry["specimen_molecular_analyte_type"] or md_get(md, "specimen_molecular_analyte_type")
                entry["identifiers"]                    = entry["identifiers"] or md_get(md, "identifiers")

        if VERBOSE:
            print(f"[SAMP:{term}] p{page}: {len(done)}/{len(per_src)}")
        if all_done:
            break
        page += 1
    return per_src


def merge_samples(results):
    merged = {}
    for _, per_src in results:
        for src, info in per_src.items():
            ms = merged.setdefault(src, {})
            for samp_id, details in info["samples"].items():
                entry = ms.setdefault(
                    samp_id,
                    {
                        "subject_id": details["subject_id"],
                        "diagnoses": set(),
                        "depositions": "",
                        "age_at_diagnosis": "",
                        "anatomical_sites": "",
                        "disease_phase": "",
                        "library_selection_method": "",
                        "tissue_type": "",
                        "tumor_classification": "",
                        "tumor_tissue_morphology": "",
                        "age_at_collection": "",
                        "library_strategy": "",
                        "library_source_material": "",
                        "preservation_method": "",
                        "tumor_grade": "",
                        "specimen_molecular_analyte_type": "",
                        "identifiers": "",
                    }
                )
                entry["diagnoses"].update(details["diagnoses"])
                for k in (
                    "depositions", "age_at_diagnosis", "anatomical_sites", "disease_phase",
                    "library_selection_method", "tissue_type", "tumor_classification",
                    "tumor_tissue_morphology", "age_at_collection", "library_strategy",
                    "library_source_material", "preservation_method", "tumor_grade",
                    "specimen_molecular_analyte_type", "identifiers"
                ):
                    if not entry[k] and details[k]:
                        entry[k] = details[k]
    return merged


def classify_samples(merged_map, regexes):
    out = {}
    for src, smap in merged_map.items():
        o2 = {}
        for samp_id, details in smap.items():
            diags = details["diagnoses"]
            mt = {t for t, r in regexes.items() if any(r.search(v) for v in diags)}
            if mt:
                o2[samp_id] = {
                    "subject_id": details["subject_id"],
                    "diagnoses": set(diags),
                    "matched_types": mt,
                    **{k: details[k] for k in (
                        "depositions", "age_at_diagnosis", "anatomical_sites", "disease_phase",
                        "library_selection_method", "tissue_type", "tumor_classification",
                        "tumor_tissue_morphology", "age_at_collection", "library_strategy",
                        "library_source_material", "preservation_method", "tumor_grade",
                        "specimen_molecular_analyte_type", "identifiers"
                    )}
                }
        out[src] = o2
    return out


def filter_samples(matches, include):
    inc = set(include)
    rows = []
    for src, smap in matches.items():
        for samp_id, info in smap.items():
            if inc.issubset(info["matched_types"]):
                rows.append((
                    src,
                    samp_id,
                    info["subject_id"],
                    info["diagnoses"],
                    info["depositions"],
                    info["age_at_diagnosis"],
                    info["anatomical_sites"],
                    info["disease_phase"],
                    info["library_selection_method"],
                    info["tissue_type"],
                    info["tumor_classification"],
                    info["tumor_tissue_morphology"],
                    info["age_at_collection"],
                    info["library_strategy"],
                    info["library_source_material"],
                    info["preservation_method"],
                    info["tumor_grade"],
                    info["specimen_molecular_analyte_type"],
                    info["identifiers"],
                ))
    return rows


def export_samples(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SAMPLE_OUT_COLUMNS)
        for (
            src, samp_id, subj_id, diags, dep, a_dx, anat, dphase, lib_sel, tissue,
            tclass, morph, a_coll, lib_strat, lib_src, pres, tgrade, analyte, ids
        ) in rows:
            w.writerow([
                src,
                samp_id,
                subj_id,
                "|".join(sorted(diags)),
                dep,
                a_dx,
                anat,
                dphase,
                lib_sel,
                tissue,
                tclass,
                morph,
                a_coll,
                lib_strat,
                lib_src,
                pres,
                tgrade,
                analyte,
                ids,
            ])
    print(f"Wrote {len(rows):,} rows to {out_path}")


def write_idmap_parquet(sample_rows, cancer_type):
    """
    sample_rows: list of tuples in the same order as SAMPLE_OUT_COLUMNS
    Writes/updates data/gdm/indexes/idmap.parquet
    """
    if not sample_rows:
        return
    # Build a slim DataFrame with (source, sample_id, subject_id, diagnoses_text, study_id)
    df = pd.DataFrame(sample_rows, columns=SAMPLE_OUT_COLUMNS)
    # keep just these:
    slim = df[["source", "sample_id", "subject_id", "diagnosis"]].copy()
    slim["study_id"] = STUDY_IDS.get(cancer_type, None)
    slim["node_id_subject"] = "patient:" + slim["subject_id"].astype(str)
    slim["node_id_sample"]  = "sample:"  + slim["sample_id"].astype(str)

    outp = GDM_INDEX / "idmap.parquet"
    if outp.exists():
        old = pd.read_parquet(outp)
        slim = pd.concat([old, slim], ignore_index=True)

    slim = slim.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    slim.to_parquet(outp, compression="zstd", index=False)
    if VERBOSE:
        print(f"[GDM] wrote/updated {outp} ({len(slim):,} rows)")


# ─── Example: data_fetch() for interested cancer types ─────────────────────────

def data_fetch():
    # Subjects
    subj_res     = [(t, harvest_subjects_for_term(t)) for t in search_terms]
    merged_subj  = merge_subjects(subj_res)
    subj_matches = classify_entities(merged_subj, active_type_regexes)
    for ctype in active_type_terms:
        rows = filter_entities(subj_matches, [ctype])
        outf = SUBJECT_IDS_DIR / f"{ctype.replace(' ','_').lower()}.csv"
        export_subjects(rows, outf)
    
    # Samples
    samp_res     = [(t, harvest_samples_for_term(t)) for t in search_terms]
    merged_samp  = merge_samples(samp_res)
    samp_matches = classify_samples(merged_samp, active_type_regexes)
    for ctype in active_type_terms:
        rows = filter_samples(samp_matches, [ctype])
        outf = SAMPLE_IDS_DIR / f"{ctype.replace(' ','_').lower()}.csv"
        export_samples(rows, outf)
        # Also write/update GDM idmap for this cancer type
        write_idmap_parquet(rows, ctype)
    

if __name__ == "__main__":
    data_fetch()