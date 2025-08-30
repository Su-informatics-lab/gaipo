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
import itertools
import re
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
import yaml


# ─── User configures ────────────────────────────────────────────────────────
cfg_env = os.getenv("CONFIG_PATH")
cfg_path = Path(cfg_env) if cfg_env else ROOT / "config" / "pipeline_config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Service endpoints
BASE_SUBJECT = cfg["services"]["ccdi_subject_diagnosis"].rstrip("/")
BASE_SAMPLE = cfg["services"]["ccdi_sample_diagnosis"].rstrip("/")

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
custom_type_terms = cfg.get("custom_type_terms") or {}
cancer_type_interest = cfg["cancer_type_interest"]
TARGET_SOURCE = cfg.get("target_source")
STUDY_IDS = cfg.get("studyId", {})  # e.g., {"Wilms Tumor": "wt_target_2018_pub"}

# Output directories
ROOT = Path(__file__).resolve().parents[1]

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

def harvest_subjects_for_term(term):
    page = 1
    per_src = {}
    done = set()
    while True:
        blocks = _get_page(BASE_SUBJECT, term, page)
        all_done = True
        for b in blocks:
            src = b.get("source")
            if not src or b.get("errors"): continue
            if TARGET_SOURCE and src!=TARGET_SOURCE: continue

            total = b["summary"]["counts"]["all"]
            print(f"total: {total}")
            data = b["data"]
            print("data num:" + str(len(data)))
            info = per_src.setdefault(src, {"total":total, "seen":0, "subjects":defaultdict(set)})
            info["seen"] += len(data)
            if info["seen"] < total:
                all_done = False
            else:
                done.add(src)

            for subj in data:
                sid = subj["id"]["name"] # subj.get("id", {}).get("name")
                diags = subj["metadata"].get("associated_diagnoses") or [] # subj.get("metadata", {}).get("associated_diagnoses")
                for d in diags:
                    v = d.get("value")
                    if v: info["subjects"][sid].add(v)

        if VERBOSE:
            print(f"[SUBJ:{term}] p{page}: {len(done)}/{len(per_src)}")
        if all_done: break
        page += 1
    return per_src

def merge_subjects(results):
    merged = {}
    for _, per_src in results:
        for src, info in per_src.items():
            m = merged.setdefault(src, {})
            for sid, dx in info["subjects"].items():
                m.setdefault(sid, set()).update(dx)
    return merged

def classify_entities(merged_map, regexes):
    out = {}
    for src, emap in merged_map.items():
        o2 = {}
        for eid, diags in emap.items():
            mt = {t for t,r in regexes.items() if any(r.search(v) for v in diags)}
            if mt:
                o2[eid] = {"diagnoses":diags, "matched_types":mt}
        out[src] = o2
    return out

def filter_entities(matches, include):
    inc = set(include)
    rows = []
    for src, emap in matches.items():
        for eid, info in emap.items():
            if inc.issubset(info["matched_types"]):
                rows.append((src, eid, info["diagnoses"]))
    return rows

def export_subjects(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source","subject_id","diagnoses"])
        for src, sid, diags in rows:
            w.writerow([src, sid, "|".join(sorted(diags))])
    print(f"Wrote {len(rows):,} rows to {out_path}")

# ─── Sample pipeline ─────────────────────────────────────────────────

def harvest_samples_for_term(term):
    page = 1
    per_src = {}
    done = set()
    while True:
        blocks = _get_page(BASE_SAMPLE, term, page)
        all_done = True
        for b in blocks:
            src = b.get("source")
            if not src or b.get("errors"): continue
            if TARGET_SOURCE and src!=TARGET_SOURCE: continue

            total = b["summary"]["counts"]["all"]
            data = b["data"]
            info = per_src.setdefault(src, {"total":total, "seen":0, "samples":{}})
            info["seen"] += len(data)
            if info["seen"] < total:
                all_done = False
            else:
                done.add(src)

            for samp in data:
                samp_id = samp["id"]["name"]
                subj_id = samp["subject"]["name"]
                diag_entry = samp["metadata"].get("diagnosis")
                diag_val = diag_entry.get("value") if isinstance(diag_entry, dict) else None
                entry = info["samples"].setdefault(
                    samp_id,
                    {"subject_id": subj_id, "diagnoses": set()}
                )
                if isinstance(diag_val, str):
                    entry["diagnoses"].add(diag_val)

        if VERBOSE:
            print(f"[SAMP:{term}] p{page}: {len(done)}/{len(per_src)}")
        if all_done: break
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
                    {"subject_id": details["subject_id"], "diagnoses": set()}
                )
                entry["diagnoses"].update(details["diagnoses"])
    return merged

def classify_samples(merged_map, regexes):
    out = {}
    for src, smap in merged_map.items():
        o2 = {}
        for samp_id, details in smap.items():
            diags = details["diagnoses"]
            mt = {t for t,r in regexes.items() if any(r.search(v) for v in diags)}
            if mt:
                o2[samp_id] = {
                    "subject_id": details["subject_id"],
                    "diagnoses": diags,
                    "matched_types": mt
                }
        out[src] = o2
    return out

def filter_samples(matches, include):
    inc = set(include)
    rows = []
    for src, smap in matches.items():
        for samp_id, info in smap.items():
            if inc.issubset(info["matched_types"]):
                rows.append((src, samp_id, info["subject_id"], info["diagnoses"]))
    return rows

def export_samples(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source","sample_id","subject_id","diagnoses"])
        for src, samp_id, subj_id, diags in rows:
            w.writerow([src, samp_id, subj_id, "|".join(sorted(diags))])
    print(f"Wrote {len(rows):,} rows to {out_path}")


def write_idmap_parquet(sample_rows, cancer_type):
    """
    sample_rows: list[(source, sample_id, subject_id, diagnoses_set)]
    Writes/updates data/gdm/indexes/idmap.parquet
    """
    if not sample_rows:
        return
    study_id = STUDY_IDS.get(cancer_type, None)
    df = pd.DataFrame(sample_rows, columns=["source","sample_id","subject_id","diagnoses"])
    df["study_id"] = study_id
    df["node_id_subject"] = "patient:" + df["subject_id"].astype(str)
    df["node_id_sample"]  = "sample:"  + df["sample_id"].astype(str)

    outp = GDM_INDEX / "idmap.parquet"
    # append or create; drop dups
    if outp.exists():
        old = pd.read_parquet(outp)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    df.to_parquet(outp, compression="zstd", index=False)
    if VERBOSE:
        print(f"[GDM] wrote/updated {outp} ({len(df):,} rows)")


# ─── Example: data_fetch() for interested cancer types ──────────────────────────────────────────────────────────────────────

def data_fetch():
    # Subjects
    print(active_type_regexes)
    subj_res = [(t, harvest_subjects_for_term(t)) for t in search_terms]
    merged_subj = merge_subjects(subj_res)
    subj_matches = classify_entities(merged_subj, active_type_regexes)
    for ctype in active_type_terms:
        rows = filter_entities(subj_matches, [ctype])
        outf = SUBJECT_IDS_DIR / f"{ctype.replace(' ','_').lower()}.csv"
        export_subjects(rows, outf)

    # Samples
    samp_res = [(t, harvest_samples_for_term(t)) for t in search_terms]
    merged_samp = merge_samples(samp_res)
    samp_matches = classify_samples(merged_samp, active_type_regexes)
    for ctype in active_type_terms:
        rows = filter_samples(samp_matches, [ctype])
        outf = SAMPLE_IDS_DIR / f"{ctype.replace(' ','_').lower()}.csv"
        export_samples(rows, outf)
        # Also write GDM idmap for this cancer type
        write_idmap_parquet(rows, ctype)

if __name__ == "__main__":
    data_fetch()