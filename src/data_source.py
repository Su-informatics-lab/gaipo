# src/data_source.py
"""
data_source.py — Harvest CCDI Federation subjects & samples by cancer-type terms

What it does
------------
1) Builds the search for each type in cancer_type_interest (config `search.source`):
     - ontology (default in pipeline_config.yaml): query terms from the ontology layer, i.e.
       the NCIt concept set of the cohort (config/ontology/analysis_groups.tsv) and every
       label / ICD-O-3 code mapped to those concepts in the SSSOM files; see
       src/ontology_search.py
     - config: the fixed lists default_type_terms + custom_type_terms (previous behaviour),
       compiled to a word-boundary regex per type
2) Pages CCDI Federation endpoints to harvest (case-insensitive substring search):
     - /subject-diagnosis  → subject_id + diagnoses (per source)
     - /sample-diagnosis   → sample_id + subject_id + diagnoses (per source)
3) Assigns entities to cancer types: ontology mode resolves each diagnosis value to an NCIt
   concept and keeps it if the concept is in the cohort; config mode uses the regexes.
4) Writes per-type CSVs and an id map parquet for downstream steps, the term provenance
   (search_terms.tsv) and, in ontology mode with compare_with_config, a comparison with the
   fixed list on the same records (search_report.tsv, search_values.tsv).

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
  search_terms.tsv                # every term, its origin (NCIt / SSSOM row) and whether it was queried
  search_report.tsv, search_values.tsv   # ontology vs fixed list (ontology mode, compare_with_config)
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
import threading
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
default_type_terms = cfg.get("default_type_terms") or {}
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
combined_type_terms = {**(default_type_terms or {}), **custom_type_terms}

# Search mode (search: block in pipeline_config.yaml). Without the block: the fixed term list.
SEARCH_CFG    = cfg.get("search") or {}
SEARCH_SOURCE = str(SEARCH_CFG.get("source", "config")).lower()   # ontology | config

# Validate interest list (the fixed list is required only when it is used)
if SEARCH_SOURCE == "config":
    for t in cancer_type_interest:
        if t not in combined_type_terms:
            raise KeyError(f"Interest {t!r} not in defined types")

# Restrict to active
active_type_terms = {
    t: sorted(combined_type_terms[t])
    for t in cancer_type_interest
    if t in combined_type_terms
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


def _fixed_lists(types):
    """Fixed term lists and their word-boundary regexes for the given cancer types."""
    terms = {t: sorted(combined_type_terms[t]) for t in types if t in combined_type_terms}
    regexes = {t: re.compile(r"\b(?:" + "|".join(re.escape(x) for x in terms[t]) + r")\b", re.I)
               for t in terms}
    return terms, regexes


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

_LOCAL = threading.local()


def _session():
    """One requests.Session per thread (queries can run in parallel, see search.max_workers)."""
    if not hasattr(_LOCAL, "s"):
        _LOCAL.s = requests.Session()
        _LOCAL.s.headers.update(SESSION.headers)
    return _LOCAL.s


def _get_page(base, term, page, per_page=PER_PAGE):
    params = {"search": term, "page": page, "per_page": per_page}
    for i in range(5):
        r = _session().get(base, params=params, timeout=TIMEOUT)
        if r.status_code in RETRY_STATUS:
            time.sleep(2**i)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


# Per-source responses that returned no records, so a run does not silently lose a source.
# Each (endpoint, source, kind) is printed once; all occurrences go to search_skipped.tsv.
SKIPPED = []
_SKIP_LOCK = threading.Lock()
_SKIP_PRINTED = set()


def _skip(endpoint, term, page, src, kind, message):
    with _SKIP_LOCK:
        SKIPPED.append({"endpoint": endpoint, "source": src, "kind": kind, "message": message,
                        "search": term, "page": page})
        key = (endpoint, src, kind)
        if key in _SKIP_PRINTED:
            return
        _SKIP_PRINTED.add(key)
    print(f"[WARN] {endpoint}: source {src} returned no records ({kind}: {message[:160]}); "
          f"further occurrences are counted in search_skipped.tsv")


def _usable_blocks(blocks, endpoint, term, page, complete=()):
    """Yield (source, total, data) for each per-source block that carries results.
    `complete` = sources whose records for this search were all received on earlier pages;
    their "unable to calculate offset" reply to a page past the end is expected and ignored."""
    if not isinstance(blocks, list):
        _skip(endpoint, term, page, "-", "BadResponse", f"response is a {type(blocks).__name__}: {str(blocks)[:300]}")
        return
    for b in blocks:
        if not isinstance(b, dict):
            _skip(endpoint, term, page, "-", "BadResponse", f"unexpected item: {str(b)[:300]}")
            continue
        src = b.get("source")
        if TARGET_SOURCE and src != TARGET_SOURCE:
            continue
        if not src or b.get("errors"):
            errs = b.get("errors")
            first = errs[0] if isinstance(errs, list) and errs and isinstance(errs[0], dict) else {}
            kind, msg = str(first.get("kind", "Error")), str(first.get("message", json.dumps(errs, default=str)))
            if src in complete and "offset" in msg.lower():
                continue                                    # paged past this source's last record
            _skip(endpoint, term, page, src or "-", kind, msg)
            continue
        counts = (b.get("summary") or {}).get("counts") or {}
        data = b.get("data")
        if "all" not in counts or not isinstance(data, list):
            _skip(endpoint, term, page, src, "BadResponse", "no summary/data, keys "
                  + ",".join(sorted(map(str, b))) + ": " + json.dumps(b, default=str)[:300])
            continue
        yield src, counts["all"], data


def _finish_term(tag, endpoint, term, pages, per_src):
    """Flag sources that sent fewer records than they reported; print one line per search."""
    for src, info in per_src.items():
        if info["seen"] < info["total"]:
            _skip(endpoint, term, pages, src, "Incomplete",
                  f"received {info['seen']} of {info['total']} records")
    if VERBOSE:
        got = ", ".join(f"{s} {i['seen']}" for s, i in sorted(per_src.items())) or "no records"
        print(f"[{tag}:{term}] {pages} page(s): {got}")


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
        progressed = False
        for src, total, data in _usable_blocks(blocks, "subject-diagnosis", term, page, complete=done):
            info  = per_src.setdefault(
                src,
                {"total": total, "seen": 0, "subjects": {}}
            )
            info["seen"] += len(data)
            progressed = progressed or bool(data)
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

        if all_done or not progressed:      # no source sent records on this page: stop paging
            break
        page += 1
    _finish_term("SUBJ", "subject-diagnosis", term, page, per_src)
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
        progressed = False
        for src, total, data in _usable_blocks(blocks, "sample-diagnosis", term, page, complete=done):
            info  = per_src.setdefault(
                src,
                {"total": total, "seen": 0, "samples": {}}
            )
            info["seen"] += len(data)
            progressed = progressed or bool(data)
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

        if all_done or not progressed:      # no source sent records on this page: stop paging
            break
        page += 1
    _finish_term("SAMP", "sample-diagnosis", term, page, per_src)
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


# ─── Search plan: query terms + cohort matchers ────────────────────────────────

def search_plan(source=None, compare_with_config=None, types=None):
    """
    Returns (queries, matchers, compare, terms_table).
      queries   API search strings (union over cancer types, minimal set)
      matchers  {cancer_type: object with .search(value)} deciding cohort membership
      compare   {cancer_type: fixed-list regex} for the ontology-vs-fixed-list report ({} = none)
      terms_table  provenance of every term (written as search_terms.tsv)
    source "config":   the fixed lists default_type_terms + custom_type_terms (previous behaviour).
    source "ontology": terms and membership from the ontology layer (src/ontology_search.py).
    """
    source = (source or SEARCH_SOURCE).lower()
    types = list(types or cancer_type_interest)
    fixed_terms, fixed_regexes = _fixed_lists(types)
    if source == "config":
        missing = [t for t in types if t not in fixed_terms]
        if missing:
            raise KeyError(f"No default_type_terms/custom_type_terms for {missing}")
        table = pd.DataFrame([{"cancer_type": t, "term": x, "origin": "default_type_terms/custom_type_terms",
                               "queried": True} for t, xs in fixed_terms.items() for x in xs])
        return [x for xs in fixed_terms.values() for x in xs], fixed_regexes, {}, table
    if source != "ontology":
        raise ValueError(f"search.source must be 'ontology' or 'config', not {source!r}")

    from .ontology import OntologyLayer
    from .ontology_search import CohortMatcher, cohort_terms, minimal_queries

    onto_dir = ROOT / SEARCH_CFG.get("ontology_dir", "config/ontology")
    onto = OntologyLayer(onto_dir, ROOT / SEARCH_CFG.get("schemas_dir", "src/gdcdictionary/schemas"))
    cohorts = SEARCH_CFG.get("cohorts") or {}
    matchers, tables, queries = {}, [], []
    for t in types:
        group = cohorts.get(t, t.replace(" ", "_").lower())
        ct = cohort_terms(group, onto_dir, min_length=int(SEARCH_CFG.get("min_length", 4)),
                          query_codes=bool(SEARCH_CFG.get("query_codes", False)))
        matchers[t] = CohortMatcher(onto, ct, include_unresolved=bool(SEARCH_CFG.get("include_unresolved", True)))
        tables.append(ct.table.assign(cancer_type=t, cohort=group))
        queries += ct.table.loc[ct.table.queried, "term"].tolist()
        if VERBOSE:
            print(f"[search] {t}: cohort {group!r}, {len(ct.concepts)} NCIt concepts, "
                  f"{len(ct.table)} terms, {int(ct.table.queried.sum())} queries")

    if compare_with_config is None:
        compare_with_config = bool(SEARCH_CFG.get("compare_with_config", True))
    compare = {t: fixed_regexes[t] for t in matchers if t in fixed_regexes} if compare_with_config else {}
    for t in compare:                     # query the fixed list too, so both selections see the same records
        queries += fixed_terms[t]
    table = pd.concat(tables, ignore_index=True)[["cancer_type", "cohort", "term", "origin",
                                                  "concept_id", "predicate", "queried"]]
    return minimal_queries(queries), matchers, compare, table


def write_search_report(merged_subj, merged_samp, matchers, compare, out_dir):
    """
    Ontology selection vs the fixed term list, on the same retrieved records:
      search_report.tsv  counts per cancer type and entity (subject / sample)
      search_values.tsv  every diagnosis value either selection matched, with the ontology decision
    """
    from collections import Counter
    summary, values = [], []
    for entity, merged, key in (("subject", merged_subj, "diagnosis"), ("sample", merged_samp, "diagnoses")):
        for ct, m in matchers.items():
            if ct not in compare:
                continue
            fixed = compare[ct]
            ont_ids, fix_ids, seen = set(), set(), Counter()
            for src, emap in merged.items():
                for eid, row in emap.items():
                    diags = row[key]
                    o = any(m.search(v) for v in diags)
                    f = any(fixed.search(v) for v in diags)
                    if o:
                        ont_ids.add((src, eid))
                    if f:
                        fix_ids.add((src, eid))
                    for v in diags:
                        seen[v] += 1
            summary.append({
                "cancer_type": ct, "entity": entity, "cohort": m.cohort.group,
                "ontology_concepts": len(m.cohort.concepts), "ontology_terms": len(m.cohort.terms),
                "fixed_list_terms": len(_fixed_lists([ct])[0].get(ct, [])),
                "n_ontology": len(ont_ids), "n_fixed_list": len(fix_ids), "n_both": len(ont_ids & fix_ids),
                "n_ontology_only": len(ont_ids - fix_ids), "n_fixed_list_only": len(fix_ids - ont_ids),
            })
            for v, n in seen.items():
                decision, cid, label = m.explain(v)
                by_fixed = bool(fixed.search(v))
                if decision == "no_match" and not by_fixed:
                    continue
                values.append({"cancer_type": ct, "entity": entity, "diagnosis": v, "n_records": n,
                               "ontology_decision": decision, "concept_id": cid, "concept_label": label,
                               "selected_by_ontology": m.search(v), "selected_by_fixed_list": by_fixed})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out_dir / "search_report.tsv", sep="\t", index=False)
    vals = pd.DataFrame(values)
    if not vals.empty:
        vals = vals.sort_values(["cancer_type", "entity", "n_records"], ascending=[True, True, False])
    vals.to_csv(out_dir / "search_values.tsv", sep="\t", index=False)
    print(f"[search] wrote {out_dir}/search_report.tsv and search_values.tsv")
    return pd.DataFrame(summary)


# ─── data_fetch() for interested cancer types ──────────────────────────────────

def _harvest(fn, queries):
    """Run fn over the queries, search.max_workers at a time; results keep the query order."""
    workers = max(1, int(SEARCH_CFG.get("max_workers", 4)))
    if workers == 1 or len(queries) < 2:
        return [(t, fn(t)) for t in queries]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(queries))) as ex:
        return list(zip(queries, ex.map(fn, queries)))


def data_fetch(write_cohorts=True, report_dir=None, source=None, compare_with_config=None, types=None):
    """
    write_cohorts=False runs the search and writes only search_terms.tsv and the comparison
    report (to report_dir), leaving subject_ids/, sample_ids/ and idmap.parquet untouched.
    """
    queries, matchers, compare, terms_table = search_plan(source, compare_with_config, types)
    SKIPPED.clear()
    _SKIP_PRINTED.clear()
    report_dir = Path(report_dir) if report_dir else SUBJECT_IDS_DIR.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    terms_table.to_csv(report_dir / "search_terms.tsv", sep="\t", index=False)
    if VERBOSE:
        print(f"[search] source={(source or SEARCH_SOURCE)}: {len(queries)} API queries: {queries}")

    # Subjects
    subj_res     = _harvest(harvest_subjects_for_term, queries)
    merged_subj  = merge_subjects(subj_res)
    subj_matches = classify_entities(merged_subj, matchers)
    if write_cohorts:
        for ctype in matchers:
            rows = filter_entities(subj_matches, [ctype])
            outf = SUBJECT_IDS_DIR / f"{ctype.replace(' ','_').lower()}.csv"
            export_subjects(rows, outf)

    # Samples
    samp_res     = _harvest(harvest_samples_for_term, queries)
    merged_samp  = merge_samples(samp_res)
    samp_matches = classify_samples(merged_samp, matchers)
    if write_cohorts:
        for ctype in matchers:
            rows = filter_samples(samp_matches, [ctype])
            outf = SAMPLE_IDS_DIR / f"{ctype.replace(' ','_').lower()}.csv"
            export_samples(rows, outf)
            # Also write/update GDM idmap for this cancer type
            write_idmap_parquet(rows, ctype)

    if compare:
        write_search_report(merged_subj, merged_samp, matchers, compare, report_dir)
    write_skipped(report_dir)


def write_skipped(out_dir):
    """search_skipped.tsv: one row per endpoint x source x kind, with how often and for which searches."""
    cols = ["endpoint", "source", "kind", "message", "n_responses", "searches"]
    raw = pd.DataFrame(SKIPPED, columns=["endpoint", "source", "kind", "message", "search", "page"])
    if raw.empty:
        pd.DataFrame(columns=cols).to_csv(Path(out_dir) / "search_skipped.tsv", sep="\t", index=False)
        return
    agg = (raw.groupby(["endpoint", "source", "kind"], sort=True)
              .agg(message=("message", "first"), n_responses=("search", "size"),
                   searches=("search", lambda x: "; ".join(dict.fromkeys(map(str, x)))))
              .reset_index()[cols])
    agg.to_csv(Path(out_dir) / "search_skipped.tsv", sep="\t", index=False)
    print("[search] sources that returned no records (details in search_skipped.tsv):")
    for r in agg.itertuples(index=False):
        print(f"   {r.endpoint:18} {r.source:12} {r.kind:18} x{r.n_responses}")
    if (agg.kind == "Incomplete").any():
        print("[WARN] some sources sent fewer records than they reported (kind Incomplete); "
              "lower per_page or rerun before using the results.")


if __name__ == "__main__":
    data_fetch()