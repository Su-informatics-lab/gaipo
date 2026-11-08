"""
cohort_table.py - characteristics of the modeling cohort per class (Table 2 of the manuscript).

The cohort is the patients data_process kept (clinical data and every configured omics
modality): labels_train.csv + labels_test.csv of each modeled study. Their clinical values are
read from the extract, so the table describes exactly the patients that were modeled:

    data/gdm/filtered/<portal>/<type>/<study_id>/labels_train.csv, labels_test.csv
    data/fetch/cbioportal_api_request/<portal>/<type>/<study_id>/clinical_patients.csv, clinical_samples.csv

Columns: one per class (label 1 first, as in the manuscript: DAWT, FHWT; High-grade, Low-grade),
then All. Class names: processor.labels.<type>.class_names, else from the label map
(src/cohort.py). Rows (the first column found is used; patient-level before sample-level):

    age         AGE (years), AGE_AT_DIAGNOSIS (years), AGE_IN_DAYS, AGE_AT_DIAGNOSIS_DAYS (days)
                -> No. (%) and median (IQR) overall and in 0-5, 6-11, 12+ years (<6, 6 to <12, >=12)
    sex, ethnicity, race   SEX, ETHNICITY, RACE (race: Black or African American, White, Other,
                Not Available; every other reported race is counted in Other and listed in the notes)
    os_status   OS_STATUS (1:DECEASED / 0:LIVING)
    os_time     OS_MONTHS, OS_DAYS -> follow-up, months, median (IQR)
    efs_status  EFS_STATUS, EFS_EVENT_TYPE, EVENT_TYPE (TARGET: blank = no event; Relapse,
                Progression = event)
    efs_time    EFS_MONTHS, EFS_DAYS, DAYS_TO_EVENT -> event-free follow-up, months
    stage       CLINICAL_STAGE, counted by its stage group (IIIB, III/V -> III)

A sample-level column is taken per patient over all of the patient's samples: the lowest value
for age (age at the first diagnosis), the first reported value otherwise. Days are converted
with 365.25 days per year and 30.4375 days per month. Override a column with
--column [Type:]variable=COLUMN[:years|months|days].

Outputs (data/analysis/cohort_table/): cohort_table.tsv and cohort_table.md (the table),
cohort_values.tsv (every source value and the row it was counted in, per class) and
cohort_patients.tsv (one row per patient with the values used).

    python scripts/cohort_table.py                         # every type in cancer_type_interest
    python scripts/cohort_table.py Glioma --column "Glioma:age=AGE_AT_DIAGNOSIS_DAYS:days"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cohort import FETCH_ROOT, FILTERED_ROOT, class_names, extract_dir, load_labels  # noqa: E402
from src.study_scope import modeling_studies  # noqa: E402

DAYS_PER = {"years": 365.25, "months": 30.4375, "days": 1.0}
NA = "Not Available"
EVENT, NO_EVENT = "Event", "No Event"

# variable -> candidate columns: (column, unit or None, value when missing)
CANDIDATES: Dict[str, List[Tuple[str, Optional[str], Optional[str]]]] = {
    "age": [("AGE", "years", None), ("AGE_AT_DIAGNOSIS", "years", None),
            ("AGE_IN_DAYS", "days", None), ("AGE_AT_DIAGNOSIS_DAYS", "days", None)],
    "sex": [("SEX", None, None)],
    "ethnicity": [("ETHNICITY", None, None)],
    "race": [("RACE", None, None)],
    "os_status": [("OS_STATUS", None, None)],
    "os_time": [("OS_MONTHS", "months", None), ("OS_DAYS", "days", None)],
    "efs_status": [("EFS_STATUS", None, None), ("EFS_EVENT_TYPE", None, None),
                   ("EVENT_TYPE", None, NO_EVENT)],
    "efs_time": [("EFS_MONTHS", "months", None), ("EFS_DAYS", "days", None), ("DAYS_TO_EVENT", "days", None)],
    "stage": [("CLINICAL_STAGE", None, None)],
}
NUMERIC = {"age", "os_time", "efs_time"}
TARGET_UNIT = {"age": "years", "os_time": "months", "efs_time": "months"}

MISSING = {"", "nan", "none", "na", "n/a", "unknown", "not reported", "not available", "unavailable",
           "not applicable", "unavailable/not reported", "other/unavailable/not reported",
           "reported unknown", "not allowed to collect", "[not available]", "[unknown]", "[not reported]"}
NO_EVENT_VALUES = {"no event", "none", "not applicable", "censored", "diseasefree", "disease free",
                   "event free", "eventfree", "no", "0"}
EFS_MISSING = MISSING - {"none", "not applicable"}

AGE_BINS = [("0-5", 0, 6), ("6-11", 6, 12), ("12+", 12, np.inf)]
STAGES = ["I", "II", "III", "IV", "V"]


# ---------------------------------------------------------------- value -> category
def _norm(v) -> Optional[str]:
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    return None if s.lower() in {"", "nan"} else s


def sex_category(v) -> str:
    s = (_norm(v) or "").lower()
    return {"female": "Female", "male": "Male", "f": "Female", "m": "Male"}.get(s, NA)


def ethnicity_category(v) -> str:
    s = (_norm(v) or "").lower()
    if "not hispanic" in s or "non-hispanic" in s or "non hispanic" in s:
        return "Not Hispanic or Latino"
    if "hispanic" in s or "latino" in s:
        return "Hispanic or Latino"
    return NA


def race_category(v) -> str:
    s = (_norm(v) or "").lower()
    if s in MISSING:
        return NA
    if s == "white":
        return "White"
    if s in {"black or african american", "black", "african american"}:
        return "Black or African American"
    return "Other"


def os_category(v) -> str:
    s = (_norm(v) or "").lower()
    if s.startswith("1:") or "deceased" in s or s in {"dead", "1"}:
        return "Deceased"
    if s.startswith("0:") or "living" in s or s in {"alive", "0"}:
        return "Living"
    return NA


def efs_category(v, when_missing: Optional[str]) -> str:
    s = _norm(v)
    if s is None:
        return when_missing or NA
    sl = s.lower()
    if sl.startswith("1:"):
        return EVENT
    if sl.startswith("0:"):
        return NO_EVENT
    if sl in NO_EVENT_VALUES:
        return NO_EVENT
    if sl in EFS_MISSING:
        return NA
    return EVENT              # Relapse, Progression, Recurrence, Deceased due to disease, ...


def stage_category(v) -> str:
    s = (_norm(v) or "").upper().replace("STAGE", "").strip()
    m = re.match(r"(IV|V|III|II|I)", s)            # IIIB, III/V -> III; IV/V -> IV
    return m.group(1) if m else NA


CATEGORIZE = {"sex": sex_category, "ethnicity": ethnicity_category, "race": race_category,
              "os_status": os_category, "stage": stage_category}
ROWS = {"sex": ["Female", "Male"],
        "ethnicity": ["Hispanic or Latino", "Not Hispanic or Latino", NA],
        "race": ["Black or African American", "White", "Other", NA],
        "os_status": ["Living", "Deceased"],
        "efs_status": [NO_EVENT, EVENT],
        "stage": ["I", "II", "III", "IV"]}
OPTIONAL_ROWS = {"sex": [NA], "os_status": [NA], "efs_status": [NA], "stage": ["V", NA]}
TITLES = {"age": "Age at diagnosis, years, No. (%); Median (IQR)",
          "sex": "Sex, No. (%)", "ethnicity": "Ethnicity, No. (%)", "race": "Race, No. (%)",
          "os_status": "Overall Survival (OS) Status, No. (%)",
          "os_time": "Follow-up time, months, Median (IQR)",
          "efs_status": "Event-Free Survival (EFS) Status, No. (%)",
          "efs_time": "Event-Free Follow-up time, months, Median (IQR)",
          "stage": "Clinical stage, No. (%)"}
ORDER = ["age", "sex", "ethnicity", "race", "os_status", "os_time", "efs_status", "efs_time", "stage"]


# ---------------------------------------------------------------- reading
def _overrides(items: List[str]) -> Dict[Tuple[Optional[str], str], Tuple[str, Optional[str]]]:
    """--column [Type:]variable=COLUMN[:unit] -> {(type or None, variable): (column, unit)}"""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--column {it!r}: expected [Type:]variable=COLUMN[:unit]")
        left, right = it.split("=", 1)
        ct, var = (left.rsplit(":", 1) if ":" in left else (None, left))
        var = var.strip()
        if var not in CANDIDATES:
            raise SystemExit(f"--column {it!r}: variable must be one of {', '.join(CANDIDATES)}")
        col, unit = right, None
        if ":" in right and right.rsplit(":", 1)[1].strip().lower() in DAYS_PER:
            col, unit = right.rsplit(":", 1)
            unit = unit.strip().lower()
        out[(ct.strip() if ct else None, var)] = (col.strip(), unit)
    return out


def _per_patient(sam: pd.DataFrame, col: str, numeric: bool) -> pd.Series:
    s = sam[["patientId", col]].copy()
    s["patientId"] = s["patientId"].astype(str).str.strip()
    if numeric:
        return pd.to_numeric(s[col], errors="coerce").groupby(s["patientId"]).min()
    s[col] = s[col].map(_norm)
    return s.dropna(subset=[col]).groupby("patientId")[col].first()


def collect(cohort: pd.DataFrame, pat: pd.DataFrame, sam: pd.DataFrame, ct: str,
            overrides) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Per patient raw value and category of every variable; {variable: source description}; warnings."""
    pat = pat.copy()
    pat["patientId"] = pat["patientId"].astype(str).str.strip()
    pat = pat.drop_duplicates("patientId").set_index("patientId")
    ids = cohort.patientId
    out = cohort.set_index("patientId")[["label", "split"]].copy()
    sources, warns = {}, []
    for var in ORDER:
        given = overrides.get((ct, var)) or overrides.get((None, var))
        cands = ([(given[0], given[1] or TARGET_UNIT.get(var), None)] if given else CANDIDATES[var])
        if given and var == "efs_status" and given[0] == "EVENT_TYPE":
            cands = [("EVENT_TYPE", None, NO_EVENT)]
        found = None
        for col, unit, when_missing in cands:
            if col in pat.columns:
                found = (col, unit, when_missing, "patients",
                         pat[col].reindex(ids).to_numpy() if var not in NUMERIC
                         else pd.to_numeric(pat[col], errors="coerce").reindex(ids).to_numpy())
                break
            if sam is not None and col in sam.columns and "patientId" in sam.columns:
                found = (col, unit, when_missing, "samples",
                         _per_patient(sam, col, var in NUMERIC).reindex(ids).to_numpy())
                break
        if found is None:
            if given:
                warns.append(f"{ct}: column {given[0]!r} for {var} is in neither clinical table")
            sources[var] = None
            continue
        col, unit, when_missing, level, raw = found
        out[f"{var}_raw"] = raw
        if var in NUMERIC:
            unit = unit or TARGET_UNIT[var]
            vals = pd.Series(raw, index=out.index, dtype=float) * DAYS_PER[unit] / DAYS_PER[TARGET_UNIT[var]]
            out[var] = vals
            conv = "" if unit == TARGET_UNIT[var] else f", {unit} converted to {TARGET_UNIT[var]}"
            agg = ", lowest over the patient's samples" if level == "samples" else ""
            sources[var] = f"{col} ({level}{conv}{agg})"
            if var == "age" and vals.notna().any() and vals.median() > 60:
                warns.append(f"{ct}: median age from {col} is {vals.median():.0f} years; if {col} is in days, "
                             f"add --column \"{ct}:age={col}:days\"")
        else:
            fn = (lambda v: efs_category(v, when_missing)) if var == "efs_status" else CATEGORIZE[var]
            out[var] = [fn(v) for v in raw]
            agg = ", first reported over the patient's samples" if level == "samples" else ""
            sources[var] = f"{col} ({level}{agg})" + (f"; blank = {when_missing}" if when_missing else "")
    return out, sources, warns


# ---------------------------------------------------------------- formatting
def _num(x: float, decimals_fixed: bool) -> str:
    s = f"{x:.1f}"
    return s if decimals_fixed or not s.endswith(".0") else s[:-2]


def _median_iqr(v: pd.Series, fixed: bool) -> str:
    v = v.dropna()
    if v.empty:
        return "-"
    return f"{_num(v.median(), fixed)} ({_num(v.quantile(.25), fixed)}-{_num(v.quantile(.75), fixed)})"


def _n_pct(n: int, total: int) -> str:
    if not total:
        return "0"
    return f"{n} (100)" if n == total else f"{n} ({100 * n / total:.1f})"


def column_cells(d: pd.DataFrame, sources: Dict[str, str]) -> Dict[str, str]:
    """Cells of one column (one class or All), keyed by row id."""
    total = len(d)
    cells = {}
    for var in ORDER:
        if sources.get(var) is None:
            cells[var] = "-"
            for r in _subs(var):
                cells[f"{var}:{r}"] = "-"
            continue
        if var == "age":
            a = d["age"]
            cells["age"] = f"{_n_pct(int(a.notna().sum()), total)}; {_median_iqr(a, False)}"
            for name, lo, hi in AGE_BINS:
                b = a[(a >= lo) & (a < hi)]
                cells[f"age:{name}"] = f"{_n_pct(len(b), total)}; {_median_iqr(b, False)}" if len(b) else _n_pct(0, total)
            cells[f"age:{NA}"] = _n_pct(int(a.isna().sum()), total)
        elif var in ("os_time", "efs_time"):
            cells[var] = _median_iqr(d[var], True)
        else:
            cells[var] = ""
            counts = d[var].value_counts()
            for r in ROWS[var] + OPTIONAL_ROWS.get(var, []):
                cells[f"{var}:{r}"] = _n_pct(int(counts.get(r, 0)), total)
    return cells


def _subs(var: str) -> List[str]:
    """Sub-rows of a variable (optional ones included)."""
    if var == "age":
        return [b[0] for b in AGE_BINS] + [NA]
    return ROWS.get(var, []) + OPTIONAL_ROWS.get(var, [])


def row_ids(blocks: List[dict]) -> List[Tuple[str, str]]:
    """(row id, label) in table order; optional rows only where some column has a count."""
    rows = []
    for var in ORDER:
        rows.append((var, TITLES[var]))
        for r in _subs(var):
            optional = (var == "age" and r == NA) or r in OPTIONAL_ROWS.get(var, [])
            if optional and not any(b["counts"].get(f"{var}:{r}", 0) for b in blocks):
                continue
            rows.append((f"{var}:{r}", f"  {r}"))
    return rows


def build_block(ct: str, portal: str, study: str, cfg: dict, overrides, fetch_root: Path,
                filtered_root: Path) -> dict:
    cohort = load_labels(portal, ct, study, filtered_root)
    raw_dir = extract_dir(portal, ct, study, fetch_root)
    pat_f, sam_f = raw_dir / "clinical_patients.csv", raw_dir / "clinical_samples.csv"
    if not pat_f.exists():
        raise FileNotFoundError(f"{pat_f} not found: run data_extract first")
    pat = pd.read_csv(pat_f, dtype={"patientId": str}, low_memory=False)
    sam = pd.read_csv(sam_f, dtype={"patientId": str, "sampleId": str}, low_memory=False) if sam_f.exists() else None
    d, sources, warns = collect(cohort, pat, sam, ct, overrides)
    missing = sorted(set(cohort.patientId) - set(pat["patientId"].astype(str).str.strip()))
    if missing:
        warns.append(f"{ct}: {len(missing)} modeled patients have no row in {pat_f.name} "
                     f"(e.g. {', '.join(missing[:3])})")
    names = class_names(cfg, ct)
    classes = sorted(d.label.unique(), reverse=True)            # label 1 first, as in the manuscript
    cols = []
    for k in classes:
        sub = d[d.label == k]
        cols.append((f"{names.get(int(k), f'label {k}')} (n = {len(sub)})", column_cells(sub, sources), sub))
    cols.append((f"All (n = {len(d)})", column_cells(d, sources), d))
    counts = {}
    for var in ORDER:
        if sources.get(var) is None or var in NUMERIC:
            continue
        for r, n in d[var].value_counts().items():
            counts[f"{var}:{r}"] = int(n)
    if sources.get("age"):
        counts[f"age:{NA}"] = int(d["age"].isna().sum())
    return {"ct": ct, "portal": portal, "study": study, "names": names, "cols": cols, "data": d,
            "sources": sources, "warns": warns, "counts": counts}


def value_table(block: dict) -> pd.DataFrame:
    """Every source value -> category, with patients per class: shows what was folded where."""
    d, names, rows = block["data"], block["names"], []
    for var in ORDER:
        if block["sources"].get(var) is None or var in NUMERIC:
            continue
        raw = d[f"{var}_raw"].map(lambda v: _norm(v) if _norm(v) is not None else "<blank>")
        g = pd.crosstab([raw, d[var]], d.label)
        for (value, cat), n in g.iterrows():
            row = {"cancer_type": block["ct"], "study": block["study"], "variable": var,
                   "source": block["sources"][var], "source_value": value, "counted_as": cat}
            for k in sorted(g.columns, reverse=True):
                row[names.get(int(k), f"label {k}")] = int(n[k])
            row["All"] = int(n.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def notes(block: dict) -> List[str]:
    lines = [f"**{block['ct']}** ({block['portal']}:{block['study']}): "
             + "; ".join(f"{v}: {s}" for v, s in block["sources"].items() if s) + "."]
    missing = [v for v, s in block["sources"].items() if s is None]
    if missing:
        lines.append(f"Not in the extract (shown as -): {', '.join(missing)}.")
    d = block["data"]
    if block["sources"].get("race"):
        other = d.loc[d.race == "Other", "race_raw"].map(_norm).value_counts()
        if len(other):
            lines.append("Race counted as Other: " + ", ".join(f"{k} ({v})" for k, v in sorted(other.items(), key=lambda kv: (-kv[1], kv[0]))) + ".")
    if block["sources"].get("efs_status"):
        ev = d.loc[d.efs_status == EVENT, "efs_status_raw"].map(_norm).value_counts()
        if len(ev):
            lines.append("EFS events: " + ", ".join(f"{k} ({v})" for k, v in sorted(ev.items(), key=lambda kv: (-kv[1], kv[0]))) + ".")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cancer_types", nargs="*", help="default: cancer_type_interest in the config")
    ap.add_argument("--config", default=None, help="default: $CONFIG_PATH or config/pipeline_config.yaml")
    ap.add_argument("--column", action="append", default=[],
                    help='[Type:]variable=COLUMN[:years|months|days], e.g. "Glioma:age=AGE_AT_DIAGNOSIS_DAYS:days"')
    ap.add_argument("--fetch-root", default=str(FETCH_ROOT))
    ap.add_argument("--filtered-root", default=str(FILTERED_ROOT))
    ap.add_argument("--out", default="data/analysis/cohort_table")
    args = ap.parse_args()

    import os
    cfg_path = args.config or os.getenv("CONFIG_PATH", "config/pipeline_config.yaml")
    cfg = yaml.safe_load(open(cfg_path)) or {}
    cts = args.cancer_types or list(cfg.get("cancer_type_interest") or [])
    overrides = _overrides(args.column)

    blocks = []
    for ct in cts:
        studies = modeling_studies(cfg, ct)
        if not studies:
            print(f"[cohort_table] {ct}: no study configured; skipping")
        for portal, study in studies:
            try:
                blocks.append(build_block(ct, portal, study, cfg, overrides, Path(args.fetch_root),
                                          Path(args.filtered_root)))
            except FileNotFoundError as e:
                print(f"[cohort_table] {ct} [{portal}:{study}]: {e}; skipping")
    if not blocks:
        raise SystemExit("[cohort_table] nothing to tabulate")

    multi = {b["ct"] for b in blocks if sum(x["ct"] == b["ct"] for x in blocks) > 1}
    rows = row_ids(blocks)
    group_row, header = [""], ["Category"]
    for b in blocks:
        title = f"{b['ct']} ({b['study']})" if b["ct"] in multi else b["ct"]
        for i, (name, _, _) in enumerate(b["cols"]):
            group_row.append(title if i == 0 else "")
            header.append(name)
    body = []
    for rid, label in rows:
        line = [label]
        for b in blocks:
            for _, cells, _ in b["cols"]:
                line.append(cells.get(rid, ""))
        body.append(line)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([group_row, header] + body).to_csv(out / "cohort_table.tsv", sep="\t", index=False, header=False)

    def md_row(cells):
        return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |"
    md = ["# Cohort characteristics (patients with clinical and complete omics data)", "",
          md_row(group_row), "|" + "---|" * len(header), md_row(header)]
    md += [md_row([c.replace("  ", "&nbsp;&nbsp;", 1) if c.startswith("  ") else c for c in r]) for r in body]
    md += ["", "Age groups: 0-5 = under 6 years, 6-11 = 6 to under 12, 12+ = 12 or older. "
           "Percentages are of all patients in the column. Days are converted with 365.25 days per year "
           "and 30.4375 days per month.", ""]
    for b in blocks:
        md += notes(b) + [""]
    warns = [w for b in blocks for w in b["warns"]]
    if warns:
        md += ["Warnings:", ""] + [f"- {w}" for w in warns] + [""]
    (out / "cohort_table.md").write_text("\n".join(md))

    pd.concat([value_table(b) for b in blocks], ignore_index=True).to_csv(
        out / "cohort_values.tsv", sep="\t", index=False)
    pats = []
    for b in blocks:
        d = b["data"].reset_index()
        d.insert(0, "cancer_type", b["ct"])
        d.insert(1, "study", b["study"])
        d.insert(3, "class", d.label.map(lambda k: b["names"].get(int(k), f"label {k}")))
        pats.append(d)
    pd.concat(pats, ignore_index=True).to_csv(out / "cohort_patients.tsv", sep="\t", index=False)

    for b in blocks:
        print(f"[cohort_table] {b['ct']} [{b['portal']}:{b['study']}]: "
              + ", ".join(name for name, _, _ in b["cols"]))
        for v, s in b["sources"].items():
            print(f"    {v:<10} {s or '- (not in the extract)'}")
    for w in warns:
        print(f"[cohort_table] WARN {w}")
    print(f"[cohort_table] wrote {out}/cohort_table.tsv, cohort_table.md, cohort_values.tsv, cohort_patients.tsv")


if __name__ == "__main__":
    main()
