"""
label_concordance.py - From the data model to the modeling cohort: how the harmonized NCIt concept
of each case relates to the classification label used for training.

For one cancer type and study it follows every case from the data-model step to the modeling
cohort, and compares the grade implied by the case's NCIt concept (GDC diagnosis.tumor_grade,
derived through concept_relations.tsv: is_a + has_grade) with the label that data_process read
from the source column (processor.labels.<type>.column / map in pipeline_config.yaml):

    cases in the study -> resolved to an NCIt concept -> in the cohort concept set
      -> concept-derived tumor_grade (Low Grade / High Grade / not determined)
      -> modeling cohort (labels_train.csv + labels_test.csv) -> label classes

With --compare, it also compares the concept-derived grade of the patients that two studies share
(e.g. pbta_all and brain_cptac_2020).

Inputs (run after scripts/ontology_report.py):
    data/metadata/ontology_report/<Type>__<portal>__<study>_concept_annotations.tsv
    labels of the modeling cohort, from either
      data/gdm/filtered/<portal>/<type>/<study>/labels_train.csv, labels_test.csv   (written by data_process), or
      the current extract, rebuilt read-only with processor.load_and_prepare (same label map, omics
      alignment and train/test split as data_process; nothing is written under data/gdm)
    data/fetch/cbioportal_api_request/<portal>/<type>/<study>/clinical_samples.csv   (source label values)

Outputs (data/metadata/label_concordance/<Type>__<portal>__<study>/):
    flow.tsv          counts at each step (the numbers for a flow diagram)
    crosstab.tsv      concept-derived grade x label class, for the modeling cohort
    patients.tsv      one row per case: concept, predicate, grade, in cohort?, label, split
    shared.tsv        with --compare: grade in each study for the shared patients
    summary.md        all of the above, with agreement and Cohen's kappa

    python scripts/label_concordance.py Glioma --study pedcbio:pbta_all --compare cbio:brain_cptac_2020
    python scripts/label_concordance.py Glioma --study pedcbio:pbta_all --grade-of-class "0=Low Grade,1=High Grade"
    python scripts/label_concordance.py Glioma --study pedcbio:pbta_all --labels-dir path/to/old/run/pbta_all

--labels auto (default) uses labels_train/test.csv when they exist and otherwise rebuilds the labels
from the extract; --labels files / --labels extract forces one source.

Read-only for the pipeline.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
GRADES = ("Low Grade", "High Grade")
NOT_DETERMINED = "not determined by concept"


def slug(s: str) -> str:
    return str(s).strip().replace(" ", "_").lower()


def cohen_kappa(a: pd.Series, b: pd.Series) -> Tuple[int, int, float]:
    """(agreeing, n, kappa) for two aligned categorical series."""
    n = len(a)
    if n == 0:
        return 0, 0, float("nan")
    agree = int((a.values == b.values).sum())
    po = agree / n
    cats = set(a) | set(b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    kappa = float("nan") if pe == 1 else (po - pe) / (1 - pe)
    return agree, n, kappa


def grade_of_class(label_map: Optional[str], override: Optional[str]) -> Dict[int, str]:
    """{0: "Low Grade", 1: "High Grade"} from --grade-of-class, else from the label-map keys."""
    if override:
        out = {}
        for tok in override.split(","):
            k, v = tok.split("=", 1)
            out[int(k.strip())] = v.strip()
        return out
    out: Dict[int, str] = {}
    for tok in str(label_map or "").split(","):
        if "=" not in tok:
            continue
        k, v = tok.rsplit("=", 1)
        k = k.lower()
        g = "Low Grade" if "low" in k else "High Grade" if "high" in k else None
        if g:
            if out.get(int(v), g) != g:
                raise SystemExit(f"label map class {v} mixes low and high grade; pass --grade-of-class")
            out[int(v)] = g
    if set(out.values()) != set(GRADES):
        raise SystemExit(f"cannot tell which label class is low/high grade from map {label_map!r}; "
                         "pass --grade-of-class \"0=Low Grade,1=High Grade\"")
    return out


def labels_from_files(mdir: Path) -> pd.DataFrame:
    y = []
    for split in ("train", "test"):
        s = pd.read_csv(mdir / f"labels_{split}.csv", index_col=0).iloc[:, 0]
        y.append(pd.DataFrame({"label": s.astype(int).values, "split": split}, index=s.index.astype(str)))
    return pd.concat(y)


def labels_from_extract(cfg: dict, ct: str, portal: str, study: str, lc: dict) -> pd.DataFrame:
    """Modeling-cohort labels exactly as data_process builds them, without feature selection or writing."""
    from src.processor import MODALITY_SYNONYMS, OMICS_MODALITIES, load_and_prepare
    raw = Path("data/fetch/cbioportal_api_request") / portal / slug(ct) / study
    if not raw.exists():
        raise SystemExit(f"{raw} not found: run data_extract first")
    mods = [MODALITY_SYNONYMS.get(m.lower(), m.lower()) for m in (cfg.get("expression_profiles") or {}).get(ct, [])]
    mods = [m for m in mods if m in OMICS_MODALITIES]
    ccdi = Path("data/fetch/ccdi_api_request/sample_ids") / f"{slug(ct)}.csv"
    try:
        out = load_and_prepare(rawData_dir=raw, ccdi_csv=ccdi if ccdi.exists() else None, cancer_type=ct,
                               modalities=mods, label_source=lc.get("source", "samples"),
                               label_column=lc.get("column"), test_size=float(lc.get("test_size", 0.2)),
                               random_state=int(lc.get("seed", 42)), label_map=lc.get("map"),
                               select_features=False)
    except ValueError as e:
        raise SystemExit(f"could not build the labels as data_process would: {e}\n"
                         f"Fix processor.labels.{ct} in the config (README, 'Labels for process'), "
                         "or pass --labels-dir with the labels of the run you modeled.")
    y_tr, y_te = out[2], out[3]
    return pd.concat([pd.DataFrame({"label": v.astype(int).values, "split": sp}, index=v.index.astype(str))
                      for sp, v in (("train", y_tr), ("test", y_te))])


def load_annotations(report_dir: Path, ct: str, portal: str, study: str) -> pd.DataFrame:
    f = report_dir / f"{ct.replace(' ', '_')}__{portal}__{study}_concept_annotations.tsv"
    if not f.exists():
        raise SystemExit(f"{f} not found: run python scripts/ontology_report.py \"{ct}\" first")
    a = pd.read_csv(f, sep="\t", dtype=str)
    if "gdc_tumor_grade" not in a.columns:
        raise SystemExit(f"{f} has no gdc_tumor_grade column: rerun scripts/ontology_report.py with this version")
    a["subject_id"] = a.subject_id.astype(str)
    return a.set_index("subject_id")


def concept_status(a: pd.DataFrame, cohort: set) -> pd.Series:
    st = pd.Series("in cohort", index=a.index)
    st[a.disease_concept_id.isna()] = "unresolved"
    st[a.disease_concept_id.notna() & ~a.disease_concept_id.isin(cohort)] = "outside cohort"
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cancer_type")
    ap.add_argument("--study", required=True, help="portal:study_id of the modeling cohort, e.g. pedcbio:pbta_all")
    ap.add_argument("--compare", help="portal:study_id of a second study sharing patients")
    ap.add_argument("--group", help="cohort in analysis_groups.tsv (default: search.cohorts in the config)")
    ap.add_argument("--grade-of-class", help='e.g. "0=Low Grade,1=High Grade" (default: from the label map)')
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--report-dir", default="data/metadata/ontology_report")
    ap.add_argument("--labels", choices=("auto", "files", "extract"), default="auto",
                    help="where the modeling-cohort labels come from (see above)")
    ap.add_argument("--labels-dir", help="folder with labels_train.csv / labels_test.csv "
                                         "(default data/gdm/filtered/<portal>/<type>/<study>)")
    ap.add_argument("--out", default="data/metadata/label_concordance")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}
    labels_cfg = ((cfg.get("processor") or {}).get("labels") or {})
    ct = next((k for k in labels_cfg if slug(k) == slug(args.cancer_type)), args.cancer_type)
    lc = labels_cfg.get(ct) or {}
    group = args.group or ((cfg.get("search") or {}).get("cohorts") or {}).get(ct) or slug(ct)
    groups = pd.read_csv(ROOT / "config/ontology/analysis_groups.tsv", sep="\t", comment="#", dtype=str)
    cohort = set(groups.loc[groups.group == group, "concept_id"])
    if not cohort:
        raise SystemExit(f"no concepts for group {group!r} in config/ontology/analysis_groups.tsv")
    portal, study = args.study.split(":", 1)
    g_of = grade_of_class(lc.get("map"), args.grade_of_class)

    a = load_annotations(Path(args.report_dir), ct, portal, study)
    a["status"] = concept_status(a, cohort)
    a["concept_grade"] = a.gdc_tumor_grade.where(a.gdc_tumor_grade.isin(GRADES), NOT_DETERMINED)

    # modeling cohort
    mdir = Path(args.labels_dir) if args.labels_dir else Path("data/gdm/filtered") / portal / slug(ct) / study
    have_files = all((mdir / f"labels_{sp}.csv").exists() for sp in ("train", "test"))
    if args.labels == "files" and not have_files:
        raise SystemExit(f"{mdir}/labels_train.csv or labels_test.csv not found")
    if have_files and args.labels != "extract":
        y = labels_from_files(mdir)
        label_source = f"{mdir}/labels_train.csv + labels_test.csv"
    else:
        print(f"[labels] rebuilding the modeling-cohort labels from the extract (read-only), as data_process would")
        y = labels_from_extract(cfg, ct, portal, study, lc)
        label_source = ("rebuilt from the current extract with processor.load_and_prepare "
                        "(same label map, omics alignment and split as data_process)")
    y["label_grade"] = y.label.map(g_of)

    # source label values (display only)
    col = lc.get("column")
    sfile = Path("data/fetch/cbioportal_api_request") / portal / slug(ct) / study / "clinical_samples.csv"
    if col and sfile.exists():
        smp = pd.read_csv(sfile, dtype=str)
        pid = next((c for c in ("patientId", "PATIENT_ID", "subject_id") if c in smp.columns), None)
        if pid and col in smp.columns:
            raw = smp.groupby(pid)[col].agg(lambda v: " | ".join(sorted(set(v.dropna()))))
            y[f"source_{col}"] = raw.reindex(y.index).values

    pts = a[["disease_concept_id", "disease_concept_label", "disease_predicate", "disease_source_value",
             "status", "concept_grade"]].join(y, how="outer")
    pts["in_modeling_cohort"] = pts.label.notna()
    pts.loc[pts.status.isna(), "status"] = "not in the clinical extract"
    pts.index.name = "patient_id"

    m = pts[pts.in_modeling_cohort]
    det = m[m.concept_grade.isin(GRADES)]
    agree, n_det, kappa = cohen_kappa(det.concept_grade, det.label_grade)

    flow = [
        ("cases in the clinical extract", len(a)),
        ("  resolved to an NCIt concept", int(a.disease_concept_id.notna().sum())),
        ("    in the cohort concept set", int((a.status == "in cohort").sum())),
        ("      tumor_grade Low Grade (from concept)", int(((a.status == "in cohort") & (a.concept_grade == "Low Grade")).sum())),
        ("      tumor_grade High Grade (from concept)", int(((a.status == "in cohort") & (a.concept_grade == "High Grade")).sum())),
        (f"      {NOT_DETERMINED}", int(((a.status == "in cohort") & (a.concept_grade == NOT_DETERMINED)).sum())),
        ("    outside the cohort concept set", int((a.status == "outside cohort").sum())),
        ("  unresolved (review queue)", int((a.status == "unresolved").sum())),
        ("modeling cohort (labels_train + labels_test)", len(m)),
    ]
    for cls in sorted(g_of):
        flow.append((f"  label {cls} ({g_of[cls]})", int((m.label == cls).sum())))
    for sp in ("train", "test"):
        flow.append((f"  {sp}", int((m.split == sp).sum())))
    for st in ("outside cohort", "unresolved", "not in the clinical extract"):
        flow.append((f"  modeling cohort: {st}", int((m.status == st).sum())))
    flow.append(("  concept grade determined", n_det))
    flow.append(("  concept grade agrees with label", agree))
    flow = pd.DataFrame(flow, columns=["step", "n"])

    xt = pd.crosstab(m.concept_grade.where(m.status == "in cohort", m.status), m.label_grade,
                     margins=True, margins_name="total")

    out = Path(args.out) / f"{ct.replace(' ', '_')}__{portal}__{study}"
    out.mkdir(parents=True, exist_ok=True)
    flow.to_csv(out / "flow.tsv", sep="\t", index=False)
    xt.to_csv(out / "crosstab.tsv", sep="\t")
    pts.sort_values(["in_modeling_cohort", "status", "concept_grade"], ascending=[False, True, True]) \
       .to_csv(out / "patients.tsv", sep="\t")

    md = [f"# {ct}: data model to modeling cohort ({portal}:{study})", "",
          f"Cohort concept set: `{group}` ({len(cohort)} NCIt concepts). Label column: "
          f"`{lc.get('source', 'samples')}.{col}`, map `{lc.get('map')}` "
          f"({', '.join(f'{k} = {v}' for k, v in sorted(g_of.items()))}).", "",
          f"Modeling-cohort labels: {label_source}.", "",
          "## Flow", "", "| Step | n |", "| --- | ---: |"]
    md += [f"| {s.replace('  ', '&nbsp;&nbsp;')} | {n} |" for s, n in flow.itertuples(index=False)]
    md += ["", "## Concept-derived grade x label (modeling cohort)", "",
           "| concept grade \\ label | " + " | ".join(map(str, xt.columns)) + " |",
           "| --- |" + " ---: |" * len(xt.columns)]
    md += [f"| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in xt.iterrows()]
    md += ["", f"Agreement where the concept implies a grade: {agree} of {n_det} "
               f"({(agree / n_det if n_det else float('nan')):.1%}), Cohen's kappa {kappa:.3f}."]

    if args.compare:
        p2, s2 = args.compare.split(":", 1)
        b = load_annotations(Path(args.report_dir), ct, p2, s2)
        b["concept_grade"] = b.gdc_tumor_grade.where(b.gdc_tumor_grade.isin(GRADES), NOT_DETERMINED)
        shared = a.index.intersection(b.index)
        sh = pd.DataFrame({f"{study}_concept": a.loc[shared, "disease_concept_label"],
                           f"{study}_grade": a.loc[shared, "concept_grade"],
                           f"{s2}_concept": b.loc[shared, "disease_concept_label"],
                           f"{s2}_grade": b.loc[shared, "concept_grade"]})
        sh.index.name = "patient_id"
        sh.to_csv(out / "shared.tsv", sep="\t")
        both = sh[sh[f"{study}_grade"].isin(GRADES) & sh[f"{s2}_grade"].isin(GRADES)]
        ag2, n2, k2 = cohen_kappa(both[f"{study}_grade"], both[f"{s2}_grade"])
        xt2 = pd.crosstab(sh[f"{study}_grade"], sh[f"{s2}_grade"])
        md += ["", f"## Patients in both {study} and {s2}", "",
               f"{len(shared)} shared patients; concept-derived grade agrees in {ag2} of {n2} with a grade "
               f"in both studies (Cohen's kappa {k2:.3f}).", "",
               f"| {study} \\ {s2} | " + " | ".join(map(str, xt2.columns)) + " |",
               "| --- |" + " ---: |" * len(xt2.columns)]
        md += [f"| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in xt2.iterrows()]

    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
