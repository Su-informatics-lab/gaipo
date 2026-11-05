"""
label_concordance.py - Glioma grade from the data model to the modeling cohort.

The data model records, for every case, the grade implied by its NCIt concept (GDC
diagnosis.tumor_grade, derived through concept_relations.tsv: is_a + has_grade). This script
checks that grade at three levels:

  1. data model (always): cases -> resolved to an NCIt concept -> in the cohort concept set ->
     tumor_grade Low / High / not implied; and agreement with the grade implied by a second,
     coarser source field of the same case (--source-column, default the label column, e.g.
     CANCER_TYPE), resolved through the same ontology. Cases whose concept came from that field
     are left out of the agreement.
  2. modeling cohort (when its labels are available): agreement of the concept grade with the
     training label (labels_train.csv + labels_test.csv, or rebuilt read-only from the extract).
  3. --compare: agreement of the concept grade between two studies for the patients they share.

Inputs (run after scripts/ontology_report.py):
    data/metadata/ontology_report/<Type>__<portal>__<study>_concept_annotations.tsv
    data/fetch/cbioportal_api_request/<portal>/<type>/<study>/clinical_samples.csv
    labels (optional): data/gdm/filtered/<portal>/<type>/<study>/labels_*.csv, --labels-dir, or rebuilt
      read-only with processor.load_and_prepare (same label map, omics alignment and split as
      data_process). If none can be built, levels 1 and 3 are still reported.

Outputs (data/metadata/label_concordance/<Type>__<portal>__<study>/):
    flow.tsv, source_field.tsv (level 1), crosstab.tsv (level 2), shared.tsv (level 3),
    patients.tsv (one row per case) and summary.md (all levels, with agreement and Cohen's kappa)

    python scripts/label_concordance.py Glioma --study pedcbio:pbta_all --compare cbio:brain_cptac_2020
    python scripts/label_concordance.py Glioma --study pedcbio:pbta_all --labels none
    python scripts/label_concordance.py Glioma --study pedcbio:pbta_all --labels-dir path/to/modeled/run/pbta_all

--labels auto (default): labels_train/test.csv if present, else rebuilt from the extract, else skipped.
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


LEFT_OUT: Dict[str, int] = {}     # label values outside the map, left out when rebuilding from the extract


def labels_from_files(mdir: Path) -> pd.DataFrame:
    y = []
    for split in ("train", "test"):
        s = pd.read_csv(mdir / f"labels_{split}.csv", index_col=0).iloc[:, 0]
        y.append(pd.DataFrame({"label": s.astype(int).values, "split": split}, index=s.index.astype(str)))
    return pd.concat(y)


def labels_from_extract(cfg: dict, ct: str, portal: str, study: str, lc: dict) -> pd.DataFrame:
    """Modeling-cohort labels exactly as data_process builds them, without feature selection or writing."""
    from src.processor import LABELS_LEFT_OUT, MODALITY_SYNONYMS, OMICS_MODALITIES, load_and_prepare
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
                               select_features=False, drop_unmapped=True)
    except ValueError as e:
        raise SystemExit(f"could not build the labels as data_process would: {e}\n"
                         f"Fix processor.labels.{ct} in the config (README, 'Labels for process'), "
                         "or pass --labels-dir with the labels of the run you modeled.")
    y_tr, y_te = out[2], out[3]
    LEFT_OUT.update(LABELS_LEFT_OUT)
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


def source_field_grades(onto, sfile: Path, column: str) -> pd.Series:
    """Per patient: the grade implied by `column` (resolved through the ontology), 'conflicting' if
    its samples imply different grades, or NOT_DETERMINED."""
    smp = pd.read_csv(sfile, dtype=str)
    pid = next((c for c in ("patientId", "PATIENT_ID", "subject_id") if c in smp.columns), None)
    if pid is None or column not in smp.columns:
        return pd.Series(dtype=str)
    cache: Dict[str, str] = {}

    def grade(v: str) -> str:
        if v not in cache:
            r = onto.resolve("TEXT", v)
            g = onto.related(r.concept_id, "has_grade") if r.ok else None
            cache[v] = (onto.gdc_value(g[0], "diagnosis", "tumor_grade") if g else None) or NOT_DETERMINED
        return cache[v]

    def per_patient(vals: pd.Series) -> str:
        gs = {grade(v) for v in vals.dropna().astype(str).str.strip() if v} - {NOT_DETERMINED}
        return gs.pop() if len(gs) == 1 else ("conflicting" if gs else NOT_DETERMINED)

    out = smp.groupby(pid)[column].agg(per_patient)
    out.index = out.index.astype(str)
    return out


def md_table(xt: pd.DataFrame, corner: str) -> list:
    return ([f"| {corner} | " + " | ".join(map(str, xt.columns)) + " |", "| --- |" + " ---: |" * len(xt.columns)]
            + [f"| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in xt.iterrows()])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cancer_type")
    ap.add_argument("--study", required=True, help="portal:study_id, e.g. pedcbio:pbta_all")
    ap.add_argument("--compare", help="portal:study_id of a second study sharing patients")
    ap.add_argument("--group", help="cohort in analysis_groups.tsv (default: search.cohorts in the config)")
    ap.add_argument("--source-column", help="coarser source field to compare with the concept grade "
                                            "(default: processor.labels.<type>.column, else CANCER_TYPE)")
    ap.add_argument("--grade-of-class", help='e.g. "0=Low Grade,1=High Grade" (default: from the label map)')
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--report-dir", default="data/metadata/ontology_report")
    ap.add_argument("--labels", choices=("auto", "files", "extract", "none"), default="auto",
                    help="where the modeling-cohort labels come from (see above)")
    ap.add_argument("--labels-dir", help="folder with labels_train.csv / labels_test.csv "
                                         "(default data/gdm/filtered/<portal>/<type>/<study>)")
    ap.add_argument("--out", default="data/metadata/label_concordance")
    args = ap.parse_args()

    from src.ontology import OntologyLayer
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
    study_dir = Path("data/fetch/cbioportal_api_request") / portal / slug(ct) / study
    out = Path(args.out) / f"{ct.replace(' ', '_')}__{portal}__{study}"
    out.mkdir(parents=True, exist_ok=True)

    a = load_annotations(Path(args.report_dir), ct, portal, study)
    a["status"] = concept_status(a, cohort)
    a["concept_grade"] = a.gdc_tumor_grade.where(a.gdc_tumor_grade.isin(GRADES), NOT_DETERMINED)
    in_set = a.status == "in cohort"
    flow = [
        ("cases in the clinical extract", len(a)),
        ("  resolved to an NCIt concept", int(a.disease_concept_id.notna().sum())),
        ("    in the cohort concept set", int(in_set.sum())),
        ("      tumor_grade Low Grade (from concept)", int((in_set & (a.concept_grade == "Low Grade")).sum())),
        ("      tumor_grade High Grade (from concept)", int((in_set & (a.concept_grade == "High Grade")).sum())),
        (f"      {NOT_DETERMINED}", int((in_set & (a.concept_grade == NOT_DETERMINED)).sum())),
        ("    outside the cohort concept set", int((a.status == "outside cohort").sum())),
        ("  unresolved (review queue)", int((a.status == "unresolved").sum())),
    ]
    md = [f"# {ct}: grade from the data model to the modeling cohort ({portal}:{study})", "",
          f"Cohort concept set: `{group}` ({len(cohort)} NCIt concepts).", ""]

    # ---- level 1: concept grade vs the grade implied by a coarser source field
    scol = args.source_column or lc.get("column") or "CANCER_TYPE"
    sfile = study_dir / "clinical_samples.csv"
    a["source_field_grade"] = NOT_DETERMINED
    if sfile.exists():
        onto = OntologyLayer(ROOT / "config/ontology", ROOT / "src/gdcdictionary/schemas")
        sg = source_field_grades(onto, sfile, scol)
        a["source_field_grade"] = sg.reindex(a.index).fillna(NOT_DETERMINED)
        own = a.disease_source.fillna("").str.endswith(f".{scol}")       # concept came from that field
        cmp1 = a[~own & a.concept_grade.isin(GRADES) & a.source_field_grade.isin(GRADES)]
        ag1, n1, k1 = cohen_kappa(cmp1.concept_grade, cmp1.source_field_grade)
        xt1 = pd.crosstab(a.loc[~own, "concept_grade"], a.loc[~own, "source_field_grade"])
        xt1.to_csv(out / "source_field.tsv", sep="\t")
        flow += [(f"{scol}: grade implied for both (concept from another field)", n1),
                 (f"{scol}: grade agrees", ag1),
                 (f"{scol}: concept taken from {scol} itself (not compared)", int(own.sum()))]
        md += [f"## Concept grade vs. the grade implied by {scol}", "",
               f"The concept comes from the most specific field that resolves (e.g. CANCER_TYPE_DETAILED); "
               f"{scol} is resolved through the same mappings. Where both imply a grade, they agree in "
               f"{ag1} of {n1} cases ({(ag1 / n1 if n1 else float('nan')):.1%}; Cohen's kappa {k1:.3f}). "
               f"{int(own.sum())} cases whose concept came from {scol} itself are not compared.", ""]
        md += md_table(xt1, f"concept grade \\ {scol}") + [""]
    else:
        md += [f"({sfile} not found: source-field comparison skipped)", ""]

    # ---- level 2: modeling cohort (optional)
    y, label_source = None, None
    if args.labels != "none":
        mdir = Path(args.labels_dir) if args.labels_dir else Path("data/gdm/filtered") / portal / slug(ct) / study
        have_files = all((mdir / f"labels_{sp}.csv").exists() for sp in ("train", "test"))
        if args.labels == "files" and not have_files:
            raise SystemExit(f"{mdir}/labels_train.csv or labels_test.csv not found")
        if have_files and args.labels != "extract":
            y, label_source = labels_from_files(mdir), f"{mdir}/labels_train.csv + labels_test.csv"
        else:
            print("[labels] rebuilding the modeling-cohort labels from the extract (read-only), as data_process would")
            try:
                y = labels_from_extract(cfg, ct, portal, study, lc)
                label_source = ("rebuilt from the current extract with processor.load_and_prepare "
                                "(same label map, omics alignment and split as data_process)")
                if LEFT_OUT:
                    label_source += ("; samples with a label value outside the map were left out: "
                                     + ", ".join(f"{k} ({v})" for k, v in LEFT_OUT.items()))
            except SystemExit as e:
                if args.labels == "extract":
                    raise
                print(f"[labels] {e}\n[labels] modeling-cohort comparison skipped; data-model checks still reported")
                md += ["## Modeling cohort", "", "Skipped: the labels could not be built from the current extract "
                       "and config. Pass --labels-dir with labels_train.csv / labels_test.csv of the modeled run.", ""]

    pts = a[["disease_concept_id", "disease_concept_label", "disease_predicate", "disease_source",
             "disease_source_value", "status", "concept_grade", "source_field_grade"]]
    if y is not None:
        g_of = grade_of_class(lc.get("map"), args.grade_of_class)
        y["label_grade"] = y.label.map(g_of)
        pts = pts.join(y, how="outer")
        pts["in_modeling_cohort"] = pts.label.notna()
        pts.loc[pts.status.isna(), "status"] = "not in the clinical extract"
        m = pts[pts.in_modeling_cohort]
        det = m[m.concept_grade.isin(GRADES)]
        agree, n_det, kappa = cohen_kappa(det.concept_grade, det.label_grade)
        flow.append(("modeling cohort (labels_train + labels_test)", len(m)))
        for cls in sorted(g_of):
            flow.append((f"  label {cls} ({g_of[cls]})", int((m.label == cls).sum())))
        for sp in ("train", "test"):
            flow.append((f"  {sp}", int((m.split == sp).sum())))
        for st in ("outside cohort", "unresolved", "not in the clinical extract"):
            flow.append((f"  modeling cohort: {st}", int((m.status == st).sum())))
        flow += [("  concept grade determined", n_det), ("  concept grade agrees with label", agree)]
        xt = pd.crosstab(m.concept_grade.where(m.status == "in cohort", m.status), m.label_grade,
                         margins=True, margins_name="total")
        xt.to_csv(out / "crosstab.tsv", sep="\t")
        md += ["## Concept grade vs. training label (modeling cohort)", "",
               f"Labels: {label_source}. Map `{lc.get('map')}` "
               f"({', '.join(f'{k} = {v}' for k, v in sorted(g_of.items()))}).", ""]
        md += md_table(xt, "concept grade \\ label")
        md += ["", f"Agreement where the concept implies a grade: {agree} of {n_det} "
                   f"({(agree / n_det if n_det else float('nan')):.1%}), Cohen's kappa {kappa:.3f}.", ""]
    pts.index.name = "patient_id"
    pts.to_csv(out / "patients.tsv", sep="\t")

    # ---- level 3: patients in two studies
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
        flow += [(f"shared with {s2}", len(shared)), (f"  grade in both studies", n2), ("  grade agrees", ag2)]
        md += [f"## Patients in both {study} and {s2}", "",
               f"{len(shared)} shared patients; the concept grade agrees in {ag2} of {n2} with a grade "
               f"in both studies (Cohen's kappa {k2:.3f}).", ""]
        md += md_table(pd.crosstab(sh[f"{study}_grade"], sh[f"{s2}_grade"]), f"{study} \\ {s2}") + [""]

    flow = pd.DataFrame(flow, columns=["step", "n"])
    flow.to_csv(out / "flow.tsv", sep="\t", index=False)
    md[4:4] = ["## Flow", "", "| Step | n |", "| --- | ---: |"] + \
              [f"| {s.replace('  ', '&nbsp;&nbsp;')} | {n} |" for s, n in flow.itertuples(index=False)] + [""]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
