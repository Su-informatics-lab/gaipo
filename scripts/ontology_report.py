"""
ontology_report.py - Counts for the manuscript Results sentence.

For each cancer type and each source portal, reads the clinical CSVs written by the
data_extract stage, runs the ontology layer, rebuilds the case node and validates it
against the GDC dictionary.

Input layout (as written by src/extractor.py):

    data/fetch/cbioportal_api_request/cbio/<cancer_type>/<study_id>/clinical_patients.csv     # cBioPortal API
    data/fetch/cbioportal_api_request/pedcbio/<cancer_type>/<study_id>/clinical_patients.csv  # PedcBioPortal API
    (clinical_samples.csv alongside; <cancer_type> folder matched case-insensitively,
     e.g. wilms_tumor or Wilms_Tumor; the <study_id> level is optional)

Outputs (data/metadata/ontology_report/):

    summary.tsv                                         one row per cancer type x portal x study, plus "ALL"
    <cancer_type>__<portal>__<study>_concept_annotations.tsv
    <cancer_type>__<portal>__<study>_unmapped_terms.tsv

Run from the repository root (after data_fetch + data_extract):

    python scripts/ontology_report.py                              # cancer types from config/gdc_mapping.yaml
    python scripts/ontology_report.py "Wilms Tumor" "Glioma"
    python scripts/ontology_report.py "Glioma" --portal pedcbio    # one portal only

Read-only with respect to the pipeline: it does not write under data/gdm/.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_model import DataModel, _derive_disease_type, cbio_to_case  # noqa: E402
from src.ontology import OntologyLayer  # noqa: E402

PORTALS = ("cbio", "pedcbio")


def find_sources(fetch_root: Path, ct: str, portals) -> list:
    """[(portal, study_id, folder)] holding clinical_patients.csv + clinical_samples.csv."""
    wanted = ct.replace(" ", "_").lower()
    found = []
    for portal in portals:
        pdir = fetch_root / portal
        if not pdir.is_dir():
            continue
        for ct_dir in pdir.iterdir():
            if not (ct_dir.is_dir() and ct_dir.name.lower() == wanted):
                continue
            for pat in sorted(ct_dir.rglob("clinical_patients.csv")):
                if (pat.parent / "clinical_samples.csv").exists():
                    study = pat.parent.name if pat.parent != ct_dir else "-"
                    found.append((portal, study, pat.parent))
    return found


def load_groups(path: Path) -> dict:
    if not path.exists():
        return {}
    g = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    return {name: set(df.concept_id) for name, df in g.groupby("group")}


GROUPS: dict = {}


def summarize(onto, dm, cfg, args, patients, samples):
    ann = onto.annotate_cases(patients, samples)
    unmapped = onto.unmapped_report(patients, samples)
    case = cbio_to_case(patients, samples, "GAIPO", strict=args.strict, cfg=cfg, ontology_ann=ann)
    rep = dm.validate_df_report("case", case, dm.schemas.required_fields_for("case"))
    legacy_dt = _derive_disease_type(patients, samples, dict(cfg, ontology={"enabled": False}))
    allowed = set(onto.gdc_enum("case", "disease_type"))

    def counts(s):
        return "; ".join(f"{k} ({v})" for k, v in s.value_counts().items())

    row = {
        "cases_N": len(ann),
        "disease_resolved_n": int(ann.disease_concept_id.notna().sum()),
        "disease_exactMatch_n": int((ann.disease_predicate == "skos:exactMatch").sum()),
        "disease_broadMatch_n": int((ann.disease_predicate == "skos:broadMatch").sum()),
        "site_from_topography_n": int(ann.icdo3_topography.notna().sum()),
        "site_inferred_n": int((ann.site_predicate == "inferred").sum()),
        "site_resolved_n": int(ann.site_concept_id.notna().sum()),
        "case_validation_errors": len(rep["errors"]),
        "unmapped_distinct_values": len(unmapped),
        "disease_concepts": counts(ann.disease_concept_label),
        "gdc_disease_type": counts(case.disease_type),
        "legacy_disease_type": counts(legacy_dt),
        "legacy_invalid_disease_type_n": int((~legacy_dt.isin(allowed)).sum()),
    }
    for name, concepts in GROUPS.items():   # cohort counts from config/ontology/analysis_groups.tsv
        row[f"group_{name}_n"] = int(ann.disease_concept_id.isin(concepts).sum())
    return row, ann, unmapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cancer_types", nargs="*", help="default: cancer_type_interest in gdc_mapping.yaml")
    ap.add_argument("--fetch-root", default="data/fetch/cbioportal_api_request")
    ap.add_argument("--portal", choices=("all",) + PORTALS, default="all")
    ap.add_argument("--out", default="data/metadata/ontology_report")
    ap.add_argument("--strict", action="store_true", help="validate the case node in strict mode")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config/gdc_mapping.yaml"))
    cfg.setdefault("ontology", {}).update({"enabled": True})
    cancer_types = args.cancer_types or cfg.get("cancer_type_interest", [])
    portals = PORTALS if args.portal == "all" else (args.portal,)
    schemas = cfg.get("schemas_dir", "src/gdcdictionary/schemas")
    onto = OntologyLayer(cfg["ontology"].get("dir", "config/ontology"), schemas)
    GROUPS.update(load_groups(Path(cfg["ontology"].get("dir", "config/ontology")) / "analysis_groups.tsv"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with DataModel(schemas_dir=schemas, gdm_root=tempfile.mkdtemp(prefix="gaipo_onto_"),
                   strict=args.strict, cfg=cfg) as dm:
        for ct in cancer_types:
            sources = find_sources(Path(args.fetch_root), ct, portals)
            if not sources:
                print(f"[skip] {ct}: no clinical CSVs under {args.fetch_root}/{{{','.join(portals)}}}/"
                      f"{ct.replace(' ', '_').lower()}/ (run data_fetch,data_extract first)")
                continue
            all_p, all_s = [], []
            for portal, study, folder in sources:
                patients = pd.read_csv(folder / "clinical_patients.csv")
                samples = pd.read_csv(folder / "clinical_samples.csv")
                row, ann, unmapped = summarize(onto, dm, cfg, args, patients, samples)
                rows.append({"cancer_type": ct, "portal": portal, "study_id": study, **row})
                tag = f"{ct.replace(' ', '_')}__{portal}__{study}"
                ann.to_csv(out / f"{tag}_concept_annotations.tsv", sep="\t", index=False)
                unmapped.to_csv(out / f"{tag}_unmapped_terms.tsv", sep="\t", index=False)
                all_p.append(patients)
                all_s.append(samples)
            if len(sources) > 1:
                # combined, one record per patient / sample (the same patient can appear in both portals)
                pid = "patientId" if "patientId" in all_p[0].columns else "submitter_id"
                p = pd.concat(all_p, ignore_index=True).drop_duplicates(pid)
                s = pd.concat(all_s, ignore_index=True).drop_duplicates("sampleId")
                row, ann, unmapped = summarize(onto, dm, cfg, args, p, s)
                rows.append({"cancer_type": ct, "portal": "ALL", "study_id": "ALL (deduplicated)", **row})
                tag = f"{ct.replace(' ', '_')}__ALL"
                ann.to_csv(out / f"{tag}_concept_annotations.tsv", sep="\t", index=False)
                unmapped.to_csv(out / f"{tag}_unmapped_terms.tsv", sep="\t", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    if summary.empty:
        print("No data found.")
        return
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print(summary.set_index(["cancer_type", "portal", "study_id"]).T.to_string())
    print(f"\nWrote {out}/summary.tsv and per-source annotation/unmapped files.")


if __name__ == "__main__":
    main()
