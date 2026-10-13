"""
ontology_report.py - Counts for the manuscript Results sentence.

For each cancer type, reads the clinical CSVs written by the data_extract stage,
runs the ontology layer, rebuilds the case node and validates it against the GDC
dictionary. Prints one summary row per cancer type and writes, per cancer type:

    data/metadata/ontology_report/<Cancer_Type>_concept_annotations.tsv
    data/metadata/ontology_report/<Cancer_Type>_unmapped_terms.tsv
    data/metadata/ontology_report/summary.tsv

Run from the repository root (after data_fetch + data_extract):

    python scripts/ontology_report.py                          # cancer types from config/gdc_mapping.yaml
    python scripts/ontology_report.py "Wilms Tumor" "Glioma"   # or name them

Read-only with respect to the pipeline: it does not overwrite data/gdm/.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cancer_types", nargs="*", help="default: cancer_type_interest in gdc_mapping.yaml")
    ap.add_argument("--fetch-root", default="data/fetch/cbioportal_api_request")
    ap.add_argument("--out", default="data/metadata/ontology_report")
    ap.add_argument("--strict", action="store_true", help="validate the case node in strict mode")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config/gdc_mapping.yaml"))
    cfg.setdefault("ontology", {}).update({"enabled": True})
    cancer_types = args.cancer_types or cfg.get("cancer_type_interest", [])
    schemas = cfg.get("schemas_dir", "src/gdcdictionary/schemas")
    onto = OntologyLayer(cfg["ontology"].get("dir", "config/ontology"), schemas)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with DataModel(schemas_dir=schemas, gdm_root=tempfile.mkdtemp(prefix="gaipo_onto_"),
                   strict=args.strict, cfg=cfg) as dm:
        for ct in cancer_types:
            d = Path(args.fetch_root) / ct.replace(" ", "_")
            pat, sam = d / "clinical_patients.csv", d / "clinical_samples.csv"
            if not (pat.exists() and sam.exists()):
                print(f"[skip] {ct}: no clinical CSVs in {d} (run data_fetch,data_extract first)")
                continue
            patients, samples = pd.read_csv(pat), pd.read_csv(sam)

            ann = onto.annotate_cases(patients, samples)
            unmapped = onto.unmapped_report(patients, samples)
            case = cbio_to_case(patients, samples, "GAIPO", strict=args.strict, cfg=cfg, ontology_ann=ann)
            rep = dm.validate_df_report("case", case, dm.schemas.required_fields_for("case"))

            # what the legacy string maps would have produced, for the "before" comparison
            legacy_cfg = dict(cfg, ontology={"enabled": False})
            legacy_dt = _derive_disease_type(patients, samples, legacy_cfg)
            allowed = set(onto.gdc_enum("case", "disease_type"))

            slug = ct.replace(" ", "_")
            ann.to_csv(out / f"{slug}_concept_annotations.tsv", sep="\t", index=False)
            unmapped.to_csv(out / f"{slug}_unmapped_terms.tsv", sep="\t", index=False)

            n_total = len(ann)
            rows.append({
                "cancer_type": ct,
                "cases_N": n_total,
                "disease_resolved_n": int(ann.disease_concept_id.notna().sum()),
                "disease_exactMatch_n": int((ann.disease_predicate == "skos:exactMatch").sum()),
                "disease_broadMatch_n": int((ann.disease_predicate == "skos:broadMatch").sum()),
                "site_from_topography_n": int(ann.icdo3_topography.notna().sum()),
                "site_inferred_n": int((ann.site_predicate == "inferred").sum()),
                "site_resolved_n": int(ann.site_concept_id.notna().sum()),
                "case_validation_errors": len(rep["errors"]),
                "unmapped_distinct_values": len(unmapped),
                "disease_concepts": "; ".join(f"{k} ({v})" for k, v in
                                              ann.disease_concept_label.value_counts().items()),
                "gdc_disease_type": "; ".join(f"{k} ({v})" for k, v in
                                              case.disease_type.value_counts().items()),
                "legacy_disease_type": "; ".join(f"{k} ({v})" for k, v in
                                                 legacy_dt.value_counts().items()),
                "legacy_invalid_disease_type_n": int((~legacy_dt.isin(allowed)).sum()),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print(summary.T.to_string(header=False) if len(summary) == 1 else summary.T.to_string())
    print(f"\nWrote {out}/summary.tsv and per-cancer annotation/unmapped files.")


if __name__ == "__main__":
    main()
