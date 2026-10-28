"""
search_report.py - Ontology-derived CCDI search vs the fixed term list, without touching the cohorts.

Runs the data_fetch search (CCDI Data Federation /subject-diagnosis and /sample-diagnosis) with
the query terms of BOTH the ontology layer and the fixed lists in pipeline_config.yaml, then
selects records both ways on the same retrieved data. Writes only:

    data/metadata/search_report/search_terms.tsv    every ontology term, its origin and NCIt concept
    data/metadata/search_report/search_report.tsv   per cancer type and entity: n by ontology,
                                                    n by fixed list, overlap, ontology-only, fixed-only
    data/metadata/search_report/search_values.tsv   every diagnosis value either way selected, with the
                                                    ontology decision (concept / outside_cohort /
                                                    term_only) and the NCIt concept it resolved to

data/fetch/ccdi_api_request/{subject_ids,sample_ids}/ and data/gdm/indexes/idmap.parquet are not
written, so the cohorts used by the processor stay as they are.

Run from the repository root (needs network access to federation.ccdi.cancer.gov):

    python scripts/search_report.py                          # cancer_type_interest from the config
    python scripts/search_report.py "Wilms Tumor" "Glioma"
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_source as ds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cancer_types", nargs="*", help="default: cancer_type_interest in pipeline_config.yaml")
    ap.add_argument("--out", default="data/metadata/search_report")
    args = ap.parse_args()
    ds.data_fetch(write_cohorts=False, report_dir=Path(args.out), source="ontology",
                  compare_with_config=True, types=args.cancer_types or None)


if __name__ == "__main__":
    main()
