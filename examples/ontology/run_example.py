"""
Worked example: Wilms tumor diagnosis and primary site through the ontology layer.

    python -m examples.ontology.run_example

Input is a small SYNTHETIC cBioPortal-shaped extract (examples/ontology/*.csv) with the
spelling and coding variants seen across sources. Output: one row per subject showing the
source value, the NCIt concept it resolved to, the cross-references, and the GDC values
derived from the concept - next to what the legacy string maps produce.
"""
from pathlib import Path

import pandas as pd
import yaml

from src.data_model import _derive_disease_type, _derive_primary_site
from src.ontology import OntologyLayer

HERE = Path(__file__).parent


def main() -> None:
    patients = pd.read_csv(HERE / "clinical_patients.csv")
    samples = pd.read_csv(HERE / "clinical_samples.csv")

    onto = OntologyLayer("config/ontology", "src/gdcdictionary/schemas")
    ann = onto.annotate_cases(patients, samples)

    cfg = yaml.safe_load(open("config/gdc_mapping.yaml"))
    ann["legacy_disease_type"] = ann.subject_id.map(_derive_disease_type(patients, samples, cfg))
    ann["legacy_primary_site"] = ann.subject_id.map(_derive_primary_site(patients, samples, cfg))

    pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_colwidth", 40)
    cols = ["subject_id", "disease_source", "disease_source_value", "disease_concept_id",
            "icdo3_morphology", "snomedct_code", "icd10cm_code", "site_concept_id", "site_predicate",
            "omop_icdo3_code", "gdc_disease_type", "legacy_disease_type",
            "gdc_primary_site", "legacy_primary_site"]
    print(ann[cols].to_string(index=False))
    print("\nUnresolved source values (for curator review):")
    print(onto.unmapped_report(patients, samples).to_string(index=False))

    out = HERE / "output"
    out.mkdir(exist_ok=True)
    ann.to_csv(out / "concept_annotations.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
