import pandas as pd
import pytest

from src.ontology import (OntologyLayer, normalize_label, parse_icdo3_morphology,
                          parse_icdo3_topography, snomed_to_icd10cm)


@pytest.fixture(scope="module")
def onto():
    return OntologyLayer("config/ontology", "src/gdcdictionary/schemas")


def test_parsers():
    assert parse_icdo3_morphology("8960/3 : Nephroblastoma, NOS") == "8960/3"
    assert parse_icdo3_topography("C649") == "C64.9"
    assert parse_icdo3_topography("c64") == "C64.9"
    assert normalize_label("Wilms' Tumor") == normalize_label("wilms tumor")


@pytest.mark.parametrize("system,raw", [
    ("TEXT", "Wilms Tumor"), ("TEXT", "Wilms' Tumor"), ("TEXT", "Wilms tumour"),
    ("TEXT", "Nephroblastoma"), ("TEXT", "8960/3 : Nephroblastoma, NOS"),
    ("ICDO3M", "8960/3"), ("ONCOTREE", "WT"),
])
def test_wilms_variants_resolve_to_one_concept(onto, system, raw):
    assert onto.resolve(system, raw).concept_id == "NCIT:C3267"


def test_unknown_value_is_not_guessed(onto):
    assert not onto.resolve("TEXT", "Clear Cell Sarcoma of Kidney").ok


def test_gdc_values_come_from_dictionary_termdefs(onto):
    assert onto.gdc_value("NCIT:C12415", "case", "primary_site") == "Kidney"
    assert onto.gdc_value("NCIT:C3267", "diagnosis", "primary_diagnosis") == "Wilms tumor"
    assert onto.disease_type_for_morphology("8960/3") == "Complex Mixed and Stromal Neoplasms"
    assert onto.disease_type_for_morphology("9380/3") == "Gliomas"


def test_snomed_to_icd10cm_rules():
    # Rows follow the NLM technical specification example (237145004 Unexplained infertility)
    rows = pd.DataFrame([
        dict(referencedComponentId="237145004", mapGroup=1, mapPriority=1,
             mapRule="IFA 248152002 | Female (finding) |", mapAdvice="", mapTarget="N97.9"),
        dict(referencedComponentId="237145004", mapGroup=1, mapPriority=2,
             mapRule="IFA 248153007 | Male (finding) |", mapAdvice="", mapTarget="N46"),
        dict(referencedComponentId="237145004", mapGroup=1, mapPriority=3,
             mapRule="OTHERWISE TRUE", mapAdvice="MAP IS CONTEXT DEPENDENT FOR GENDER", mapTarget=""),
    ])
    assert snomed_to_icd10cm(rows, "237145004", {"248152002"})[0]["icd10cm"] == "N97.9"
    assert snomed_to_icd10cm(rows, "237145004", {"248153007"})[0]["icd10cm"] == "N46"
    assert snomed_to_icd10cm(rows, "237145004")[0]["icd10cm"] is None
