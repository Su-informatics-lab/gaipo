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


def test_broad_match_does_not_assert_morphology(onto):
    p = pd.DataFrame({"patientId": ["G1", "G3"], "ICD_O_3_SITE": [None, "9440/3"]})
    s = pd.DataFrame({"sampleId": ["G1-01", "G3-01"], "ONCOTREE_CODE": ["LGG", "GBM"],
                      "ICD_0_3_T": ["C71.1", "C71.9"]})
    a = onto.annotate_cases(p, s).set_index("subject_id")
    assert a.loc["G1", "gdc_disease_type"] == "Gliomas"          # block-level is still right
    assert pd.isna(a.loc["G1", "gdc_morphology"])                 # 9380/3 not asserted for LGG
    assert pd.isna(a.loc["G1", "omop_icdo3_code"])
    assert a.loc["G3", "omop_icdo3_code"] == "9440/3-C71.9"


def test_pbta_style_ids_link_through_patient_id(onto):
    p = pd.DataFrame({"patientId": ["PT_A", "PT_B"]})
    s = pd.DataFrame({"sampleId": ["BS_1", "BS_2"], "patientId": ["PT_A", "PT_B"],
                      "CANCER_TYPE_DETAILED": ["Pilocytic astrocytoma", "Low-grade glioma, KIAA1549-BRAF"],
                      "ONCOTREE_CODE": ["lggnos", "lggnos"]})
    a = onto.annotate_cases(p, s).set_index("subject_id")
    assert a.loc["PT_A", "disease_concept_id"] == "NCIT:C4047"       # detailed text wins over OncoTree
    assert a.loc["PT_B", "disease_predicate"] == "skos:broadMatch"
    assert (a.gdc_disease_type == "Gliomas").all()
