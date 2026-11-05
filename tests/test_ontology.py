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


# ---- grade-specific glioma concepts (most specific NCIt concept; coarser GDC values derived) ----

def test_ancestors_follow_is_a_nearest_first(onto):
    assert onto.ancestors("NCIT:C129309") == ["NCIT:C185368", "NCIT:C4822", "NCIT:C3059"]
    assert onto.ancestors("NCIT:C3267") == []


@pytest.mark.parametrize("label,concept,predicate,pdx,pdx_from,grade,morph", [
    # grade group only -> grade concept, exact; no GDC value of its own -> Glioma, NOS from the ancestor
    ("Low-grade glioma/astrocytoma (WHO grade I/II)", "NCIT:C132067", "skos:exactMatch",
     "Glioma, NOS", "NCIT:C3059", "Low Grade", None),
    # grade group + molecular qualifier -> grade concept, broad; qualifier stays in the source value
    ("Low-grade glioma, KIAA1549-BRAF", "NCIT:C132067", "skos:broadMatch",
     "Glioma, NOS", "NCIT:C3059", "Low Grade", None),
    ("High-grade glioma, IDH-wildtype and H3-wildtype", "NCIT:C4822", "skos:broadMatch",
     "Glioma, malignant", "NCIT:C4822", "High Grade", None),
    # WHO CNS5 entity (K28 numbering = K27): exact, so its ICD-O-3 code is asserted
    ("Diffuse midline glioma, H3 K28-altered", "NCIT:C185368", "skos:exactMatch",
     "Glioma, malignant", "NCIT:C4822", "High Grade", "9385/3"),
    # a grade 1 histology inherits Low Grade through is_a; keeps its own GDC value
    ("Pilocytic astrocytoma", "NCIT:C4047", "skos:exactMatch",
     "Pilocytic astrocytoma", "NCIT:C4047", "Low Grade", "9421/1"),
])
def test_glioma_labels_map_to_most_specific_concept(onto, label, concept, predicate, pdx, pdx_from,
                                                    grade, morph):
    p = pd.DataFrame({"patientId": ["P1"]})
    s = pd.DataFrame({"sampleId": ["S1"], "patientId": ["P1"], "CANCER_TYPE_DETAILED": [label]})
    a = onto.annotate_cases(p, s).iloc[0]
    assert (a.disease_concept_id, a.disease_predicate) == (concept, predicate)
    assert (a.gdc_primary_diagnosis, a.gdc_primary_diagnosis_concept) == (pdx, pdx_from)
    assert a.gdc_tumor_grade == grade
    assert a.gdc_disease_type == "Gliomas"                      # ICD-O-3 block, inherited if needed
    assert (a.icdo3_morphology if pd.notna(a.icdo3_morphology) else None) == morph
    assert a.site_concept_label == "Brain" and a.site_predicate == "inferred"


def test_concept_without_grade_group_gets_no_grade(onto):
    p = pd.DataFrame({"patientId": ["P1"]})
    s = pd.DataFrame({"sampleId": ["S1"], "patientId": ["P1"], "CANCER_TYPE_DETAILED": ["Oligodendroglioma"]})
    a = onto.annotate_cases(p, s).iloc[0]
    assert a.disease_concept_id == "NCIT:C3288" and pd.isna(a.gdc_tumor_grade)


def test_diagnosis_node_carries_concept_derived_grade(onto):
    pytest.importorskip("duckdb")
    from src.data_model import cbio_to_diagnosis
    p = pd.DataFrame({"patientId": ["P1", "P2"]})
    s = pd.DataFrame({"sampleId": ["S1", "S2"], "patientId": ["P1", "P2"],
                      "CANCER_TYPE_DETAILED": ["Diffuse midline glioma, H3 K28-altered", "Oligodendroglioma"]})
    d = cbio_to_diagnosis(p, ontology_ann=onto.annotate_cases(p, s)).set_index("case_submitter_id")
    assert d.loc["P1", ["primary_diagnosis", "morphology", "tumor_grade"]].tolist() == \
        ["Glioma, malignant", "9385/3", "High Grade"]
    assert d.loc["P2", "tumor_grade"] == "Not Reported"
    assert set(d.tumor_grade) <= set(onto.gdc_enum("diagnosis", "tumor_grade"))
