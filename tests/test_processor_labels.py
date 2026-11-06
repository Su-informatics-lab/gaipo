"""Label building in src/processor.py: values outside the label map."""
import numpy as np
import pandas as pd
import pytest

from src.processor import LABELS_LEFT_OUT, _build_binary_labels_general

SAMPLES = pd.DataFrame({"sampleId": ["S1", "S2", "S3", "S4", "S5"],
                        "CANCER_TYPE": ["low-grade", "high-grade", np.nan, "Ganglioglioma", "low-grade"]})
MAPPING = pd.Series(["P1", "P2", "P3", "P4", "P5"], index=SAMPLES.sampleId)
KW = dict(label_source="samples", label_column="CANCER_TYPE", clin_pat=pd.DataFrame({"patientId": []}),
          clin_sam=SAMPLES, mapping_series=MAPPING, label_map_override={"low-grade": 0, "high-grade": 1})


def test_unmapped_values_stop_with_a_readable_message():
    # a missing value used to raise TypeError ("'<' not supported between 'float' and 'str'") on pandas 3
    with pytest.raises(ValueError, match=r"'<missing>' \(1\).*'Ganglioglioma' \(1\)|'Ganglioglioma' \(1\).*'<missing>' \(1\)"):
        _build_binary_labels_general(**KW)


def test_drop_unmapped_leaves_those_samples_out():
    y, inv = _build_binary_labels_general(**KW, drop_unmapped=True)
    assert y.to_dict() == {"P1": 0, "P2": 1, "P5": 0} and inv == {0: "low-grade", 1: "high-grade"}
    assert LABELS_LEFT_OUT == {"<missing>": 1, "Ganglioglioma": 1}


def test_no_value_in_the_map_is_an_error_even_when_dropping():
    with pytest.raises(ValueError, match="No CANCER_TYPE value matches"):
        _build_binary_labels_general(**{**KW, "label_map_override": {"Low-grade glioma": 0, "High-grade glioma": 1}},
                                     drop_unmapped=True)


def test_label_map_as_yaml_mapping_with_commas():
    from src.processor import _parse_label_map_str
    m = _parse_label_map_str({"Low-Grade Glioma, NOS or NEC": 0, "High-Grade Glioma, NOS or NEC": "1"})
    assert m == {"Low-Grade Glioma, NOS or NEC": 0, "High-Grade Glioma, NOS or NEC": 1}
    assert _parse_label_map_str("FHWT=0,DAWT=1") == {"FHWT": 0, "DAWT": 1}
    with pytest.raises(ValueError):
        _parse_label_map_str({"x": 2})


def test_omics_file_override_and_portal_sample_mapping(tmp_path):
    """omics_files picks one of several files of a modality; samples the CCDI list lacks are mapped
    to their patient through the portal's sample table."""
    from src.processor import _find_omics_file, load_and_prepare
    raw = tmp_path / "study"
    raw.mkdir()
    pts = [f"T-{i}" for i in range(10)]
    smp = [f"{p}-01" for p in pts]
    pd.DataFrame({"patientId": pts}).to_csv(raw / "clinical_patients.csv", index=False)
    pd.DataFrame({"sampleId": smp, "patientId": pts, "Y": ["A"] * 6 + ["B"] * 4}).to_csv(
        raw / "clinical_samples.csv", index=False)
    rng = np.random.default_rng(0)
    for name in ("s_rna_seq_mrna_median_all_sample_Zscores.tsv", "s_mrna.tsv", "s_mirna.tsv"):
        m = pd.DataFrame(rng.normal(size=(4, 10)), columns=smp)
        m.insert(0, "hugoGeneSymbol", [f"G{i}" for i in range(4)])
        m.to_csv(raw / name, sep="\t", index=False)
    assert _find_omics_file(raw, "mrna").name == "s_rna_seq_mrna_median_all_sample_Zscores.tsv"
    assert _find_omics_file(raw, "mrna", "s_mrna.tsv").name == "s_mrna.tsv"
    with pytest.raises(FileNotFoundError):
        _find_omics_file(raw, "mrna", "nothing_*.tsv")
    ccdi = tmp_path / "ccdi.csv"                       # CCDI lists only 6 of the 10 samples
    pd.DataFrame({"sample_id": smp[:6], "subject_id": pts[:6]}).to_csv(ccdi, index=False)
    out = load_and_prepare(rawData_dir=raw, ccdi_csv=ccdi, cancer_type="T", modalities=["mrna", "mirna"],
                           label_source="samples", label_column="Y", label_map="A=0,B=1",
                           select_features=False, omics_files={"mrna": "s_mrna.tsv"})
    y = pd.concat([out[2], out[3]])
    assert len(y) == 10                                # all 10 patients, not only the 6 CCDI lists
