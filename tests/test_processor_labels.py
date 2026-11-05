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
