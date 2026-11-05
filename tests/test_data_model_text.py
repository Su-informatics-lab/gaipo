"""Demographic text normalizers in src/data_model.py with missing values (pandas 2 and 3)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
from src import data_model as dm  # noqa: E402


def test_missing_values_do_not_break_the_normalizers():
    # pandas 3 keeps NaN through astype(str); _norm_race used to raise
    # "TypeError: bad operand type for unary ~: 'float'" on a patient without RACE
    race = pd.Series(["White", np.nan, "Asian", "white;asian"], dtype=object)
    assert dm._norm_race(race).tolist() == ["white", "not reported", "asian", "white"]
    eth = pd.Series(["Not Hispanic or Latino", np.nan, "Hispanic or Latino"], dtype=object)
    assert dm._norm_ethnicity(eth).tolist() == ["not hispanic or latino", "not reported", "hispanic or latino"]
    sex = pd.Series(["Female", np.nan, "Male"], dtype=object)
    assert dm._clinical_gender_from_sex(sex).tolist() == ["female", "unspecified", "male"]
    assert dm._derive_sex_at_birth(pd.DataFrame({"SEX": sex})).tolist() == ["female", "unknown", "male"]


def test_demographic_values_are_gdc_permissible():
    import yaml
    enum = yaml.safe_load(open("src/gdcdictionary/schemas/demographic.yaml"))["properties"]
    p = pd.DataFrame({"patientId": ["P1", "P2", "P3", "P4"],
                      "SEX": ["Female", "Not Reported", np.nan, "Unknown"],
                      "RACE": ["White", "Not Reported", np.nan, "Other"],
                      "ETHNICITY": ["Not Hispanic or Latino", "Not reported", np.nan, "Unknown"],
                      "OS_STATUS": ["0:LIVING", "1:DECEASED", np.nan, "0:LIVING"]})
    d = dm.cbio_to_demographic(p)
    assert d.sex_at_birth.tolist() == ["female", "unknown", "unknown", "unknown"]   # "not reported" is not allowed
    for col in ("sex_at_birth", "vital_status", "ethnicity", "race"):
        assert set(d[col]) <= set(enum[col]["enum"]), col
