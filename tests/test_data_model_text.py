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
