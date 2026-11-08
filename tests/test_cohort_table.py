"""scripts/cohort_table.py (Table 2) and the class names in src/cohort.py."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.cohort import class_names, load_labels

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cohort_table", ROOT / "scripts/cohort_table.py")
ct_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct_mod)


def test_class_names():
    lab = lambda m, **kw: {"processor": {"labels": {"X": {"map": m, **kw}}}}
    assert class_names(lab("FHWT=0,DAWT=1"), "X") == {0: "FHWT", 1: "DAWT"}
    grade = {"Low-grade glioma/astrocytoma (WHO grade I/II)": 0, "Low-Grade Glioma, NOS or NEC": 0,
             "High-grade glioma/astrocytoma (WHO grade III/IV)": 1}
    assert class_names(lab(grade), "X") == {0: "Low-grade", 1: "High-grade"}
    assert class_names(lab({"A": 0, "B": 0, "C": 1}), "X") == {0: "label 0", 1: "C"}
    assert class_names(lab(grade, class_names={0: "LGG", 1: "HGG"}), "X") == {0: "LGG", 1: "HGG"}
    assert class_names(lab(grade, class_names="1=HGG,0=LGG"), "X") == {0: "LGG", 1: "HGG"}


def _write(tmp, portal, ct, study, pat, sam, labels):
    raw = tmp / "data/fetch/cbioportal_api_request" / portal / ct / study
    raw.mkdir(parents=True)
    pat.to_csv(raw / "clinical_patients.csv", index=False)
    sam.to_csv(raw / "clinical_samples.csv", index=False)
    f = tmp / "data/gdm/filtered" / portal / ct / study
    f.mkdir(parents=True)
    y = pd.Series(labels, name="label")
    y.index.name = "patientId"
    y.iloc[1:].to_csv(f / "labels_train.csv", header=True)
    y.iloc[:1].to_csv(f / "labels_test.csv", header=True)


def test_table_for_target_and_pbta_layouts(tmp_path, monkeypatch, capsys):
    # TARGET-like: everything at patient level, blank EVENT_TYPE = no event, stage groups
    wt_pat = pd.DataFrame({
        "patientId": ["W1", "W2", "W3", "W4"], "AGE": [3, 5, 7, 13], "SEX": ["Female", "Male", "Female", "Female"],
        "ETHNICITY": ["Not reported", "Hispanic or Latino", "Not Hispanic or Latino", "Not Hispanic or Latino"],
        "RACE": ["White", "Asian", "Black or African American", "Not reported"],
        "OS_STATUS": ["0:LIVING", "1:DECEASED", "0:LIVING", "0:LIVING"], "OS_MONTHS": [10, 20, 30, 40],
        "EVENT_TYPE": [np.nan, "Relapse", "Progression", np.nan], "DAYS_TO_EVENT": [3043.75, 304.375, 608.75, 913.125],
        "CLINICAL_STAGE": ["IIIB/V", "IV/V", "I", "II"]})
    wt_sam = pd.DataFrame({"sampleId": [f"{p}-01" for p in wt_pat.patientId], "patientId": wt_pat.patientId,
                           "HISTOLOGY_CLASSIFICATION_IN_PRIMARY_TUMOR": ["DAWT", "DAWT", "FHWT", "FHWT"]})
    _write(tmp_path, "cbio", "wilms_tumor", "wt_s", wt_pat, wt_sam, {"W1": 1, "W2": 1, "W3": 0, "W4": 0})
    # PBTA-like: age at sample level in days, EFS_STATUS words, sample IDs without the patient ID
    g_pat = pd.DataFrame({"patientId": ["PT_A", "PT_B", "PT_C"], "SEX": ["Male", "Female", "Not Reported"],
                          "ETHNICITY": ["Not Hispanic or Latino"] * 3, "RACE": ["White", "Other", "Asian"],
                          "OS_STATUS": ["DECEASED", "LIVING", "LIVING"], "OS_MONTHS": [6.0, 50.0, 70.0],
                          "EFS_STATUS": ["Progression", "Not Applicable", "Not Reported"],
                          "EFS_MONTHS": [3.0, 50.0, 70.0]})
    g_sam = pd.DataFrame({"sampleId": ["7316-1", "7316-2", "7316-3", "7316-4"],
                          "patientId": ["PT_A", "PT_A", "PT_B", "PT_C"],
                          "AGE_AT_DIAGNOSIS_DAYS": [3652.5, 4383.0, 1826.25, 5113.5]})
    _write(tmp_path, "pedcbio", "glioma", "pbta_s", g_pat, g_sam, {"PT_A": 1, "PT_B": 0, "PT_C": 0})
    cfg = {"cancer_type_interest": ["Wilms Tumor", "Glioma"],
           "cBio_studyId": {"Wilms Tumor": "wt_s"}, "pedcBio_studyId": {"Glioma": "pbta_s"},
           "processor": {"labels": {"Wilms Tumor": {"map": "FHWT=0,DAWT=1"},
                                    "Glioma": {"map": {"Low-grade glioma/astrocytoma (WHO grade I/II)": 0,
                                                       "High-grade glioma/astrocytoma (WHO grade III/IV)": 1}}}}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["cohort_table.py", "--config", "cfg.yaml"])
    ct_mod.main()

    out = tmp_path / "data/analysis/cohort_table"
    t = pd.read_csv(out / "cohort_table.tsv", sep="\t", header=None, keep_default_na=False)
    assert list(t.iloc[0]) == ["", "Wilms Tumor", "", "", "Glioma", "", ""]
    assert list(t.iloc[1]) == ["Category", "DAWT (n = 2)", "FHWT (n = 2)", "All (n = 4)",
                               "High-grade (n = 1)", "Low-grade (n = 2)", "All (n = 3)"]
    rows, section = {}, None
    for r in t.iloc[2:].itertuples(index=False):
        label, cells = r[0], list(r[1:])
        section = label if not label.startswith("  ") else section
        rows[(section.split(",")[0], label.strip())] = cells
    age = "Age at diagnosis"
    assert rows[(age, age + ", years, No. (%); Median (IQR)")][:3] == ["2 (100); 4 (3.5-4.5)", "2 (100); 10 (8.5-11.5)",
                                                   "4 (100); 6 (4.5-8.5)"]
    assert rows[(age, "0-5")][3:] == ["0 (0.0)", "1 (50.0); 5 (5-5)", "1 (33.3); 5 (5-5)"]
    assert rows[(age, "6-11")][3] == "1 (100); 10 (10-10)"              # PT_A: lowest age over its samples
    assert rows[(age, "12+")][3:] == ["0 (0.0)", "1 (50.0); 14 (14-14)", "1 (33.3); 14 (14-14)"]
    assert rows[("Race", "Other")] == ["1 (50.0)", "0 (0.0)", "1 (25.0)", "0 (0.0)", "2 (100)", "2 (66.7)"]
    assert rows[("Ethnicity", "Not Available")][:3] == ["1 (50.0)", "0 (0.0)", "1 (25.0)"]
    assert rows[("Sex", "Female")][3:] == ["0 (0.0)", "1 (50.0)", "1 (33.3)"]
    assert rows[("Sex", "Not Available")][3:] == ["0 (0.0)", "1 (50.0)", "1 (33.3)"]   # shown: Glioma has one
    assert ("Overall Survival (OS) Status", "Not Available") not in rows                 # none anywhere
    efs = "Event-Free Survival (EFS) Status"
    assert rows[(efs, "Event")] == ["1 (50.0)", "1 (50.0)", "2 (50.0)", "1 (100)", "0 (0.0)", "1 (33.3)"]
    assert rows[(efs, "No Event")][:3] == ["1 (50.0)", "1 (50.0)", "2 (50.0)"]
    assert rows[(efs, "Not Available")][3:] == ["0 (0.0)", "1 (50.0)", "1 (33.3)"]
    eft = "Event-Free Follow-up time"
    assert rows[(eft, eft + ", months, Median (IQR)")][:2] == ["55.0 (32.5-77.5)", "25.0 (22.5-27.5)"]
    assert rows[("Clinical stage", "III")] == ["1 (50.0)", "0 (0.0)", "1 (25.0)", "-", "-", "-"]
    assert rows[("Clinical stage", "IV")][:3] == ["1 (50.0)", "0 (0.0)", "1 (25.0)"]

    vals = pd.read_csv(out / "cohort_values.tsv", sep="\t")
    race = vals[(vals.cancer_type == "Glioma") & (vals.variable == "race")].set_index("source_value")
    assert race.loc["Asian", "counted_as"] == "Other" and race.loc["Other", "All"] == 1
    efs = vals[(vals.cancer_type == "Glioma") & (vals.variable == "efs_status")].set_index("source_value")
    assert efs.counted_as.to_dict() == {"Not Applicable": "No Event", "Not Reported": "Not Available",
                                        "Progression": "Event"}
    md = (out / "cohort_table.md").read_text()
    assert "AGE_AT_DIAGNOSIS_DAYS (samples, days converted to years, lowest over the patient's samples)" in md
    assert "Race counted as Other: Asian (1), Other (1)." in md
    assert "Not in the extract (shown as -): stage." in md
    pats = pd.read_csv(out / "cohort_patients.tsv", sep="\t")
    assert len(pats) == 7 and set(pats["class"]) == {"DAWT", "FHWT", "High-grade", "Low-grade"}


def test_column_override_and_unit(tmp_path):
    ov = ct_mod._overrides(["Glioma:age=AGE_DX:days", "os_time=OS_D:days", "efs_status=EVENT_TYPE"])
    assert ov == {("Glioma", "age"): ("AGE_DX", "days"), (None, "os_time"): ("OS_D", "days"),
                  (None, "efs_status"): ("EVENT_TYPE", None)}
    assert ct_mod.stage_category("IIIB/V") == "III" and ct_mod.stage_category("V") == "V"
    assert ct_mod.stage_category("U") == "Not Available"
    assert load_labels.__doc__
