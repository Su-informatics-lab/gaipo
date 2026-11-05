"""scripts/label_concordance.py on a small glioma cohort annotated by the real ontology layer."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.ontology import OntologyLayer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("label_concordance", ROOT / "scripts/label_concordance.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

# patient, source label (CANCER_TYPE_DETAILED), label class read by data_process (None = not modeled)
PBTA = [("P1", "Low-grade glioma, KIAA1549-BRAF", 0), ("P2", "Pilocytic astrocytoma", 0),
        ("P3", "Diffuse midline glioma, H3 K28-altered", 1), ("P4", "High-grade glioma, IDH-mutant", 1),
        ("P5", "Low-grade glioma, BRAF V600E", 1),          # disagrees with its concept
        ("P6", "Oligodendroglioma", 0),                      # concept implies no grade
        ("P7", "Ependymoma", None), ("P8", "Not Reported", None)]
CPTAC = [("P1", "Pediatric Low Grade Gliomas"), ("P3", "Pediatric High Grade Gliomas"),
         ("P4", "Pediatric Low Grade Gliomas")]


def _annotate(rows):
    p = pd.DataFrame({"patientId": [r[0] for r in rows]})
    s = pd.DataFrame({"sampleId": [f"S{r[0]}" for r in rows], "patientId": [r[0] for r in rows],
                      "CANCER_TYPE_DETAILED": [r[1] for r in rows]})
    return OntologyLayer(ROOT / "config/ontology", ROOT / "src/gdcdictionary/schemas").annotate_cases(p, s)


def test_flow_crosstab_and_shared(tmp_path, monkeypatch):
    rep = tmp_path / "data/metadata/ontology_report"
    rep.mkdir(parents=True)
    _annotate(PBTA).to_csv(rep / "Glioma__pedcbio__pbta_all_concept_annotations.tsv", sep="\t", index=False)
    _annotate(CPTAC).to_csv(rep / "Glioma__cbio__brain_cptac_2020_concept_annotations.tsv", sep="\t", index=False)
    mdir = tmp_path / "data/gdm/filtered/pedcbio/glioma/pbta_all"
    mdir.mkdir(parents=True)
    y = pd.Series({r[0]: r[2] for r in PBTA if r[2] is not None}, name="label")
    y.iloc[:4].to_csv(mdir / "labels_train.csv", header=True)
    y.iloc[4:].to_csv(mdir / "labels_test.csv", header=True)
    cfg = {"processor": {"labels": {"Glioma": {"source": "samples", "column": "CANCER_TYPE",
                                               "map": "Low-grade glioma=0,High-grade glioma=1"}}},
           "search": {"cohorts": {"Glioma": "glioma"}}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["label_concordance.py", "Glioma", "--study", "pedcbio:pbta_all",
                                      "--compare", "cbio:brain_cptac_2020", "--config", "cfg.yaml"])
    lc.main()

    out = tmp_path / "data/metadata/label_concordance/Glioma__pedcbio__pbta_all"
    flow = pd.read_csv(out / "flow.tsv", sep="\t").assign(step=lambda d: d.step.str.strip()).set_index("step").n
    assert flow["cases in the clinical extract"] == 8
    assert flow["unresolved (review queue)"] == 1 and flow["outside the cohort concept set"] == 1
    assert flow["tumor_grade Low Grade (from concept)"] == 3
    assert flow["tumor_grade High Grade (from concept)"] == 2
    assert flow["modeling cohort (labels_train + labels_test)"] == 6
    assert flow["concept grade determined"] == 5 and flow["concept grade agrees with label"] == 4
    xt = pd.read_csv(out / "crosstab.tsv", sep="\t", index_col=0)
    assert xt.loc["Low Grade", "High Grade"] == 1 and xt.loc[lc.NOT_DETERMINED, "Low Grade"] == 1
    sh = pd.read_csv(out / "shared.tsv", sep="\t", index_col=0)
    assert list(sh.index) == ["P1", "P3", "P4"]
    assert (sh.pbta_all_grade == sh.brain_cptac_2020_grade).sum() == 2


def test_grade_of_class_from_map_and_override():
    assert lc.grade_of_class("low-grade=0,high-grade=1", None) == {0: "Low Grade", 1: "High Grade"}
    assert lc.grade_of_class(None, "0=High Grade,1=Low Grade") == {0: "High Grade", 1: "Low Grade"}
    agree, n, kappa = lc.cohen_kappa(pd.Series(["a", "a", "b", "b"]), pd.Series(["a", "a", "b", "a"]))
    assert (agree, n) == (3, 4) and round(kappa, 3) == 0.5
