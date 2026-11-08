"""post_analysis KM: group by the modeled class (group_col: label), per cancer type, on one split."""
import pandas as pd
import pytest
import yaml


def _study(tmp, portal, ct, study, n, label_of, group_vals=None):
    ids = [f"{ct[:1]}{i}" for i in range(n)]
    f = tmp / "data/gdm/filtered" / portal / ct / study
    f.mkdir(parents=True)
    clin = pd.DataFrame({"patientId": ids, "OS_MONTHS": [5.0 + 3 * i for i in range(n)],
                         "OS_STATUS": ["1:DECEASED" if i % 3 == 0 else "0:LIVING" for i in range(n)]})
    if group_vals:
        clin["HISTOLOGY_CLASSIFICATION_IN_PRIMARY_TUMOR"] = [group_vals[i % 2] for i in range(n)]
    clin.to_csv(f / "clinical_filtered.csv", index=False)
    y = pd.Series([label_of(i) for i in range(n)], index=pd.Index(ids, name="patientId"), name="label")
    y.iloc[: n // 4].to_csv(f / "labels_test.csv", header=True)
    y.iloc[n // 4:].to_csv(f / "labels_train.csv", header=True)
    return ids


def test_km_by_label_on_the_test_set(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("lifelines")
    from src import post_analysis
    _study(tmp_path, "pedcbio", "glioma", "pbta_all", 40, lambda i: i % 2)
    _study(tmp_path, "cbio", "wilms_tumor", "wt", 20, lambda i: i % 2, group_vals=["FHWT", "DAWT"])
    cfg = {"cancer_type_interest": ["Wilms Tumor", "Glioma"],
           "cBio_studyId": {"Wilms Tumor": "wt"}, "pedcBio_studyId": {"Glioma": "pbta_all"},
           "processor": {"labels": {"Glioma": {"map": {"Low-grade glioma/astrocytoma (WHO grade I/II)": 0,
                                                       "High-grade glioma/astrocytoma (WHO grade III/IV)": 1}}}},
           "post_analysis": {"survival": {
               "enabled": True,
               "group_col": {"Wilms Tumor": "HISTOLOGY_CLASSIFICATION_IN_PRIMARY_TUMOR", "Glioma": "label"},
               "subset": {"Glioma": "test"},
               "title": {"Wilms Tumor": "OS by histology", "Glioma": "OS by grade group"}}}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "cfg.yaml"))
    monkeypatch.chdir(tmp_path)
    post_analysis.run_km_survival()

    g = pd.read_csv(tmp_path / "data/analysis/km/pedcbio/glioma/pbta_all/km_stats.tsv", sep="\t")
    assert g.group.tolist() == ["Low-grade", "High-grade"] and g.n.tolist() == [5, 5]   # 10 test patients
    assert set(g.subset) == {"test"} and g.logrank_p.notna().all()
    w = pd.read_csv(tmp_path / "data/analysis/km/cbio/wilms_tumor/wt/km_stats.tsv", sep="\t")
    assert w.group.tolist() == ["DAWT", "FHWT"] and w.n.sum() == 20 and set(w.subset) == {"all"}
    assert (tmp_path / "data/analysis/km/pedcbio/glioma/pbta_all/km.png").exists()


def test_one_group_col_for_every_type_still_works():
    pytest.importorskip("torch")
    from src.post_analysis import _for_type
    assert _for_type("OS", "Glioma") == "OS" and _for_type({"Glioma": "label"}, "Wilms Tumor") is None
    assert _for_type(None, "Glioma", "all") == "all" and _for_type({"Glioma": "test"}, "Glioma", "all") == "test"
