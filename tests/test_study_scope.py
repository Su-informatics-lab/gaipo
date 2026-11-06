"""modeling_studyId: fetch/extract use every configured study; modeling steps use the selected ones."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.study_scope import configured_studies, is_modeled, modeling_studies

ROOT = Path(__file__).resolve().parents[1]
CFG = {"cancer_type_interest": ["Glioma"],
       "cBio_studyId": {"Glioma": "brain_cptac_2020"},
       "pedcBio_studyId": {"Glioma": "pbta_all"}}


def test_selection():
    assert configured_studies(CFG, "Glioma") == [("cbio", "brain_cptac_2020"), ("pedcbio", "pbta_all")]
    assert modeling_studies(CFG, "Glioma") == configured_studies(CFG, "Glioma")       # no key: all, as before
    for sel in ("pbta_all", "pedcbio:pbta_all", ["pbta_all"]):
        cfg = dict(CFG, modeling_studyId={"Glioma": sel})
        assert modeling_studies(cfg, "Glioma") == [("pedcbio", "pbta_all")]
        assert not is_modeled(cfg, "Glioma", "cbio", "brain_cptac_2020")
    with pytest.raises(ValueError, match="pbta"):
        modeling_studies(dict(CFG, modeling_studyId={"Glioma": "pbta"}), "Glioma")   # typo is an error


def test_post_analysis_uses_the_modeled_study():
    pytest.importorskip("torch")
    from src import post_analysis
    assert post_analysis._study_for_ct(CFG, "Glioma") == ("cbio", "brain_cptac_2020")   # old behavior
    cfg = dict(CFG, modeling_studyId={"Glioma": "pbta_all"})
    assert post_analysis._study_for_ct(cfg, "Glioma") == ("pedcbio", "pbta_all")
    assert [s for _, _, s in post_analysis._iter_studies(cfg)] == ["pbta_all"]


def test_processor_skips_studies_that_are_not_modeled(tmp_path, monkeypatch, capsys):
    from src import processor
    cfg = dict(CFG, modeling_studyId={"Glioma": "pbta_all"}, expression_profiles={"Glioma": ["mrna"]},
               processor={"labels": {"Glioma": {"column": "CANCER_TYPE"}}})
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "cfg.yaml"))
    monkeypatch.chdir(tmp_path)
    processor.data_process()                       # no extracts here: each modeled study is reported as missing
    out = capsys.readouterr().out
    assert "pedcbio:pbta_all" in out and "brain_cptac_2020" not in out


def test_data_model_skips_studies_that_are_not_modeled(tmp_path, monkeypatch, capsys):
    pytest.importorskip("duckdb")
    from src import data_model
    fetch = tmp_path / "fetch"
    for portal, study, pts in (("cbio", "brain_cptac_2020", ["C1", "C2"]), ("pedcbio", "pbta_all", ["P1", "P2"])):
        d = fetch / portal / "glioma" / study
        d.mkdir(parents=True)
        pd.DataFrame({"patientId": pts, "SEX": ["Female", "Male"]}).to_csv(d / "clinical_patients.csv", index=False)
        pd.DataFrame({"sampleId": [f"S{p}" for p in pts], "patientId": pts,
                      "CANCER_TYPE_DETAILED": ["Low-grade glioma, KIAA1549-BRAF"] * 2}).to_csv(
            d / "clinical_samples.csv", index=False)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(dict(CFG, modeling_studyId={"Glioma": "pbta_all"})))
    gm = yaml.safe_load(open(ROOT / "config/gdc_mapping.yaml"))
    gm["cancer_type_interest"] = ["Glioma"]
    (tmp_path / "gdc_mapping.yaml").write_text(yaml.safe_dump(gm))
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "cfg.yaml"))
    monkeypatch.setenv("GDC_MAPPING_PATH", str(tmp_path / "gdc_mapping.yaml"))
    data_model.build_data_model(gdm_root=str(tmp_path / "gdm"), fetch_root=str(fetch),
                                schemas_dir=str(ROOT / "src/gdcdictionary/schemas"))
    out = capsys.readouterr().out
    assert "not modeled (modeling_studyId): Glioma | cbio | brain_cptac_2020" in out
    built = sorted(p.name for p in (tmp_path / "gdm/by_source").glob("*/*/*"))
    assert built == ["pbta_all"]
