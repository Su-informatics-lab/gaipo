"""Synthetic end-to-end test of src/benchmark.py (no network, a few seconds on CPU)."""
import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

from src import benchmark  # noqa: E402
from src.processor import data_process  # noqa: E402

N_PATIENTS, N_INFORMATIVE = 30, 8


def _write_study(root, rng):
    """cBioPortal-like study folder as written by data_extract (two omics, patient labels)."""
    d = root / "data/fetch/cbioportal_api_request/cbio/test_tumor/study1"
    d.mkdir(parents=True)
    pids = [f"P{i:02d}" for i in range(N_PATIENTS)]
    label = np.array(["A", "B"] * (N_PATIENTS // 2))
    pd.DataFrame({"patientId": pids, "SUBTYPE": label,
                  "AGE": rng.integers(1, 15, N_PATIENTS),
                  "SEX": rng.choice(["Male", "Female"], N_PATIENTS)}).to_csv(d / "clinical_patients.csv", index=False)
    pd.DataFrame({"sampleId": [f"{p}-01" for p in pids], "patientId": pids}).to_csv(
        d / "clinical_samples.csv", index=False)
    shift = (label == "B").astype(float)
    for fname, prefix, n_feat in [("study1_rna_seq_mrna_median_all_sample_Zscores.tsv", "GENE", 60),
                                  ("study1_mirna.tsv", "hsa-mir-", 40)]:
        X = rng.normal(5.0, 1.0, size=(n_feat, N_PATIENTS))
        X[:N_INFORMATIVE] += 2.0 * shift
        df = pd.DataFrame(X, columns=[f"{p}-01" for p in pids])
        df.insert(0, "entrezGeneId", range(n_feat))
        df.insert(0, "hugoGeneSymbol", [f"{prefix}{i}" for i in range(n_feat)])
        df.to_csv(d / fname, sep="\t", index=False)


@pytest.fixture()
def study(tmp_path, monkeypatch):
    _write_study(tmp_path, np.random.default_rng(0))
    cfg = {
        "cancer_type_interest": ["Test Tumor"],
        "cBio_studyId": {"Test Tumor": "study1"},
        "expression_profiles": {"Test Tumor": ["mrna", "mirna"]},
        "processor": {"labels": {"Test Tumor": {
            "source": "patients", "column": "SUBTYPE", "map": "A=0,B=1",
            "var_threshold": [0.0, 0.0], "alpha": [0.5, 0.5], "top_k": [20, 10],
            "test_size": 0.2, "seed": 42}}},
        "model_mogonet": {"edges_per_node": 3, "gcn_hidden": [16, 16, 8],
                          "epochs": {"pretrain": 2, "train": 5}, "device": "cpu",
                          "clinical_view": {"enabled": True, "features": ["AGE", "SEX"]}},
        "benchmark": {"n_splits": 3, "seeds": [0, 1], "out_dir": "data/analysis/benchmark",
                      "random_forest": {"n_estimators": 20}},
    }
    (tmp_path / "config").mkdir()
    (tmp_path / "config/pipeline_config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "config/pipeline_config.yaml"))
    return tmp_path, cfg


def test_in_fold_selection_never_looks_at_the_test_fold():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(5, 1, (40, 30)), columns=[f"g{i}" for i in range(30)])
    y = pd.Series([0, 1] * 20)
    X.loc[y == 1, ["g0", "g1", "g2"]] += 3
    tr, te = X.index[:30], X.index[30:]
    a_tr, a_te = benchmark.select_in_fold(X.loc[tr], X.loc[te], y.loc[tr], 0, 0.0, 0.05, 5)
    X2 = X.copy()
    X2.loc[te] = rng.normal(100, 50, (len(te), 30))           # change only the test fold
    b_tr, b_te = benchmark.select_in_fold(X2.loc[tr], X2.loc[te], y.loc[tr], 0, 0.0, 0.05, 5)
    assert list(a_tr.columns) == list(b_tr.columns)
    pd.testing.assert_frame_equal(a_tr, b_tr)                 # same features, same scaling
    assert {"g0", "g1", "g2"} <= set(a_tr.columns)
    assert a_tr.min().min() >= 0 and a_tr.max().max() <= 1 + 1e-6


@pytest.mark.parametrize("mode", ["in_fold", "precomputed"])
def test_benchmark_writes_folds_and_mean_sd(study, mode):
    root, cfg = study
    cfg["benchmark"]["feature_selection"] = mode
    (root / "config/pipeline_config.yaml").write_text(yaml.safe_dump(cfg))
    if mode == "precomputed":
        data_process()
    benchmark.run_benchmark()

    out = root / "data/analysis/benchmark"
    folds = pd.read_csv(out / "cbio/test_tumor/study1/cv_folds.csv")
    assert len(folds) == 3 * 2 * 3                            # models x seeds x folds
    assert set(folds.model) == {"mogonet", "naive_bayes", "random_forest"}
    assert folds.groupby(["model", "seed"]).n_test.sum().eq(N_PATIENTS).all()
    assert folds[["acc", "f1", "auc"]].notna().all().all()

    splits = pd.read_csv(out / "cbio/test_tumor/study1/cv_splits.csv")
    assert splits.groupby("seed").patientId.nunique().eq(N_PATIENTS).all()   # each patient tested once per seed
    assert not splits.duplicated(["seed", "patientId"]).any()
    assert splits.groupby(["seed", "test_fold"]).label.nunique().eq(2).all()  # stratified

    summ = pd.read_csv(out / "cv_summary.csv").set_index("model")
    for m, g in folds.groupby("model"):
        assert summ.loc[m, "n_folds"] == 6
        assert np.isclose(summ.loc[m, "acc_mean"], g.acc.mean())
        assert np.isclose(summ.loc[m, "auc_sd"], g.auc.std(ddof=1))
    assert "±" in (out / "cv_summary.md").read_text()
    assert (out / "cv_settings.json").exists()
