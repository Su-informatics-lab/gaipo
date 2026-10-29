"""Feature selection (same steps as the PCGS preprocessing) and CNA support in src/processor.py."""
import numpy as np
import pandas as pd

from src.processor import SELECTION_SUMMARY, _find_omics_file, feature_filter_scaler, load_and_prepare


def _data(n=80, p=400, informative=30, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.gamma(2.0, 1.0, (n, p)), index=[f"P{i}" for i in range(n)],
                     columns=[f"g{j}" for j in range(p)])
    y = pd.Series([0, 1] * (n // 2), index=X.index)
    X.iloc[:, :informative] += 2.0 * y.values[:, None]
    X["all_zero"] = 0.0
    X["mostly_zero"] = np.r_[np.ones(10), np.zeros(n - 10)]
    return X, y


def test_standard_path_keeps_top_k_of_the_fdr_survivors():
    X, y = _data()
    df, meta = feature_filter_scaler(X, y, 0.0, 0.05, 10, return_meta=True)
    assert meta["after_basic"] == 400                    # all-zero and mostly-zero columns dropped
    assert meta["after_fdr"] >= 10 and not meta["used_fallback"]
    assert df.shape[1] == 10 and set(df.columns) <= {f"g{j}" for j in range(30)}
    assert df.min().min() >= 0 and df.max().max() <= 1


def test_fallback_when_fewer_than_k_pass_fdr():
    X, y = _data()
    df, meta = feature_filter_scaler(X, y, 0.0, 0.05, 100, return_meta=True)
    assert meta["after_fdr"] < 100 and meta["used_fallback"]
    assert df.shape[1] == 100                            # top 100 by F after the variance filter
    assert {f"g{j}" for j in range(30)} <= set(df.columns)
    df2, meta2 = feature_filter_scaler(X, y, 0.0, 0.05, 100, fallback_if_fdr_too_few=False, return_meta=True)
    assert df2.shape[1] == meta2["after_fdr"] < 100 and not meta2["used_fallback"]


def test_variance_threshold_and_per_modality_lists():
    X, y = _data()
    X["low_var"] = 1.0 + 1e-3 * np.arange(len(X))
    _, meta = feature_filter_scaler(X, y, [0.5, 0.0], [0.05, 0.05], [10, 10], mod_idx=0, return_meta=True)
    _, meta_b = feature_filter_scaler(X, y, [0.5, 0.0], [0.05, 0.05], [10, 10], mod_idx=1, return_meta=True)
    assert meta["after_var"] < meta_b["after_var"]


def test_linear_cna_file_is_found_and_preferred_over_gistic(tmp_path):
    for name in ("pbta_all_cna.tsv", "pbta_all_linear_CNA.tsv", "pbta_all_rna_seq_mrna.tsv"):
        (tmp_path / name).write_text("hugoGeneSymbol\tentrezGeneId\n")
    assert _find_omics_file(tmp_path, "linear_cna").name == "pbta_all_linear_CNA.tsv"
    assert _find_omics_file(tmp_path, "mrna").name == "pbta_all_rna_seq_mrna.tsv"


def test_load_and_prepare_with_mrna_and_cna(tmp_path):
    rng = np.random.default_rng(1)
    n = 40
    pids = [f"PT_{i:02d}" for i in range(n)]
    lab = np.array(["low", "high"] * (n // 2))
    pd.DataFrame({"patientId": pids, "AGE": rng.integers(1, 18, n)}).to_csv(tmp_path / "clinical_patients.csv", index=False)
    pd.DataFrame({"sampleId": [f"BS_{i:02d}" for i in range(n)], "patientId": pids,
                  "CANCER_TYPE": lab}).to_csv(tmp_path / "clinical_samples.csv", index=False)
    ccdi = tmp_path / "ccdi.csv"
    pd.DataFrame({"sample_id": [f"BS_{i:02d}" for i in range(n)], "subject_id": pids}).to_csv(ccdi, index=False)
    shift = (lab == "high").astype(float)
    for fname, k in (("study_rna_seq_mrna.tsv", 200), ("study_linear_CNA.tsv", 150)):
        M = rng.normal(0.5, 1.0, (k, n)) + np.outer(np.r_[np.ones(20), np.zeros(k - 20)], shift)
        df = pd.DataFrame(M, columns=[f"BS_{i:02d}" for i in range(n)])
        df.insert(0, "entrezGeneId", range(k))
        df.insert(0, "hugoGeneSymbol", [f"G{j}" for j in range(k)])
        df.to_csv(tmp_path / fname, sep="\t", index=False)
    out = load_and_prepare(tmp_path, ccdi, "Glioma", ["mrna", "linear_cna"], "samples", "CANCER_TYPE",
                           top_k=[50, 40], var_threshold=[0.1, 0.01], alpha=[0.05, 0.05], label_map="low=0,high=1")
    omics = out[7]
    assert omics["mrna"].shape == (n, 50) and omics["linear_cna"].shape == (n, 40)
    assert [s["modality"] for s in SELECTION_SUMMARY] == ["mrna", "linear_cna"]
    assert all(s["final"] == s["requested_k"] for s in SELECTION_SUMMARY)
