"""post_analysis feature importance: probability score (no ties at 0) on the test split or over CV folds."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src import graph_ai, post_analysis  # noqa: E402
from src.graph_constructor import graph_construct  # noqa: E402
from src.processor import data_process  # noqa: E402
from tests.test_clinical_graph import study  # noqa: E402,F401  (fixture)

OUT = "data/analysis/feature_selection/cbio/test_tumor/study1"


def _train(root, write, **fs):
    cfg = write(graph=False)
    cfg["model_mogonet"]["epochs"] = {"pretrain": 20, "train": 60}
    cfg["post_analysis"]["feature_selection"].update(fs)
    write()
    data_process()
    graph_construct()
    graph_ai.graph_ai_model()


def test_test_split_probability_score(study):
    root, write = study
    _train(root, write)                                    # score: prob, evaluate_on: test (defaults)
    post_analysis.run_feature_selection()
    d = pd.read_csv(root / OUT / "mrna_selection.csv")
    assert list(d.columns) == ["feature", "delta_prob", "delta_f1"] and len(d) == 20
    assert (d.delta_prob.diff().dropna() <= 0).all()                      # ranked by delta_prob
    assert (d.delta_prob != 0).sum() >= 18                                # a value for (almost) every feature
    assert d.delta_prob.nunique() >= 18
    # the eight genes shifted between the classes are the informative ones
    top = set(d.feature.head(8))
    assert len(top & {f"GENE{i}" for i in range(8)}) >= 5


def test_f1_score_keeps_mogonet_ranking_with_stable_ties(study):
    root, write = study
    _train(root, write, score="f1")
    post_analysis.run_feature_selection()
    d = pd.read_csv(root / OUT / "mrna_selection.csv")
    assert (d.delta_f1.diff().dropna() <= 0).all()
    for _, g in d.groupby("delta_f1", sort=False):                        # ties in feature-name order
        assert list(g.feature) == sorted(g.feature)


def test_cv_folds(study):
    root, write = study
    _train(root, write, evaluate_on="cv")
    post_analysis.run_feature_selection()
    d = pd.read_csv(root / OUT / "mrna_selection_cv.csv")
    assert list(d.columns) == ["feature", "delta_prob", "delta_prob_sd", "folds_positive", "delta_f1"]
    assert len(d) == 20 and d.folds_positive.between(0, 3).all()         # benchmark.n_splits = 3
    assert (d.delta_prob.diff().dropna() <= 0).all() and d.delta_prob_sd.notna().all()
    assert (d.delta_prob != 0).sum() >= 18
    assert not (root / OUT / "mrna_selection.csv").exists()


def test_ablation_matches_a_direct_prediction():
    """_ablate_view: the drop for a feature equals base - prediction with that feature zeroed."""
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.random((20, 6)).astype(np.float32), rng.random((5, 6)).astype(np.float32)
    ytr = pd.Series([0, 1] * 10)
    yte = pd.Series([0, 1, 0, 1, 0])
    _, md = graph_ai.train_with_split(["mrna"], {"mrna": pd.DataFrame(Xtr)}, {"mrna": pd.DataFrame(Xte)}, ytr, yte,
                                      edges_per_node=3, dim_he_list=(8, 8, 4), num_epoch_pretrain=2, num_epoch=3)
    params = post_analysis.MOGONETGraphParams(edges_per_node=3)
    bx, ba = post_analysis._trte_view(Xtr, Xte, params, graph=True)
    te_idx = list(range(20, 25))
    base = post_analysis.test_epoch(*post_analysis.tensors_from_numpy_lists([bx], [ba], device="cpu"), te_idx, md)
    y_idx = yte.to_numpy()
    dprob, df1 = post_analysis._ablate_view(md, ["mrna"], "mrna", {"mrna": bx}, {"mrna": ba}, Xtr, Xte, te_idx,
                                            params, y_idx, base)
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    Xtr2[:, 2] = 0
    Xte2[:, 2] = 0
    x2, a2 = post_analysis._trte_view(Xtr2, Xte2, params, graph=True)
    p2 = post_analysis.test_epoch(*post_analysis.tensors_from_numpy_lists([x2], [a2], device="cpu"), te_idx, md)
    rows = np.arange(5)
    assert np.allclose(dprob[2], base[rows, y_idx] - p2[rows, y_idx]) and dprob.shape == (6, 5)
