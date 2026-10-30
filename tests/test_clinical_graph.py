"""model_mogonet.clinical_view.graph: clinical view with or without a patient-similarity graph."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

from src import benchmark, graph_ai, post_analysis  # noqa: E402
from src.graph_constructor import (MOGONETGraphParams, build_trte_block_adjacency_mogonet,  # noqa: E402
                                   compute_adaptive_radius_parameter_mogonet, graph_construct)
from src.processor import data_process  # noqa: E402

GRAPHS = "data/gdm/graphs/cbio/test_tumor/study1"
N_PATIENTS = 30


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
        X[:8] += 2.0 * shift
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
        "patient_similarity_graph": {"edges_per_node": 3, "save_graphml": True},
        "model_mogonet": {"edges_per_node": 3, "gcn_hidden": [16, 16, 8], "checkpoints_dir": "data/models",
                          "epochs": {"pretrain": 2, "train": 5}, "device": "cpu",
                          "clinical_view": {"enabled": True, "features": ["AGE", "SEX"], "graph": True}},
        "post_analysis": {"feature_selection": {"enabled": True, "edges_per_node": 3,
                                                "out_dir": "data/analysis/feature_selection"}},
        "benchmark": {"n_splits": 3, "seed": 7, "out_dir": "data/analysis/benchmark",
                      "models": ["mogonet"]},
    }
    (tmp_path / "config").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "config/pipeline_config.yaml"))

    def write(**clinical_view):
        cfg["model_mogonet"]["clinical_view"].update(clinical_view)
        (tmp_path / "config/pipeline_config.yaml").write_text(yaml.safe_dump(cfg))
        return cfg

    write()
    return tmp_path, write


def test_graph_views_and_encoders():
    views = ["mrna", "mirna", "clinical"]
    assert graph_ai.graph_view_names(views) == ["mrna", "mirna"]
    assert graph_ai.graph_view_names(views, clinical_graph=True) == views
    for gv, enc, c_in in ((None, graph_ai.Identity_E, 7), (views, graph_ai.GCN_E, 8)):
        md = graph_ai.init_model_dict(views, 2, [20, 10, 7], [16, 16, 8], 8, graph_views=gv)
        assert isinstance(md["E3"], enc) and md["C3"].clf.in_features == c_in
        assert isinstance(md["E1"], graph_ai.GCN_E)


def test_old_checkpoints_keep_clinical_without_graph():
    assert post_analysis._graph_views({}, ["mrna", "clinical"]) == ["mrna"]
    assert post_analysis._graph_views({"graph_views": ["mrna", "clinical"]},
                                      ["mrna", "clinical"]) == ["mrna", "clinical"]


@pytest.mark.parametrize("use_graph", [False, True])
def test_pipeline_with_and_without_clinical_graph(study, use_graph):
    root, write = study
    write(graph=use_graph)
    data_process()
    graph_construct()

    g = root / GRAPHS
    assert (g / "mrna_train.adj.npz").exists() and (g / "mrna_train.graphml").exists()
    assert (g / "clinical_train.adj.npz").exists() == use_graph
    assert (g / "clinical_trte_block.graphml").exists() == use_graph
    # patient_similarity_graph.edges_per_node is read (about 3 connections per patient incl. self-loop,
    # not the default 15)
    import scipy.sparse as sp
    A = sp.load_npz(g / "mrna_train.adj.npz")
    assert A.getnnz() / A.shape[0] < 5

    graph_ai.graph_ai_model()
    meta = json.loads((root / "data/models/cbio/test_tumor/study1/meta.json").read_text())
    assert meta["views"] == ["mrna", "mirna", "clinical"]
    assert meta["graph_views"] == (["mrna", "mirna", "clinical"] if use_graph else ["mrna", "mirna"])
    model_dict, _ = graph_ai.load_model_bundle(root / "data/models/cbio/test_tumor/study1")
    assert isinstance(model_dict["E3"], graph_ai.GCN_E if use_graph else graph_ai.Identity_E)

    post_analysis.run_feature_selection()                       # rebuilds the clinical graph if trained with one
    out = root / "data/analysis/feature_selection/cbio/test_tumor/study1"
    assert (out / "mrna_selection.csv").exists() and not (out / "clinical_selection.csv").exists()

    benchmark.run_benchmark()
    settings = json.loads((root / "data/analysis/benchmark/cv_settings.json").read_text())
    assert settings["model_mogonet"]["clinical_graph"] is use_graph


@pytest.mark.parametrize("use_graph", [False, True])
def test_post_analysis_rebuilds_the_graphs_used_in_training(tmp_path, use_graph):
    rng = np.random.default_rng(3)
    ids = [f"P{i}" for i in range(36)]
    y = pd.Series([0, 1] * 18, index=ids)
    om = pd.DataFrame(rng.normal(0, 1, (36, 12)), index=ids, columns=[f"g{j}" for j in range(12)])
    om.iloc[:, :3] += 2 * y.values[:, None]
    cl = pd.DataFrame({"AGE": rng.random(36), "SEX_F": rng.integers(0, 2, 36)}, index=ids).astype(np.float32)
    tr, te = ids[:28], ids[28:]
    views = ["mrna", "clinical"]
    X = {"mrna": om, "clinical": cl}
    _, md = graph_ai.train_with_split(
        views, {v: X[v].loc[tr] for v in views}, {v: X[v].loc[te] for v in views}, y.loc[tr], y.loc[te],
        edges_per_node=3, dim_he_list=(8, 8, 4), num_epoch_pretrain=2, num_epoch=3, seed=5,
        save_dir=tmp_path, clinical_graph=use_graph)
    assert isinstance(md["E1"], graph_ai.GCN_E)
    assert isinstance(md["E2"], graph_ai.GCN_E if use_graph else graph_ai.Identity_E)

    loaded, meta = graph_ai.load_model_bundle(tmp_path)
    prob, te_idx, *_ = post_analysis._baseline_probs(loaded, views, {v: X[v].loc[tr] for v in views},
                                                     {v: X[v].loc[te] for v in views}, 3, meta)

    def probs(clinical_adj: bool):
        params = MOGONETGraphParams(edges_per_node=3)
        X_list, A_list = [], []
        for v in views:
            a, b = X[v].loc[tr].to_numpy(np.float32), X[v].loc[te].to_numpy(np.float32)
            X_list.append(np.vstack([a, b]))
            if v == "clinical" and not clinical_adj:
                A_list.append(None)
            else:
                r = compute_adaptive_radius_parameter_mogonet(a, params)
                A_list.append(build_trte_block_adjacency_mogonet(a, b, radius=r, params=params))
        return graph_ai.test_epoch(*graph_ai.tensors_from_numpy_lists(X_list, A_list), te_idx, loaded)

    assert np.allclose(prob, probs(use_graph), atol=1e-6)      # the graphs used in training are rebuilt
    if use_graph:
        assert not np.allclose(prob, probs(False), atol=1e-6)  # and the clinical graph matters
