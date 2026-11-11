"""post_analysis clustering: Ward clusters of the trained model's latent features, risk groups, KM."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("lifelines")

from src import graph_ai, post_analysis  # noqa: E402
from src.graph_constructor import graph_construct  # noqa: E402
from src.processor import data_process  # noqa: E402
from tests.test_clinical_graph import study  # noqa: E402,F401  (fixture)

OUT = "data/analysis/clusters/cbio/test_tumor/study1"


def _blobs(n=12, seed=0):
    rng = np.random.default_rng(seed)
    latent = np.vstack([rng.normal(0, 0.3, (n, 5)), rng.normal(4, 0.3, (n, 5))])
    clin = pd.DataFrame({"patientId": [f"P{i}" for i in range(2 * n)],
                         "OS_MONTHS": np.r_[rng.uniform(60, 120, n), rng.uniform(2, 30, n)],
                         "OS_STATUS_Binary": np.r_[np.zeros(n - 2), [1, 1], np.ones(n - 3), [0, 0, 0]]})
    return latent, clin


def test_two_clusters_lower_median_time_is_high_risk(tmp_path):
    latent, clin = _blobs()
    labels, df, stats = post_analysis.hierarchical_clustering_analysis(
        latent, clin, filename=str(tmp_path / "km.png"), title="OS by Hierarchical Clusters", publish=True)
    assert sorted(set(labels)) == [1, 2] and list(df.cluster) == list(labels)
    assert (df.risk_group.iloc[12:] == "High-risk").all() and (df.risk_group.iloc[:12] == "Low-risk").all()
    assert stats.group.tolist() == ["Low-risk", "High-risk"] and stats.n.tolist() == [12, 12]
    assert stats.logrank_p.iloc[0] < 0.001
    assert (tmp_path / "km.png").exists() and (tmp_path / "km.pdf").exists()     # publication: PNG + PDF


def test_patients_without_survival_are_clustered_but_not_plotted(tmp_path):
    latent, clin = _blobs()
    clin.loc[[0, 1, 20], "OS_MONTHS"] = np.nan
    _, df, stats = post_analysis.hierarchical_clustering_analysis(latent, clin, filename=str(tmp_path / "km.png"),
                                                                  title="t")
    assert df.risk_group.notna().all() and stats.n.sum() == 21
    assert not (tmp_path / "km.pdf").exists()                                    # default style: PNG only


def test_more_than_two_clusters_are_numbered(tmp_path):
    latent, clin = _blobs()
    latent[:4] += 20.0                                                           # a third group
    labels, df, stats = post_analysis.hierarchical_clustering_analysis(
        latent, clin, n_clusters=3, filename=str(tmp_path / "km.png"), title="t")
    assert len(set(labels)) == 3 and set(df.risk_group) == {"Cluster 1", "Cluster 2", "Cluster 3"}
    assert len(stats) == 3 and stats.logrank_p.notna().all()                     # multivariate log-rank


def _train(root, write, clustering):
    cfg = write(graph=False)
    cfg["model_mogonet"]["epochs"] = {"pretrain": 20, "train": 60}
    cfg["post_analysis"] = {"clustering": {"enabled": True, **clustering},
                            "survival": {"enabled": False, "style": "publication"}}
    write()
    # survival in the fetched clinical table: class B dies early
    f = root / "data/fetch/cbioportal_api_request/cbio/test_tumor/study1/clinical_patients.csv"
    clin = pd.read_csv(f)
    b = (clin.SUBTYPE == "B").to_numpy()
    clin["OS_MONTHS"] = np.where(b, 6.0, 80.0) + np.arange(len(clin))
    clin["OS_STATUS"] = np.where(b, "1:DECEASED", "0:LIVING")
    clin.loc[0, "OS_MONTHS"] = np.nan                                            # one patient without follow-up
    clin.to_csv(f, index=False)
    data_process()
    graph_construct()
    graph_ai.graph_ai_model()


def test_clusters_of_the_trained_model(study):
    root, write = study
    _train(root, write, {"subset": "all"})
    post_analysis.run_post_analysis()
    d = pd.read_csv(root / OUT / "clusters.tsv", sep="\t")
    assert len(d) == 30 and set(d.split) == {"train", "test"} and d.patientId.is_unique
    assert set(d.cluster) == {1, 2} and set(d.risk_group) == {"Low-risk", "High-risk"}
    med = d.groupby("risk_group").OS_MONTHS.median()
    assert med["High-risk"] < med["Low-risk"]
    assert d.OS_MONTHS.isna().sum() == 1 and d.status_binary.isna().sum() == 1
    s = pd.read_csv(root / OUT / "km_clusters_stats.tsv", sep="\t")
    assert s.group.tolist() == ["Low-risk", "High-risk"] and s.n.sum() == 29 and set(s.subset) == {"all"}
    x = pd.read_csv(root / OUT / "cluster_vs_class.tsv", sep="\t", index_col=0)
    assert x.to_numpy().sum() == 30
    assert (root / OUT / "km_clusters.png").exists() and (root / OUT / "km_clusters.pdf").exists()  # survival.style


def test_test_subset_and_latent_dimensions(study):
    root, write = study
    _train(root, write, {"subset": "test", "include_clinical": True})
    post_analysis.run_clustering()
    d = pd.read_csv(root / OUT / "clusters.tsv", sep="\t")
    y_te = pd.read_csv(root / "data/gdm/filtered/cbio/test_tumor/study1/labels_test.csv", index_col=0)
    assert set(d.split) == {"test"} and set(d.patientId) == set(y_te.index.astype(str))

    cfg = post_analysis._cfg()
    t = post_analysis._load_trained(cfg, "Test Tumor", None, task="clustering")
    assert t["views"] == ["mrna", "mirna", "clinical"] and "clinical" not in t["graph_views"]
    n_clin = t["X_tr"]["clinical"].shape[1]
    lat, ids, split = post_analysis._latent_features(t, "test", ["mrna", "mirna"])
    assert lat.shape == (len(y_te), 16)                       # last GCN layer (8) per omics view
    lat_c, _, _ = post_analysis._latent_features(t, "all", t["views"])
    assert lat_c.shape == (30, 16 + n_clin)                   # clinical without graph: its features
    # test rows: the encoder on the train-test graph, as in prediction
    z = graph_ai.encode(*post_analysis.tensors_from_numpy_lists(
        [t["base_X"][m] for m in t["views"]], [t["base_A"][m] for m in t["views"]], device="cpu"), t["model_dict"])
    n_tr = len(t["X_tr"]["mrna"])
    assert np.allclose(lat, np.hstack([z[0][n_tr:], z[1][n_tr:]]))
    assert np.allclose(lat_c[n_tr:, :8], z[0][n_tr:])


def _fits(groups, horizon, seed=0):
    from lifelines import KaplanMeierFitter
    rng = np.random.default_rng(seed)
    out = []
    for g, scale in groups:
        t = np.minimum(rng.exponential(scale, 30), horizon)
        out.append(KaplanMeierFitter().fit(t, (t < horizon).astype(int), label=g))
    return out


def test_publication_figures_line_up(tmp_path, capsys):
    """Same canvas and axes box whatever the title, groups, follow-up or legend placement."""
    import matplotlib.image as mpimg
    a = post_analysis._km_figure_publication(_fits([("Low-grade", 300), ("High-grade", 20)], 270),
                                             "OS by Grade Group (test set)", 3.2e-16, "Time (Months)",
                                             str(tmp_path / "a.png"))
    b = post_analysis._km_figure_publication(_fits([("Low-risk", 200), ("High-risk", 30)], 120, seed=1),
                                             "OS by Hierarchical Clusters (test set)", 0.0123, "Time (Months)",
                                             str(tmp_path / "b.png"), legend="inside")
    c = post_analysis._km_figure_publication(_fits([("A", 100), ("B", 50)], 90, seed=2),
                                             "A very long title that would not fit across the figure at 22 pt",
                                             0.5, "Time (Months)", str(tmp_path / "c.png"), xlim=(0, 300))
    assert a["size"] == b["size"] == c["size"] and a["axes"] == b["axes"] == c["axes"]
    assert a["legend"] == "below"
    assert "shortened" in capsys.readouterr().out
    shapes = {mpimg.imread(tmp_path / f"{k}.png").shape for k in "abc"}
    assert len(shapes) == 1 and (tmp_path / "a.pdf").exists()


def test_legend_inside_avoids_curves_and_bands():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 100, 300], [1.0, 0.57, 0.57], drawstyle="steps-post", label="a")
    ax.fill_between([0, 300], [0.7, 0.7], [1.0, 1.0], step="post", alpha=0.25)     # band across the top
    ax.set_ylim(-0.02, 1.04)
    h, l = ax.get_legend_handles_labels()
    assert post_analysis._legend_inside(fig, ax, h, l, 20)
    box = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
    assert not any(path.intersects_bbox(box, filled=f) for path, f in post_analysis._curve_paths(ax))
    ax.get_legend().remove()
    ax.fill_between([0, 300], [-0.02, -0.02], [1.04, 1.04], alpha=0.1)            # no free spot left
    assert not post_analysis._legend_inside(fig, ax, h, l, 20) and ax.get_legend() is None
    plt.close(fig)
