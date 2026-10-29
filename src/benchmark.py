"""
benchmark.py - Stratified k-fold cross-validation: MOGONET vs. Naive Bayes vs. Random Forest

For each cancer type and study in the config, the modeling cohort built by processor.py (patients
with a label and every configured omics modality) is split once with
StratifiedKFold(n_splits, shuffle=True, random_state=seed), using the single benchmark.seed. The
same seed initializes MOGONET and Random Forest. Within a fold, all models see the same
training/test patients and the same features.

feature_selection (config benchmark.feature_selection)
  in_fold      (default) processor.py's selection (VarianceThreshold -> SelectFdr(ANOVA F) ->
               SelectKBest, per-modality var_threshold / alpha / top_k from processor.labels) is
               refitted on the training fold only, and min-max scaling uses training-fold min/max.
               Test-fold labels are never used for selection, scaling or training.
  precomputed  reuses processor outputs ({mod}.parquet), whose features were selected and scaled
               on all patients, so each test fold took part in feature selection and the
               estimates are optimistic. Faster; kept for comparison with earlier runs.

Models (config benchmark.models)
  mogonet        graph_ai.train_with_split: one GCN per omics view + VCDN fusion; hyperparameters
                 from model_mogonet. A clinical view is added when model_mogonet.clinical_view is
                 enabled. Metrics are read at the last epoch (no early stopping on the test fold).
                 As in MOGONET, test patients are nodes of the train+test graph at inference
                 (their features, not their labels, are used).
  naive_bayes    sklearn GaussianNB on the concatenated views
  random_forest  sklearn RandomForestClassifier(random_state=seed) on the concatenated views

Metrics: accuracy, macro F1 and ROC AUC on each test fold. Summary: mean and SD over the n_splits
test folds (benchmark.sd_ddof: 1 = sample SD, the default; 0 = population SD, as numpy.std).

Input   in_fold:     data/fetch/cbioportal_api_request/<portal>/<ct_slug>/<study_id>/  (data_extract)
        precomputed: data/gdm/filtered/<portal>/<ct_slug>/<study_id>/                  (data_process)
Output  <out_dir>/<portal>/<ct_slug>/<study_id>/cv_folds.csv    one row per model x fold
        <out_dir>/<portal>/<ct_slug>/<study_id>/cv_splits.csv   patientId, label, test fold
        <out_dir>/<portal>/<ct_slug>/<study_id>/cv_summary.csv  mean / SD per model
        <out_dir>/cv_summary.csv, <out_dir>/cv_summary.md       all studies, "mean ± SD"
        <out_dir>/cv_settings.json                              settings used for the run

Run:
  python -m src.main --call benchmark
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

from . import graph_ai as gai
from .processor import feature_filter_scaler, load_and_prepare

MODELS = ("mogonet", "naive_bayes", "random_forest")
METRICS = ("acc", "f1", "auc")
METRIC_NAMES = {"acc": "Accuracy", "f1": "Macro F1", "auc": "AUC"}
OMICS = ("mrna", "mirna", "methylation", "linear_cna")


# ==========================
# Config helpers
# ==========================
def _load_cfg() -> dict:
    with open(os.getenv("CONFIG_PATH", "config/pipeline_config.yaml"), "r") as f:
        return yaml.safe_load(f) or {}


def _bench_cfg(cfg: dict) -> dict:
    b = dict(cfg.get("benchmark", {}) or {})
    b["n_splits"] = int(b.get("n_splits", 5))
    b["seed"] = int(b.get("seed", 42))
    b["sd_ddof"] = int(b.get("sd_ddof", 1))
    b["models"] = [str(m).lower() for m in (b.get("models") or list(MODELS))]
    b["feature_selection"] = str(b.get("feature_selection", "in_fold")).lower()
    b["naive_bayes"] = dict(b.get("naive_bayes") or {})
    b["random_forest"] = dict(b.get("random_forest") or {})
    b["out_dir"] = str(b.get("out_dir", "data/analysis/benchmark"))
    unknown = sorted(set(b["models"]) - set(MODELS))
    if unknown:
        raise ValueError(f"[benchmark] unknown model(s) {unknown}; choose from {list(MODELS)}")
    if b["feature_selection"] not in {"in_fold", "precomputed"}:
        raise ValueError("[benchmark] feature_selection must be 'in_fold' or 'precomputed'")
    if b["n_splits"] < 2:
        raise ValueError("[benchmark] n_splits must be >= 2")
    if b["sd_ddof"] not in (0, 1):
        raise ValueError("[benchmark] sd_ddof must be 0 or 1")
    return b


def _mogonet_cfg(cfg: dict) -> dict:
    """Same keys and defaults as graph_ai.graph_ai_model()."""
    m = cfg.get("model_mogonet", {}) or {}
    epochs, lr = m.get("epochs", {}) or {}, m.get("lr", {}) or {}
    device = str(m.get("device", "cpu")).lower()
    if device == "cuda" and not gai.torch.cuda.is_available():
        device = "cpu"
    return {
        "edges_per_node": int(m.get("edges_per_node", 5)),
        "dim_he_list": tuple(m.get("gcn_hidden", [400, 400, 200])),
        "num_epoch_pretrain": int(epochs.get("pretrain", 50)),
        "num_epoch": int(epochs.get("train", 400)),
        "lr_e_pretrain": float(lr.get("enc_pretrain", 1e-3)),
        "lr_e": float(lr.get("enc", 5e-4)),
        "lr_c": float(lr.get("clf", 1e-3)),
        "device": device,
    }


def _label_cfg(cfg: dict, ct: str) -> dict:
    """Same keys and defaults as processor.data_process()."""
    lc = ((cfg.get("processor", {}) or {}).get("labels", {}) or {}).get(ct, {}) or {}
    return {
        "label_source": lc.get("source", "samples"),
        "label_column": lc.get("column", None),
        "label_map": lc.get("map", None),
        "var_threshold": lc.get("var_threshold", 0.01),
        "alpha": lc.get("alpha", 0.05),
        "top_k": lc.get("top_k", 500),
        "test_size": float(lc.get("test_size", 0.2)),
        "random_state": int(lc.get("seed", 42)),
        "fdr_fallback": bool(lc.get("fdr_fallback", True)),
    }


# ==========================
# Cohort loading
# ==========================
def _read_labels(path: Path) -> pd.Series:
    y = pd.read_csv(path, index_col=0).iloc[:, 0]
    y.index = y.index.astype(str)
    return y


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = df.index.astype(str)
    return df


def load_cohort(cfg: dict, ct: str, portal: str, study_id: str, mods: List[str],
                mode: str) -> Tuple[Dict[str, pd.DataFrame], pd.Series, Optional[pd.DataFrame]]:
    """Return ({mod: patients x features}, labels, clinical table indexed by patientId or None),
    all restricted to the same patients in the same order."""
    ct_slug = gai._slug(ct)
    clin = None
    if mode == "precomputed":
        root = Path("data/gdm/filtered") / portal / ct_slug / study_id
        if not (root / "labels_train.csv").exists():
            raise FileNotFoundError(f"{root}/labels_train.csv not found (run data_process first).")
        y = pd.concat([_read_labels(root / "labels_train.csv"), _read_labels(root / "labels_test.csv")])
        X = {m: _dedupe_columns(pd.read_parquet(root / f"{m}.parquet")) for m in mods}
        if (root / "clinical_filtered.parquet").exists():
            clin = pd.read_parquet(root / "clinical_filtered.parquet")
    else:
        raw_dir = Path("data/fetch/cbioportal_api_request") / portal / ct_slug / study_id
        if not raw_dir.exists():
            raise FileNotFoundError(f"{raw_dir} not found (run data_extract first).")
        ccdi = Path("data/fetch/ccdi_api_request/sample_ids") / f"{ct_slug}.csv"
        lc = _label_cfg(cfg, ct)
        out = load_and_prepare(
            rawData_dir=raw_dir, ccdi_csv=ccdi if ccdi.exists() else None, cancer_type=ct,
            modalities=mods, label_source=lc["label_source"], label_column=lc["label_column"],
            top_k=lc["top_k"], test_size=lc["test_size"], random_state=lc["random_state"],
            label_map=lc["label_map"], var_threshold=lc["var_threshold"], alpha=lc["alpha"],
            select_features=False,
        )
        _, _, y_tr, y_te, _, _, _, omics, clin, _, _ = out
        y = pd.concat([y_tr, y_te])
        y.index = y.index.astype(str)
        X = {m: _dedupe_columns(omics[m]) for m in mods}

    ids = y.index
    for m in mods:
        ids = ids.intersection(X[m].index)
    ids = pd.Index(sorted(ids))
    y = y.loc[ids]
    X = {m: X[m].loc[ids] for m in mods}
    if clin is not None and "patientId" in clin.columns:
        clin = clin.assign(patientId=clin["patientId"].astype(str)).drop_duplicates("patientId")
        clin = clin.set_index("patientId").reindex(ids)
    else:
        clin = None
    return X, y, clin


# ==========================
# Per-fold features (fit on the training fold only)
# ==========================
def select_in_fold(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, mod_idx: int,
                   var_threshold, alpha, top_k, fdr_fallback: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """processor.feature_filter_scaler fitted on the training fold; the test fold gets the same
    columns, training-fold median imputation and training-fold min-max scaling."""
    try:
        sel = feature_filter_scaler(X_tr, y_tr, var_threshold, alpha, top_k, standard=False,
                                    minimax=False, mod_idx=mod_idx, fallback_if_fdr_too_few=fdr_fallback)
        cols = list(sel.columns)
    except ValueError as e:
        if "0 feature" not in str(e):
            raise
        cols = []
    if not cols:
        raise RuntimeError("no feature passed the selection on this training fold; lower this "
                           "modality's var_threshold (or set fdr_fallback: true), or use "
                           "benchmark.feature_selection: precomputed")
    tr = X_tr[cols].astype(float)
    te = X_te[cols].astype(float).fillna(tr.median())
    lo, rng = tr.min(), tr.max() - tr.min() + 1e-12
    return ((tr - lo) / rng).astype(np.float32), ((te - lo) / rng).astype(np.float32)


def clinical_in_fold(c_tr: pd.DataFrame, c_te: pd.DataFrame,
                     features: List[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Clinical view as in graph_ai._build_clinical_view (one-hot categoricals, min-max numerics),
    but categories and min/max come from the training fold only."""
    feats = [f for f in features if f in c_tr.columns]
    if not feats:
        return None, None
    num = [f for f in feats if pd.api.types.is_numeric_dtype(c_tr[f])]
    cat = [f for f in feats if f not in num]
    tr = pd.get_dummies(c_tr[feats], columns=cat, dummy_na=True).astype(float)
    te = pd.get_dummies(c_te[feats], columns=cat, dummy_na=True).astype(float)
    te = te.reindex(columns=tr.columns, fill_value=0.0)
    lo, hi = tr.min(), tr.max()
    rng = (hi - lo).where(hi - lo > 1e-12, np.inf)   # constant columns -> 0
    tr, te = (tr - lo) / rng, (te - lo) / rng
    return (tr.fillna(0.0).astype(np.float32), te.fillna(0.0).astype(np.float32))


def fold_views(X: Dict[str, pd.DataFrame], y: pd.Series, clin: Optional[pd.DataFrame],
               tr: pd.Index, te: pd.Index, mods: List[str], mode: str, lc: dict,
               clinical_features: Optional[List[str]]):
    Xtr, Xte = {}, {}
    for mi, m in enumerate(mods):
        if mode == "precomputed":
            Xtr[m], Xte[m] = X[m].loc[tr], X[m].loc[te]
        else:
            try:
                Xtr[m], Xte[m] = select_in_fold(X[m].loc[tr], X[m].loc[te], y.loc[tr], mi,
                                                lc["var_threshold"], lc["alpha"], lc["top_k"],
                                                lc["fdr_fallback"])
            except RuntimeError as e:
                raise RuntimeError(f"{m}: {e}") from e
    if clinical_features and clin is not None:
        c_tr, c_te = clinical_in_fold(clin.loc[tr], clin.loc[te], clinical_features)
        if c_tr is not None:
            Xtr["clinical"], Xte["clinical"] = c_tr, c_te
    return Xtr, Xte


# ==========================
# Models
# ==========================
def _scores(y_true: np.ndarray, prob: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    pred = classes[prob.argmax(1)]
    out = {"acc": accuracy_score(y_true, pred), "f1": f1_score(y_true, pred, average="macro")}
    try:
        out["auc"] = (roc_auc_score(y_true, prob[:, 1]) if len(classes) == 2
                      else roc_auc_score(y_true, prob, multi_class="ovr", labels=classes))
    except ValueError:
        out["auc"] = np.nan
    return {k: float(v) for k, v in out.items()}


def _run_sklearn(name: str, Xtr: Dict[str, pd.DataFrame], Xte: Dict[str, pd.DataFrame],
                 y_tr: pd.Series, y_te: pd.Series, views: List[str], seed: int, b: dict):
    A_tr = np.hstack([Xtr[v].to_numpy(np.float32) for v in views])
    A_te = np.hstack([Xte[v].to_numpy(np.float32) for v in views])
    if name == "naive_bayes":
        clf = GaussianNB(**b["naive_bayes"])
    else:
        kw = dict(b["random_forest"])
        kw.setdefault("random_state", seed)
        clf = RandomForestClassifier(**kw)
    clf.fit(A_tr, y_tr.to_numpy())
    return _scores(y_te.to_numpy(), clf.predict_proba(A_te), clf.classes_)


def _run_mogonet(Xtr, Xte, y_tr, y_te, views: List[str], seed: int, hp: dict):
    metrics, _ = gai.train_with_split(views, dict(Xtr), dict(Xte), y_tr, y_te,
                                      save_dir=None, seed=seed, **hp)
    return {k: float(metrics.get(k, np.nan)) for k in METRICS}


# ==========================
# Summary
# ==========================
def summarize(folds: pd.DataFrame, models: List[str], ddof: int = 1) -> pd.DataFrame:
    keys = ["cancer_type", "portal", "study_id", "model"]
    g = folds.groupby(keys, sort=False)
    out = g.size().rename("n_folds").to_frame()
    for k in METRICS:
        out[f"{k}_mean"] = g[k].mean()
        out[f"{k}_sd"] = g[k].std(ddof=ddof)
    out = out.reset_index()
    out["model"] = pd.Categorical(out["model"], categories=models, ordered=True)
    return out.sort_values(["cancer_type", "portal", "study_id", "model"]).reset_index(drop=True)


def _markdown(summary: pd.DataFrame, b: dict) -> str:
    lines = [
        f"Stratified {b['n_splits']}-fold cross-validation (seed {b['seed']}, feature_selection: "
        f"{b['feature_selection']}); mean ± SD (ddof={b['sd_ddof']}) over the {b['n_splits']} test folds.",
        "",
        "| Cancer type | Portal:study | Model | n folds | " + " | ".join(METRIC_NAMES[k] for k in METRICS) + " |",
        "|---|---|---|---|" + "---|" * len(METRICS),
    ]
    for _, r in summary.iterrows():
        cells = [f"{r[f'{k}_mean']:.3f} ± {r[f'{k}_sd']:.3f}" if pd.notna(r[f"{k}_mean"]) else "-"
                 for k in METRICS]
        lines.append(f"| {r.cancer_type} | {r.portal}:{r.study_id} | {r.model} | {r.n_folds} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ==========================
# Entry point
# ==========================
def run_benchmark():
    cfg = _load_cfg()
    b = _bench_cfg(cfg)
    hp = _mogonet_cfg(cfg)
    clin_cfg = (cfg.get("model_mogonet", {}) or {}).get("clinical_view") or {}
    clinical_features = (list(clin_cfg.get("features", ["AGE", "CLINICAL_STAGE", "RACE"]))
                         if clin_cfg.get("enabled", False) else None)
    out_root = Path(b["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    c_map = gai._normalize_study_map(cfg.get("cBio_studyId", {}))
    p_map = gai._normalize_study_map(cfg.get("pedcBio_studyId", {}))
    expr = cfg.get("expression_profiles", {}) or {}
    if b["feature_selection"] == "precomputed":
        print("[benchmark] NOTE feature_selection=precomputed: features were selected on all patients "
              "by data_process, so test folds took part in selection (optimistic estimates).")

    all_folds, settings = [], {"benchmark": b, "model_mogonet": dict(hp, dim_he_list=list(hp["dim_he_list"])),
                               "clinical_features": clinical_features, "processor_labels": {},
                               "sklearn": sklearn.__version__, "torch": gai.torch.__version__}
    for ct in cfg.get("cancer_type_interest", []) or []:
        mods = [m for m in gai._normalize_modalities(expr.get(ct, [])) if m in OMICS]
        if not mods:
            print(f"[benchmark] SKIP {ct}: no mrna/mirna/methylation in expression_profiles.")
            continue
        lc = _label_cfg(cfg, ct)
        settings["processor_labels"][ct] = lc
        for portal, study_id in gai._studies_for_ct(ct, c_map, p_map):
            tag = f"{ct} [{portal}:{study_id}]"
            X, y, clin = load_cohort(cfg, ct, portal, study_id, mods, b["feature_selection"])
            counts = y.value_counts()
            print(f"[benchmark] {tag}: {len(y)} patients, labels {counts.to_dict()}, views {mods}"
                  + (" + clinical" if clinical_features and clin is not None else ""))
            if counts.min() < b["n_splits"]:
                raise ValueError(f"[benchmark] {tag}: smallest class has {counts.min()} patients "
                                 f"< n_splits={b['n_splits']}")

            rows, splits = [], []
            seed = b["seed"]
            skf = StratifiedKFold(n_splits=b["n_splits"], shuffle=True, random_state=seed)
            for k, (i_tr, i_te) in enumerate(skf.split(np.zeros(len(y)), y.to_numpy()), start=1):
                tr, te = y.index[i_tr], y.index[i_te]
                splits += [{"patientId": p, "label": y[p], "test_fold": k} for p in te]
                try:
                    Xtr, Xte = fold_views(X, y, clin, tr, te, mods, b["feature_selection"], lc,
                                          clinical_features)
                except RuntimeError as e:
                    raise RuntimeError(f"[benchmark] {tag} fold={k}: {e}") from e
                views = list(Xtr)
                for name in b["models"]:
                    t0 = time.time()
                    if name == "mogonet":
                        sc = _run_mogonet(Xtr, Xte, y.loc[tr], y.loc[te], views, seed, hp)
                    else:
                        sc = _run_sklearn(name, Xtr, Xte, y.loc[tr], y.loc[te], views, seed, b)
                    dt = time.time() - t0
                    rows.append({"cancer_type": ct, "portal": portal, "study_id": study_id,
                                 "model": name, "fold": k, "n_train": len(tr),
                                 "n_test": len(te), "n_features": sum(Xtr[v].shape[1] for v in views),
                                 **sc, "seconds": round(dt, 2)})
                    print(f"[benchmark] {tag} fold={k}/{b['n_splits']} {name:<13} "
                          + " ".join(f"{m}={sc[m]:.3f}" for m in METRICS) + f" ({dt:.1f}s)")

            folds = pd.DataFrame(rows)
            out_dir = out_root / portal / gai._slug(ct) / study_id
            out_dir.mkdir(parents=True, exist_ok=True)
            folds.to_csv(out_dir / "cv_folds.csv", index=False)
            pd.DataFrame(splits).to_csv(out_dir / "cv_splits.csv", index=False)
            summarize(folds, b["models"], b["sd_ddof"]).to_csv(out_dir / "cv_summary.csv", index=False)
            all_folds.append(folds)
            print(f"[benchmark] {tag}: wrote {out_dir}/cv_folds.csv, cv_splits.csv, cv_summary.csv")

    if not all_folds:
        print("[benchmark] nothing to do (check cancer_type_interest / study IDs).")
        return
    summary = summarize(pd.concat(all_folds, ignore_index=True), b["models"], b["sd_ddof"])
    summary.to_csv(out_root / "cv_summary.csv", index=False)
    (out_root / "cv_summary.md").write_text(_markdown(summary, b))
    with open(out_root / "cv_settings.json", "w") as f:
        json.dump(settings, f, indent=2, default=str)
    print("\n" + _markdown(summary, b))
    print(f"[benchmark] wrote {out_root}/cv_summary.csv, cv_summary.md, cv_settings.json")


if __name__ == "__main__":
    run_benchmark()
