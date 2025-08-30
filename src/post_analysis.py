# src/post_analysis.py
"""
post_analysis.py — Downstream interpretability and survival analysis

Goals
-----
1) Provide interpretability of trained multi-omics GNNs by estimating feature
   importance via ablation.
   - MOGONET-style: rebuild adjacencies after zeroing out features to capture
     network-dependent importance (not just coefficient-based).
   - Returns per-modality DataFrames with importance scores per feature.
   - To be integrated with gene ranking, biomarker discovery, and GO enrichment.

2) Provide survival analysis plots and statistics for stratified patient cohorts.
   - Kaplan–Meier curves with log-rank tests for 2 groups, or multivariate log-rank
     test for >2 groups.
   - Cox proportional hazards regression as an option (not fully integrated yet).
   - Outputs publication-ready plots (PNG/PDF) with embedded p-values.

Inputs & Dependencies
---------------------
- Feature importance:
    * X_tr_list: list of np.ndarrays (one per omics view) for TRAIN samples.
    * X_trte_list: list of np.ndarrays (train+test stacked).
    * trte_idx: dict {"tr": [...], "te": [...]} marking index boundaries.
    * model_dict: trained GNN model dict (from graph_ai.py).
    * num_class: number of output classes.
- Survival analysis:
    * clinical_df: patient-level DataFrame with OS_MONTHS, OS_STATUS_Binary, and group_col.
    * group_col: stratification variable (e.g., risk_group).
    * surv_time: time-to-event column (e.g., OS_MONTHS).
    * surv_status: event indicator (e.g., OS_STATUS_Binary).
    * filename: output file path for KM curve figure.

Notes
-----
- Feature importance:
    * Current template is placeholder-style — ablation loop sets features to zero,
      rebuilds graphs, and measures drop in performance.
    * The actual importance metric should be tied to ACC/F1/AUC vs. baseline,
      with true labels provided by caller (not included yet).
    * Scales poorly if #features is very large — recommend top-k preselection upstream.

- Survival analysis:
    * Built on `lifelines` (KaplanMeierFitter, logrank_test, CoxPHFitter).
    * KM plots include p-value annotation inside the figure.
    * Saves plots directly (no plt.show()) for headless Docker runs.

Usage
-----
# Feature importance (after training with graph_ai.train_with_split):
feat_imps = feature_importance_ablation(
    X_tr_list, X_trte_list, trte_idx, model_dict, num_class=2, edges_per_node=5
)
for v, df_imp in enumerate(feat_imps):
    df_imp.to_csv(f"results/view{v+1}_feat_importance.csv", index=False)

# Survival analysis:
plot_km_curves(
    clinical_df=df,
    group_col="risk_group",
    surv_time="OS_MONTHS",
    surv_status="OS_STATUS_Binary",
    filename="results/km_risk_groups.png",
    title="Wilms Tumor Risk Groups"
)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib MUST be set to Agg before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import yaml

# ---- graph / GNN utilities reused from graph_ai and graph_constructor ----
from .graph_ai import (
    tensors_from_numpy_lists,
    test_epoch,
    load_model_bundle,
    MOGONETGraphParams,
    compute_adaptive_radius_parameter_mogonet,
    build_train_adjacency_mogonet,
    build_trte_block_adjacency_mogonet,
)

# =============================================================================
# Small config helpers
# =============================================================================

def _cfg() -> dict:
    cfg_path = os.getenv("CONFIG_PATH", "config/pipeline_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

def _slug(s: str) -> str:
    return str(s).replace(" ", "_")

def _mods_for(ct: str, cfg: dict) -> List[str]:
    prof = (cfg.get("expression_profiles") or {}).get(ct, []) or []
    # normalize synonyms
    syn = {"microrna": "mirna"}
    return [syn.get(m.lower(), m.lower()) for m in prof if m.lower() in {"mrna","microrna","mirna","methylation"}]

def _coerce_int_labels(y) -> np.ndarray:
    """Return 1D int numpy array from any series/list of labels ('0'/'1', 0/1, etc.)."""
    import pandas as pd
    s = pd.Series(y)
    # common cases: strings "0"/"1", booleans, categorical
    try:
        return s.astype(int).to_numpy()
    except Exception:
        # map known string values
        m = {"0": 0, "1": 1, "false": 0, "true": 1}
        return s.astype(str).str.lower().map(m).astype(int).to_numpy()

# =============================================================================
# Survival analysis (KM)
# =============================================================================

def plot_km_curves(clinical_df: pd.DataFrame, group_col: str, surv_time: str, surv_status: str, filename: str, title: str):
    """
    Plot Kaplan–Meier curves stratified by group_col and save to 'filename'.
    clinical_df columns: surv_time (e.g., OS_MONTHS), surv_status (0/1), group_col.
    """
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))
    groups = clinical_df[group_col].dropna().unique()

    for grp in groups:
        ix = clinical_df[group_col] == grp
        kmf.fit(
            durations=clinical_df.loc[ix, surv_time],
            event_observed=clinical_df.loc[ix, surv_status],
            label=str(grp),
        )
        kmf.plot_survival_function()

    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_text = x_min + 0.1 * (x_max - x_min)
    y_text = y_min + 0.1 * (y_max - y_min)

    if len(groups) == 2:
        g1 = clinical_df[clinical_df[group_col] == groups[0]]
        g2 = clinical_df[clinical_df[group_col] == groups[1]]
        res = logrank_test(g1[surv_time], g2[surv_time], event_observed_A=g1[surv_status], event_observed_B=g2[surv_status])
        ax.text(x_text, y_text, f"Log-Rank p = {res.p_value:.4f}", fontsize=11, bbox=dict(facecolor="white", alpha=0.6))
    elif len(groups) > 2:
        res = multivariate_logrank_test(clinical_df[surv_time], clinical_df[group_col], clinical_df[surv_status])
        ax.text(x_text, y_text, f"Multivariate Log-Rank p = {res.p_value:.4f}", fontsize=11, bbox=dict(facecolor="white", alpha=0.6))

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"[post] Saved KM plot -> {filename}")

def _derive_os_binary(df: pd.DataFrame, status_col: str = "OS_STATUS") -> pd.Series:
    """
    cBio OS_STATUS often looks like: '1:DECEASED' or '0:LIVING'.
    Return 1 for deceased, 0 otherwise.
    """
    if status_col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float")
    s = df[status_col].astype(str).str.strip().str.upper()
    out = pd.Series(0, index=df.index, dtype="int64")
    out = out.mask(s.str.startswith("1"), 1)
    out = out.mask(s.str.contains("DECEASED"), 1)
    out = out.fillna(0)
    return out.astype(int)

def run_km_survival():
    cfg = _cfg()
    pa = (cfg.get("post_analysis") or {}).get("survival") or {}
    if not pa or not pa.get("enabled", False):
        print("[post] KM survival disabled or not configured.")
        return

    group_col = pa.get("group_col")
    time_col  = pa.get("time_col", "OS_MONTHS")
    status_c  = pa.get("status_col", "OS_STATUS")
    title     = pa.get("title", "Survival")
    out_dir   = Path(pa.get("out_dir", "data/analysis/km"))

    if not group_col:
        print("[post] KM survival requires 'group_col' in config.post_analysis.survival.")
        return

    cts = cfg.get("cancer_type_interest", [])
    for ct in cts:
        ct_slug = _slug(ct)

        # 1) Prefer filtered clinical (restricted to used patients)
        filt_dir = Path("data/gdm/filtered") / ct_slug
        cand = [filt_dir / "clinical_filtered.parquet", filt_dir / "clinical_filtered.csv"]

        pat = None
        for p in cand:
            if p.exists():
                pat = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                break

        # 2) Fallback to original cBio patients
        if pat is None:
            pat_csv = Path("data/cbioportal_api_request") / ct_slug / "clinical_patients.csv"
            if pat_csv.exists():
                pat = pd.read_csv(pat_csv)
            else:
                gdm_sub = Path("data/gdm/clinical/clinical_subjects.parquet")
                if not gdm_sub.exists():
                    print(f"[post] No clinical table for {ct}; skipping.")
                    continue
                pat = pd.read_parquet(gdm_sub)

        # Now proceed with configured columns
        if time_col not in pat.columns:
            print(f"[post] {ct}: time_col '{time_col}' not found; skipping.")
            continue
        if group_col not in pat.columns:
            print(f"[post] {ct}: group_col '{group_col}' not found; skipping.")
            continue

        df = pat.copy()
        df["__status_bin"] = _derive_os_binary(df, status_col=status_c)
        ix = df[time_col].notna() & df["__status_bin"].notna()
        df = df.loc[ix, [group_col, time_col, "__status_bin"]].copy()
        if df.empty:
            print(f"[post] {ct}: no rows with survival info; skipping.")
            continue

        out_png = out_dir / f"{ct_slug}_km.png"
        plot_km_curves(df, group_col, time_col, "__status_bin", str(out_png), f"{title} — {ct}")

# =============================================================================
# Feature-importance ablation (per-modality)
# =============================================================================

def _load_proc_for_ct(ct: str, mods: Sequence[str]):
    """
    Load processed train/test feature matrices + labels for a cancer type.
    Ensures numeric, aligned sample order across modalities, and matches labels.
    """
    root = Path("data/gdm/filtered") / _slug(ct)
    X_tr, X_te = {}, {}
    for m in mods:
        p_tr = root / f"{m}_train.parquet"
        p_te = root / f"{m}_test.parquet"
        if not p_tr.exists() or not p_te.exists():
            raise FileNotFoundError(f"[post] Missing {p_tr} or {p_te}. Run 'process' first for {ct}/{m}.")
        X_tr[m] = pd.read_parquet(p_tr)
        X_te[m] = pd.read_parquet(p_te)

    # labels
    y_tr = pd.read_csv(root / "labels_train.csv", index_col=0, header=None).iloc[:, 0]
    y_te = pd.read_csv(root / "labels_test.csv", index_col=0, header=None).iloc[:, 0]

    # choose a reference view and align sample order (rows)
    ref = mods[0]
    X_tr[ref] = X_tr[ref].loc[y_tr.index.intersection(X_tr[ref].index)]
    X_te[ref] = X_te[ref].loc[y_te.index.intersection(X_te[ref].index)]
    y_tr = y_tr.loc[X_tr[ref].index]
    y_te = y_te.loc[X_te[ref].index]
    for m in mods[1:]:
        X_tr[m] = X_tr[m].loc[y_tr.index]
        X_te[m] = X_te[m].loc[y_te.index]

    # ensure numeric dtypes
    for m in mods:
        X_tr[m] = X_tr[m].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_te[m] = X_te[m].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X_tr, X_te, y_tr, y_te

def _expected_in_dims(model_dict, num_view: int):
    # Each encoder E{i} has gc1.weight: [in_dim, hidden]
    return [int(model_dict[f"E{i+1}"].gc1.weight.shape[0]) for i in range(num_view)]

def _order_mods_like_meta(mods: Sequence[str], meta: Dict[str, any]) -> List[str]:
    """Return modalities in the exact order used at training (meta['views'])."""
    meta_views = list(meta.get("views") or [])
    if not meta_views:
        return list(mods)  # fall back to given order
    # Keep only the views present in both lists, preserve meta order
    ordered = [v for v in meta_views if v in set(mods)]
    # Warn if mismatch
    if set(ordered) != set(mods):
        missing = [m for m in mods if m not in ordered]
        if missing:
            print(f"[post] Warning: these configured views were not in the checkpoint: {missing}")
    return ordered

def _align_to_meta_features(
    X_tr: Dict[str, pd.DataFrame],
    X_te: Dict[str, pd.DataFrame],
    mods: Sequence[str],
    meta: Dict[str, any],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    fnames = (meta or {}).get("feature_names")
    if not fnames:
        raise RuntimeError(
            "Checkpoint meta.json is missing 'feature_names'. Retrain the model so "
            "meta['feature_names'] is saved, or ensure post-analysis uses the exact "
            "same filtered feature set as training."
        )

    X_tr_aligned, X_te_aligned = {}, {}
    for m in mods:
        if m not in fnames:
            raise RuntimeError(f"meta['feature_names'] has no entry for view '{m}'. Retrain to include it.")
        cols = [str(c) for c in fnames[m]]

        # add any missing columns as zeros
        miss_tr = [c for c in cols if c not in X_tr[m].columns]
        miss_te = [c for c in cols if c not in X_te[m].columns]
        if miss_tr:
            X_tr[m] = X_tr[m].join(pd.DataFrame(0.0, index=X_tr[m].index, columns=miss_tr))
        if miss_te:
            X_te[m] = X_te[m].join(pd.DataFrame(0.0, index=X_te[m].index, columns=miss_te))

        # drop extras and reorder exactly
        X_tr_aligned[m] = (
            X_tr[m].loc[:, cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
        X_te_aligned[m] = (
            X_te[m].loc[:, cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
    return X_tr_aligned, X_te_aligned

def _validate_dims_against_model(model_dict, mods: Sequence[str], X_tr: Dict[str, pd.DataFrame]):
    exp = _expected_in_dims(model_dict, len(mods))
    got = [X_tr[m].shape[1] for m in mods]
    if exp != got:
        # rich diagnostics to help debugging quickly
        details = "\n".join([f"  - {m}: got {X_tr[m].shape[1]} cols" for m in mods])
        raise RuntimeError(
            "Feature dimension mismatch after alignment.\n"
            f"Expected per-view input dims: {exp}\n"
            f"Got per-view feature dims:   {got}\n{details}\n"
            "Make sure you're using the same processed features as training, and that "
            "'feature_names' in the checkpoint matches these files."
        )

def _baseline_probs(
    model_dict,
    mods: Sequence[str],
    X_tr: Dict[str, pd.DataFrame],
    X_te: Dict[str, pd.DataFrame],
    edges_per_node: int,
    meta: Dict,    # pass the checkpoint meta
):
    # 1) make sure modality order matches training
    mods = _order_mods_like_meta(mods, meta)

    # 2) align columns to training feature names and validate dims
    X_tr, X_te = _align_to_meta_features(X_tr, X_te, mods, meta)
    _validate_dims_against_model(model_dict, mods, X_tr)

    # 3) build TRTE block adjacencies and run inference
    params = MOGONETGraphParams(edges_per_node=edges_per_node)
    X_tr_list, X_trte_list, A_trte_list = [], [], []
    for m in mods:
        Xtr = X_tr[m].to_numpy(dtype=np.float32)
        Xte = X_te[m].to_numpy(dtype=np.float32)
        radius = compute_adaptive_radius_parameter_mogonet(Xtr, params)
        A_trte = build_trte_block_adjacency_mogonet(Xtr, Xte, radius=radius, params=params)
        X_tr_list.append(Xtr)
        X_trte_list.append(np.vstack([Xtr, Xte]))
        A_trte_list.append(A_trte)

    data_trte, adj_trte = tensors_from_numpy_lists(X_trte_list, A_trte_list, device="cpu")
    te_idx = list(range(len(X_tr[mods[0]]), len(X_tr[mods[0]]) + len(X_te[mods[0]])))
    prob = test_epoch(data_trte, adj_trte, te_idx, model_dict)

    # helpful diagnostics
    print("[post] Baseline dims per view (samples_tr, samples_te, nfeat):")
    for i, m in enumerate(mods):
        print(f"  - {m}: ({X_tr[m].shape[0]}, {X_te[m].shape[0]}, {X_tr[m].shape[1]})")
    return prob, te_idx, X_tr, X_te, mods

def run_feature_ablation():
    cfg = _cfg()
    ab = (cfg.get("post_analysis") or {}).get("ablation") or {}
    if not ab or not ab.get("enabled", False):
        print("[post] Feature ablation disabled or not configured.")
        return

    edges_per_node = int(ab.get("edges_per_node", 5))
    out_dir = Path(ab.get("out_dir", "data/analysis/ablation"))

    cts = cfg.get("cancer_type_interest", [])
    for ct in cts:
        mods_cfg = _mods_for(ct, cfg)
        if not mods_cfg:
            print(f"[post] {ct}: no modalities configured for ablation; skipping.")
            continue

        # load processed data + labels
        X_tr, X_te, y_tr, y_te = _load_proc_for_ct(ct, mods_cfg)
        y_true = _coerce_int_labels(y_te.values)

        # load trained model bundle (+ meta)
        ckpt_dir = Path(cfg.get("model", {}).get("checkpoints_dir", "data/models")) / _slug(ct)
        if not (ckpt_dir / "meta.json").exists():
            print(f"[post] {ct}: checkpoints not found at {ckpt_dir}; skipping ablation.")
            continue
        model_dict, meta = load_model_bundle(ckpt_dir)

        # baseline (also returns column-aligned matrices and the view order used)
        base_prob, te_idx, X_tr, X_te, mods = _baseline_probs(
            model_dict, mods_cfg, X_tr, X_te, edges_per_node, meta
        )
        base_pred = base_prob.argmax(1)
        base_f1  = f1_score(y_true, base_pred, average="macro")
        base_acc = accuracy_score(y_true, base_pred)
        base_auc = None
        if base_prob.shape[1] == 2:
            try:
                base_auc = roc_auc_score(y_true, base_prob[:, 1])
            except Exception:
                pass
        print(f"[post] {ct}: baseline — acc={base_acc:.4f}, f1={base_f1:.4f}{'' if base_auc is None else f', auc={base_auc:.4f}'}")

        # per-modality ablation (use the ALIGNED matrices)
        params = MOGONETGraphParams(edges_per_node=edges_per_node)
        for m in mods:
            Xt = X_tr[m].to_numpy(dtype=np.float32)
            Xe = X_te[m].to_numpy(dtype=np.float32)
            V = Xt.shape[1]
            imps = np.zeros(V, dtype=float)

            # precompute other views once
            other_views = [mm for mm in mods if mm != m]
            A_trte_others, X_trte_list_others = [], []
            for om in other_views:
                Xtr_o = X_tr[om].to_numpy(dtype=np.float32)
                Xte_o = X_te[om].to_numpy(dtype=np.float32)
                rad_o = compute_adaptive_radius_parameter_mogonet(Xtr_o, params)
                A_trte_o = build_trte_block_adjacency_mogonet(Xtr_o, Xte_o, radius=rad_o, params=params)
                A_trte_others.append(A_trte_o)
                X_trte_list_others.append(np.vstack([Xtr_o, Xte_o]))

            for j in range(V):
                # ablate one feature in this view
                Xtr_v = Xt.copy(); Xte_v = Xe.copy()
                Xtr_v[:, j] = 0.0; Xte_v[:, j] = 0.0
                rad_v = compute_adaptive_radius_parameter_mogonet(Xtr_v, params)
                A_trte_v = build_trte_block_adjacency_mogonet(Xtr_v, Xte_v, radius=rad_v, params=params)

                # pack views: others unchanged, current view ablated
                X_trte_all = X_trte_list_others + [np.vstack([Xtr_v, Xte_v])]
                A_trte_all = A_trte_others + [A_trte_v]
                data_trte, adj_trte = tensors_from_numpy_lists(X_trte_all, A_trte_all, device="cpu")
                prob = test_epoch(data_trte, adj_trte, te_idx, model_dict)
                f1 = f1_score(y_true, prob.argmax(1), average="macro")
                imps[j] = float(max(0.0, base_f1 - f1))

            out_df = pd.DataFrame({"feature": X_tr[m].columns.astype(str), "delta_f1": imps})
            out_df.sort_values("delta_f1", ascending=False, inplace=True)
            out_path = out_dir / _slug(ct) / f"{m}_ablation.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_path, index=False)
            print(f"[post] {ct}/{m}: wrote importance CSV -> {out_path}")

# =============================================================================
# Orchestrator
# =============================================================================

def run_post_analysis():
    """
    Decide which post tasks to run based on config.post_analysis.*
    """
    cfg = _cfg()
    sec = cfg.get("post_analysis") or {}
    did = False

    if (sec.get("survival") or {}).get("enabled", False):
        run_km_survival()
        did = True
    if (sec.get("ablation") or {}).get("enabled", False):
        run_feature_ablation()
        did = True

    if not did:
        print("[post] Nothing to do. Configure post_analysis.survival/ablation in your YAML.")