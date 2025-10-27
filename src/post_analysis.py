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
- Feature importance: make feature selection use trained views (and load clinical), but ablate omics only.
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
feat_imps = feature_importance_selection(
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
)
from .graph_constructor import(
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
    """Legacy helper (kept for survival section)."""
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


def _study_map_from_cfg(cfg: dict) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns a normalized mapping:
      { 'cbio': {'Wilms Tumor': ['wt_target_2018_pub', ...], ...},
        'pedcbio': {'Glioma': ['pbta_all', ...], ...} }
    Supports both the new cfg.study_ids.{cbio|pedcbio} and legacy keys.
    """
    out = {"cbio": {}, "pedcbio": {}}
    study_ids = (cfg.get("study_ids") or {})
    for portal in ("cbio", "pedcbio"):
        m = study_ids.get(portal) or {}
        for ct, val in (m.items() if isinstance(m, dict) else []):
            out[portal][ct] = val if isinstance(val, list) else [val]

    # legacy compatibility
    legacy_cbio = cfg.get("cBio_studyId") or {}
    legacy_ped = cfg.get("pedcBio_studyId") or {}
    for ct, val in legacy_cbio.items():
        out["cbio"].setdefault(ct, val if isinstance(val, list) else [val])
    for ct, val in legacy_ped.items():
        out["pedcbio"].setdefault(ct, val if isinstance(val, list) else [val])

    return out
    
def _iter_studies(cfg: dict):
    """
    Yields (portal, cancer_type, study_id) for all configured studies.
    """
    m = _study_map_from_cfg(cfg)
    for portal in ("cbio", "pedcbio"):
        for ct, ids in m.get(portal, {}).items():
            for sid in ids:
                yield portal, ct, sid

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
    out_root  = Path(pa.get("out_dir", "data/analysis/km"))

    if not group_col:
        print("[post] KM survival requires 'group_col' in config.post_analysis.survival.")
        return

    # iterate per study (portal/ct/study_id)
    for portal, ct, study_id in _iter_studies(cfg):
        ct_slug = _slug(ct)
        filt_dir = Path("data/gdm/filtered") / portal / ct_slug / study_id

        pat = None
        # Prefer filtered clinical for the exact split
        for cand in (filt_dir / "clinical_filtered.parquet", filt_dir / "clinical_filtered.csv"):
            if cand.exists():
                pat = pd.read_parquet(cand) if cand.suffix == ".parquet" else pd.read_csv(cand)
                break

        # Fallbacks if needed (portal-aware)
        if pat is None:
            cbio_pat = Path("data/fetch/cbioportal_api_request") / portal / ct_slug / study_id / "clinical_patients.csv"
            if cbio_pat.exists():
                pat = pd.read_csv(cbio_pat)
            else:
                gdm_sub = Path("data/gdm/clinical/clinical_subjects.parquet")
                if not gdm_sub.exists():
                    print(f"[post] No clinical table for {portal}/{ct}/{study_id}; skipping.")
                    continue
                pat = pd.read_parquet(gdm_sub)

        if time_col not in pat.columns:
            print(f"[post] {portal}/{ct}/{study_id}: time_col '{time_col}' not found; skipping.")
            continue
        if group_col not in pat.columns:
            print(f"[post] {portal}/{ct}/{study_id}: group_col '{group_col}' not found; skipping.")
            continue

        df = pat.copy()
        df["__status_bin"] = _derive_os_binary(df, status_col=status_c)
        ix = df[time_col].notna() & df["__status_bin"].notna()
        df = df.loc[ix, [group_col, time_col, "__status_bin"]].copy()
        if df.empty:
            print(f"[post] {portal}/{ct}/{study_id}: no rows with survival info; skipping.")
            continue

        out_png = out_root / portal / ct_slug / study_id / f"km.png"
        plot_km_curves(df, group_col, time_col, "__status_bin", str(out_png), f"{title} — {ct} [{study_id}]")


# =============================================================================
# Feature-importance ablation (feature selection per-modality)
# =============================================================================

def _study_for_ct(cfg: dict, ct: str) -> Tuple[str, str]:
    """
    Resolve (portal, study_id) for a cancer type using config maps.
    Priority: cBio_studyId -> pedcBio_studyId. If not found, try to discover on disk.
    """
    maps = [
        ("cBio_studyId", "cbio"),
        ("pedcBio_studyId", "pedcbio"),
    ]
    for key, portal in maps:
        sid_map = cfg.get(key) or {}
        if ct in sid_map:
            return portal, sid_map[ct]

    # Fallback: discover any existing processed directory
    ct_slug = _slug(ct)
    base = Path("data/gdm/filtered")
    for portal in ("cbio", "pedcbio"):
        p = base / portal / ct_slug
        if p.exists():
            # pick the first subfolder as study_id
            for child in sorted(p.iterdir()):
                if child.is_dir():
                    return portal, child.name
    # Last resort
    return "cbio", "unknown_study"


def _load_proc_for_study(portal: str, ct: str, study_id: str, mods: Sequence[str]):
    """
    Load processed train/test feature matrices + labels for a cancer type/study/portal.
    Supports any view in 'mods', including 'clinical'.
    """
    root = Path("data/gdm/filtered") / portal / _slug(ct) / study_id
    X_tr, X_te = {}, {}

    for m in mods:
        p_tr = root / f"{m}_train.parquet"
        p_te = root / f"{m}_test.parquet"
        if not p_tr.exists() or not p_te.exists():
            raise FileNotFoundError(
                f"[post] Missing {p_tr} or {p_te}. "
                f"If '{m}' was used during training (meta['views']), it must exist for post-analysis. "
                f"Re-run processor/graph_ai to materialize {m} matrices at {root}."
            )
        X_tr[m] = pd.read_parquet(p_tr)
        X_te[m] = pd.read_parquet(p_te)

    # labels
    lt = root / "labels_train.csv"
    le = root / "labels_test.csv"
    if not lt.exists() or not le.exists():
        raise FileNotFoundError(f"[post] Missing labels CSVs at {root}.")
    y_tr = pd.read_csv(lt, index_col=0, header=None).iloc[:, 0]
    y_te = pd.read_csv(le, index_col=0, header=None).iloc[:, 0]

    # align all views to label indices using the first trained view as reference
    ref = mods[0]
    X_tr[ref] = X_tr[ref].loc[y_tr.index.intersection(X_tr[ref].index)]
    X_te[ref] = X_te[ref].loc[y_te.index.intersection(X_te[ref].index)]
    y_tr = y_tr.loc[X_tr[ref].index]
    y_te = y_te.loc[X_te[ref].index]
    for m in mods[1:]:
        X_tr[m] = X_tr[m].loc[y_tr.index]
        X_te[m] = X_te[m].loc[y_te.index]

    # ensure numeric dtypes across all views (clinical included)
    for m in mods:
        X_tr[m] = (
            X_tr[m].apply(pd.to_numeric, errors="coerce")
                   .replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        X_te[m] = (
            X_te[m].apply(pd.to_numeric, errors="coerce")
                   .replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    return X_tr, X_te, y_tr, y_te, root


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
            f"Got per-view feature dims: {got}\n{details}\n"
            "Make sure you're using the same processed features as training, and that "
            "'feature_names' in the checkpoint matches these files."
        )


def _reorder_to_trained_order(X_tr: Dict[str, pd.DataFrame],
                              X_te: Dict[str, pd.DataFrame],
                              mods_trained: Sequence[str]) -> Tuple[Dict[str,pd.DataFrame], Dict[str,pd.DataFrame]]:
    """Return new dicts whose keys/iteration order match mods_trained exactly."""
    X_tr_ord = {m: X_tr[m] for m in mods_trained}
    X_te_ord = {m: X_te[m] for m in mods_trained}
    return X_tr_ord, X_te_ord


def _baseline_probs(
    model_dict,
    mods: Sequence[str],
    X_tr: Dict[str,pd.DataFrame],
    X_te: Dict[str,pd.DataFrame],
    edges_per_node: int,
    meta: Dict
):
    """
    Align frames to training feature set and produce baseline TRTE probabilities
    using the exact trained views (including 'clinical' if present).
    Returns: prob, te_idx, X_tr_aln, X_te_aln, params
    """
    # align features to training feature set
    X_tr_aln, X_te_aln = _align_to_meta_features(X_tr, X_te, mods, meta)
    # check dims against encoders
    _validate_dims_against_model(model_dict, mods, X_tr_aln)

    params = MOGONETGraphParams(edges_per_node=edges_per_node)
    X_trte_list, A_trte_list = [], []
    for m in mods:
        Xtr = X_tr_aln[m].to_numpy(dtype=np.float32)
        Xte = X_te_aln[m].to_numpy(dtype=np.float32)
        radius = compute_adaptive_radius_parameter_mogonet(Xtr, params)
        A_trte = build_trte_block_adjacency_mogonet(Xtr, Xte, radius=radius, params=params)
        X_trte_list.append(np.vstack([Xtr, Xte]))
        A_trte_list.append(A_trte)

    data_trte, adj_trte = tensors_from_numpy_lists(X_trte_list, A_trte_list, device="cpu")
    te_idx = list(range(len(X_tr_aln[mods[0]]), len(X_tr_aln[mods[0]]) + len(X_te_aln[mods[0]])))
    prob = test_epoch(data_trte, adj_trte, te_idx, model_dict)
    return prob, te_idx, X_tr_aln, X_te_aln, params


def run_feature_selection():
    cfg = _cfg()
    ab = (cfg.get("post_analysis") or {}).get("feature_selection") or {}
    if not ab or not ab.get("enabled", False):
        print("[post] Feature selection disabled or not configured.")
        return

    edges_per_node = int(ab.get("edges_per_node", 5))
    out_root = Path(ab.get("out_dir", "data/analysis/feature_selection"))

    cts = cfg.get("cancer_type_interest", [])
    for ct in cts:
        ct_slug = _slug(ct)

        # load trained model bundle + metadata
        ckpt_dir = Path((cfg.get("model") or {}).get("checkpoints_dir", "data/models")) / ct_slug
        if not (ckpt_dir / "meta.json").exists():
            print(f"[post] {ct}: checkpoints not found at {ckpt_dir}; skipping selection.")
            continue
        model_dict, meta = load_model_bundle(ckpt_dir)

        # ---- USE TRAINED VIEW ORDER STRICTLY ----
        trained_views = list(meta.get("views") or [])
        if not trained_views:
            raise RuntimeError("Checkpoint meta.json missing 'views'. Retrain to save 'views' in meta.")
        mods_to_load = trained_views

        # Resolve (portal, study_id) for this cancer type (so we can find processed matrices)
        portal, study_id = _study_for_ct(cfg, ct)
        print(f"[post] {ct}: portal={portal} study_id={study_id} trained_views={mods_to_load}")

        # load processed data + labels using the trained view set
        try:
            X_tr, X_te, y_tr, y_te, filt_root = _load_proc_for_study(portal, ct, study_id, mods_to_load)
        except FileNotFoundError as e:
            print(str(e))
            continue

        y_true = _coerce_int_labels(y_te.values)

        # ---- REORDER CURRENT FRAMES EXACTLY AS TRAINED (keys/iteration order) ----
        X_tr, X_te = _reorder_to_trained_order(X_tr, X_te, mods_to_load)

        print("[post] Trained order:", mods_to_load)
        # baseline (all trained views) — ensures VCDN dims match
        base_prob, te_idx, X_tr_aln, X_te_aln, params = _baseline_probs(
            model_dict, mods_to_load, X_tr, X_te, edges_per_node, meta
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
        print(f"[post] {ct}: baseline acc={base_acc:.4f}, f1={base_f1:.4f}{'' if base_auc is None else f', auc={base_auc:.4f}'}")

        # ----- PRECOMPUTE TRTE BLOCKS FOR ALL TRAINED VIEWS (baseline, non-ablated) -----
        base_X_trte = {}
        base_A_trte = {}
        for m in mods_to_load:
            Xtr = X_tr_aln[m].to_numpy(dtype=np.float32)
            Xte = X_te_aln[m].to_numpy(dtype=np.float32)
            rad = compute_adaptive_radius_parameter_mogonet(Xtr, params)
            base_X_trte[m] = np.vstack([Xtr, Xte])
            base_A_trte[m] = build_trte_block_adjacency_mogonet(Xtr, Xte, radius=rad, params=params)

        # ----- per-modality ablation (KEEP TRAINED ORDER); omics only -----
        OMICS = {"mrna", "mirna", "methylation"}
        out_dir = out_root / portal / ct_slug / study_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for m in mods_to_load:
            if m not in OMICS:
                print(f"[post] {portal}/{ct}/{study_id}/{m}: skip importance (non-omics).")
                continue

            # aligned matrices
            Xt = X_tr_aln[m].to_numpy(dtype=np.float32)
            Xe = X_te_aln[m].to_numpy(dtype=np.float32)
            V = Xt.shape[1]
            imps = np.zeros(V, dtype=float)

            for j in range(V):
                # ablate one feature
                Xtr_v = Xt.copy(); Xte_v = Xe.copy()
                Xtr_v[:, j] = 0.0; Xte_v[:, j] = 0.0
                rad_v = compute_adaptive_radius_parameter_mogonet(Xtr_v, params)
                A_trte_v = build_trte_block_adjacency_mogonet(Xtr_v, Xte_v, radius=rad_v, params=params)
                X_trte_v = np.vstack([Xtr_v, Xte_v])

                # assemble TRTE lists STRICTLY in trained order
                X_trte_all = []
                A_trte_all = []
                for mm in mods_to_load:
                    if mm == m:
                        X_trte_all.append(X_trte_v)
                        A_trte_all.append(A_trte_v)
                    else:
                        X_trte_all.append(base_X_trte[mm])
                        A_trte_all.append(base_A_trte[mm])

                data_trte, adj_trte = tensors_from_numpy_lists(X_trte_all, A_trte_all, device="cpu")
                prob = test_epoch(data_trte, adj_trte, te_idx, model_dict)

                f1 = f1_score(y_true, prob.argmax(1), average="macro")
                imps[j] = float((base_f1 - f1) * V)

            out_df = pd.DataFrame({"feature": X_tr_aln[m].columns.astype(str), "delta_f1": imps})
            out_df.sort_values("delta_f1", ascending=False, inplace=True)
            out_path = out_dir / f"{m}_selection.csv"
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
    if (sec.get("feature_selection") or {}).get("enabled", False):
        run_feature_selection()
        did = True

    if not did:
        print("[post] Nothing to do. Configure post_analysis.survival/feature_selection in your YAML.")