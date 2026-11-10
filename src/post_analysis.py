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
from .study_scope import is_modeled, modeling_studies
from .cohort import class_names, load_labels
from .graph_ai import (
    tensors_from_numpy_lists,
    test_epoch,
    load_model_bundle,
    graph_view_names,
    _build_clinical_view,
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
    # same folder names as extractor / processor / graph_ai ("Wilms Tumor" -> "wilms_tumor")
    return str(s).replace(" ", "_").lower()


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
                if is_modeled(cfg, ct, portal, sid):          # modeling_studyId, if set
                    yield portal, ct, sid

# =============================================================================
# Survival analysis (KM)
# =============================================================================

def _p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_km_curves(clinical_df: pd.DataFrame, group_col: str, surv_time: str, surv_status: str, filename: str,
                   title: str, order: Optional[Sequence] = None, xlabel: Optional[str] = None) -> pd.DataFrame:
    """
    Plot Kaplan–Meier curves stratified by group_col and save to 'filename'.
    clinical_df columns: surv_time (e.g., OS_MONTHS), surv_status (0/1), group_col.
    order: groups in plotting (and color) order; default sorted. Returns one row per group
    (n, events, median survival) with the log-rank p-value.
    """
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))
    present = list(clinical_df[group_col].dropna().unique())
    groups = [g for g in order if g in present] if order is not None else sorted(present, key=str)

    stats = []
    for grp in groups:
        ix = clinical_df[group_col] == grp
        kmf.fit(
            durations=clinical_df.loc[ix, surv_time],
            event_observed=clinical_df.loc[ix, surv_status],
            label=str(grp),
        )
        kmf.plot_survival_function()
        stats.append({"group": grp, "n": int(ix.sum()), "events": int(clinical_df.loc[ix, surv_status].sum()),
                      "median_survival": float(kmf.median_survival_time_)})

    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_text = x_min + 0.1 * (x_max - x_min)
    y_text = y_min + 0.1 * (y_max - y_min)

    p = np.nan
    if len(groups) == 2:
        g1 = clinical_df[clinical_df[group_col] == groups[0]]
        g2 = clinical_df[clinical_df[group_col] == groups[1]]
        res = logrank_test(g1[surv_time], g2[surv_time], event_observed_A=g1[surv_status], event_observed_B=g2[surv_status])
        p = float(res.p_value)
        ax.text(x_text, y_text, f"Log-rank {_p_text(p)}", fontsize=11, bbox=dict(facecolor="white", alpha=0.6))
    elif len(groups) > 2:
        res = multivariate_logrank_test(clinical_df[surv_time], clinical_df[group_col], clinical_df[surv_status])
        p = float(res.p_value)
        ax.text(x_text, y_text, f"Multivariate log-rank {_p_text(p)}", fontsize=11, bbox=dict(facecolor="white", alpha=0.6))

    plt.title(title)
    plt.xlabel(xlabel or "Time")
    plt.ylabel("Survival Probability")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"[post] Saved KM plot -> {filename}")
    out = pd.DataFrame(stats)
    out["logrank_p"] = p
    return out


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


def _for_type(value, ct: str, default=None):
    """A survival setting given once for every cancer type, or per type as {type: value}."""
    if isinstance(value, dict):
        return value.get(ct, default)
    return default if value is None else value


def run_km_survival():
    """
    KM curves per modeled study (post_analysis.survival):
      group_col  clinical column, or "label" (the class modeled by data_process, named by
                 processor.labels.<type>.class_names); one value, or one per cancer type
      subset     all (train + test, default), train or test (the split of data_process)
      time_col, status_col, title, out_dir
    Writes km.png and km_stats.tsv (n, events, median survival per group; log-rank p).
    """
    cfg = _cfg()
    pa = (cfg.get("post_analysis") or {}).get("survival") or {}
    if not pa or not pa.get("enabled", False):
        print("[post] KM survival disabled or not configured.")
        return

    time_col  = pa.get("time_col", "OS_MONTHS")
    status_c  = pa.get("status_col", "OS_STATUS")
    out_root  = Path(pa.get("out_dir", "data/analysis/km"))

    if not pa.get("group_col"):
        print("[post] KM survival requires 'group_col' in config.post_analysis.survival.")
        return

    # iterate per study (portal/ct/study_id)
    for portal, ct, study_id in _iter_studies(cfg):
        ct_slug = _slug(ct)
        group_col = _for_type(pa.get("group_col"), ct)
        subset = str(_for_type(pa.get("subset"), ct, "all")).lower()
        title = _for_type(pa.get("title"), ct, "Survival")
        if not group_col:
            print(f"[post] {portal}/{ct}/{study_id}: no group_col for this cancer type; skipping KM.")
            continue
        if subset not in ("all", "train", "test"):
            print(f"[post] {portal}/{ct}/{study_id}: subset must be all, train or test (got {subset!r}); skipping KM.")
            continue
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
        if "patientId" not in pat.columns and "subject_id" in pat.columns:
            pat = pat.rename(columns={"subject_id": "patientId"})

        order = None
        if group_col == "label" or subset != "all":
            try:
                lab = load_labels(portal, ct, study_id)
            except FileNotFoundError as e:
                print(f"[post] {portal}/{ct}/{study_id}: {e}; skipping KM.")
                continue
            names = class_names(cfg, ct)
            lab = lab.rename(columns={"label": "__label", "split": "__split"})
            lab["__class"] = lab["__label"].map(lambda k: names.get(int(k), f"label {k}"))
            pat = pat.assign(patientId=pat["patientId"].astype(str).str.strip()).merge(lab, on="patientId", how="inner")
            if subset != "all":
                pat = pat[pat["__split"] == subset]
            if group_col == "label":
                group_col = "__class"
                order = [names.get(int(k), f"label {k}") for k in sorted(lab["__label"].unique())]

        if time_col not in pat.columns:
            print(f"[post] {portal}/{ct}/{study_id}: time_col '{time_col}' not found; skipping.")
            continue
        if group_col not in pat.columns:
            print(f"[post] {portal}/{ct}/{study_id}: group_col '{group_col}' not found; skipping.")
            continue

        df = pat.copy()
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df["__status_bin"] = _derive_os_binary(df, status_col=status_c)
        ix = df[time_col].notna() & df["__status_bin"].notna()
        df = df.loc[ix, [group_col, time_col, "__status_bin"]].copy()
        if df.empty:
            print(f"[post] {portal}/{ct}/{study_id}: no rows with survival info; skipping.")
            continue

        out_dir = out_root / portal / ct_slug / study_id
        unit = "Months" if "MONTH" in str(time_col).upper() else ("Days" if "DAY" in str(time_col).upper() else None)
        full_title = f"{title} — {ct} [{study_id}]" + ("" if subset == "all" else f" ({subset} set)")
        stats = plot_km_curves(df, group_col, time_col, "__status_bin", str(out_dir / "km.png"), full_title,
                               order=order, xlabel=f"Time ({unit})" if unit else time_col)
        stats.insert(0, "subset", subset)
        stats.to_csv(out_dir / "km_stats.tsv", sep="\t", index=False)
        print(f"[post] {portal}/{ct}/{study_id}: " + "; ".join(
            f"{r.group}: n={r.n}, events={r.events}" for r in stats.itertuples())
              + (f"; log-rank p={stats.logrank_p.iloc[0]:.3g}" if len(stats) > 1 else ""))


# =============================================================================
# Feature-importance ablation (feature selection per-modality)
# =============================================================================

def _study_for_ct(cfg: dict, ct: str) -> Tuple[str, str]:
    """
    Resolve (portal, study_id) for a cancer type using config maps.
    Priority: cBio_studyId -> pedcBio_studyId. If not found, try to discover on disk.
    """
    modeled = modeling_studies(cfg, ct)          # modeling_studyId first (e.g. pbta_all, not brain_cptac_2020)
    if modeled:
        return modeled[0]
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


def _load_proc_for_study(portal: str, ct: str, study_id: str, mods: Sequence[str],
                         clinical_cfg: Optional[dict] = None):
    """
    Load processed train/test feature matrices + labels for a cancer type/study/portal.
    Supports any view in 'mods', including 'clinical', which is rebuilt from
    clinical_train/test.parquet exactly as graph_ai_model() builds it for training.
    """
    root = Path("data/gdm/filtered") / portal / _slug(ct) / study_id
    X_tr, X_te = {}, {}

    for m in mods:
        if m == "clinical":
            c_tr = pd.read_parquet(root / "clinical_train.parquet")
            c_te = pd.read_parquet(root / "clinical_test.parquet")
            c_tr["patientId"] = c_tr["patientId"].astype(str)
            c_te["patientId"] = c_te["patientId"].astype(str)
            X_tr[m], X_te[m] = _build_clinical_view(c_tr.set_index("patientId"), c_te.set_index("patientId"),
                                                    clinical_cfg or {})
            if X_tr[m] is None:
                raise FileNotFoundError(f"[post] clinical view could not be rebuilt from {root}.")
            continue
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


def _graph_views(meta: Dict, mods: Sequence[str]) -> List[str]:
    """Views that were trained with a patient-similarity graph (meta.json "graph_views");
    checkpoints written before that key existed had no clinical graph."""
    gv = meta.get("graph_views")
    return list(gv) if gv is not None else graph_view_names(mods, clinical_graph=False)


def _expected_in_dims(model_dict, num_view: int):
    # GCN encoders: gc1.weight is [in_dim, hidden]; the clinical view (Identity_E) feeds its
    # classifier directly, so its input dim is the classifier's in_features
    dims = []
    for i in range(num_view):
        enc = model_dict[f"E{i+1}"]
        if hasattr(enc, "gc1"):
            dims.append(int(enc.gc1.weight.shape[0]))
        else:
            dims.append(int(model_dict[f"C{i+1}"].clf.in_features))
    return dims


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
    graph_views = _graph_views(meta, mods)
    X_trte_list, A_trte_list = [], []
    for m in mods:
        Xtr = X_tr_aln[m].to_numpy(dtype=np.float32)
        Xte = X_te_aln[m].to_numpy(dtype=np.float32)
        if m not in graph_views:                  # no graph (identity), as in training
            A_trte = None
        else:
            radius = compute_adaptive_radius_parameter_mogonet(Xtr, params)
            A_trte = build_trte_block_adjacency_mogonet(Xtr, Xte, radius=radius, params=params)
        X_trte_list.append(np.vstack([Xtr, Xte]))
        A_trte_list.append(A_trte)

    data_trte, adj_trte = tensors_from_numpy_lists(X_trte_list, A_trte_list, device="cpu")
    te_idx = list(range(len(X_tr_aln[mods[0]]), len(X_tr_aln[mods[0]]) + len(X_te_aln[mods[0]])))
    prob = test_epoch(data_trte, adj_trte, te_idx, model_dict)
    return prob, te_idx, X_tr_aln, X_te_aln, params


OMICS_VIEWS = ("mrna", "mirna", "methylation", "linear_cna")


def _class_index(y, classes: np.ndarray) -> np.ndarray:
    """Labels -> column index of the model output (train_with_split sorts the classes)."""
    return np.searchsorted(classes, np.asarray(y))


def _trte_view(Xtr: np.ndarray, Xte: np.ndarray, params, graph: bool):
    """Train+test block of one view as the model sees it: stacked features and, for a graph view,
    the train-test adjacency with the radius of the training patients."""
    if not graph:
        return np.vstack([Xtr, Xte]), None
    rad = compute_adaptive_radius_parameter_mogonet(Xtr, params)
    return np.vstack([Xtr, Xte]), build_trte_block_adjacency_mogonet(Xtr, Xte, radius=rad, params=params)


def _ablate_view(model_dict, views: Sequence[str], m: str, base_X: Dict[str, np.ndarray],
                 base_A: Dict[str, object], Xt: np.ndarray, Xe: np.ndarray, te_idx: Sequence[int],
                 params, y_idx: np.ndarray, base_prob: np.ndarray, label: str = ""):
    """Zero each feature of view m (training and test patients), rebuild that view's graph and
    predict the test patients with the trained model. Returns, per feature, the drop of the
    predicted probability of the true class for every test patient (features x patients) and the
    drop in macro F1."""
    rows = np.arange(len(y_idx))
    base_p = base_prob[rows, y_idx]
    base_f1 = f1_score(y_idx, base_prob.argmax(1), average="macro")
    V = Xt.shape[1]
    dprob = np.zeros((V, len(y_idx)), dtype=np.float64)
    df1 = np.zeros(V, dtype=np.float64)
    for j in range(V):
        Xtr_v, Xte_v = Xt.copy(), Xe.copy()
        Xtr_v[:, j] = 0.0
        Xte_v[:, j] = 0.0
        X_v, A_v = _trte_view(Xtr_v, Xte_v, params, graph=True)
        X_all = [X_v if mm == m else base_X[mm] for mm in views]
        A_all = [A_v if mm == m else base_A[mm] for mm in views]
        data_trte, adj_trte = tensors_from_numpy_lists(X_all, A_all, device="cpu")
        prob = test_epoch(data_trte, adj_trte, te_idx, model_dict)
        dprob[j] = base_p - prob[rows, y_idx]
        df1[j] = base_f1 - f1_score(y_idx, prob.argmax(1), average="macro")
        if (j + 1) % 250 == 0 and j + 1 < V:
            print(f"[post] {label}{m}: {j + 1}/{V} features")
    return dprob, df1


def _write_importance(path: Path, features: Sequence[str], cols: Dict[str, np.ndarray], score: str) -> pd.DataFrame:
    """One row per feature, highest score first (ties by feature name, so the order is stable)."""
    out = pd.DataFrame({"feature": [str(f) for f in features], **cols})
    key = "delta_prob" if score == "prob" else "delta_f1"
    out = out.sort_values([key, "feature"], ascending=[False, True], kind="mergesort")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _report(out: pd.DataFrame, label: str, path: Path) -> None:
    n = len(out)
    pos, zero = int((out.delta_prob > 0).sum()), int((out.delta_prob == 0).sum())
    top = ", ".join(f"{r.feature} ({r.delta_prob:.2e})" for r in out.head(3).itertuples())
    print(f"[post] {label}: {n} features, delta_prob > 0 for {pos}, = 0 for {zero}; top: {top} -> {path}")


def _importance_test_split(cfg: dict, ct: str, ab: dict, score: str, out_root: Path) -> None:
    """Ablation on the held-out test split with the model trained by graph_ai_model."""
    ct_slug = _slug(ct)
    edges_per_node = int(ab.get("edges_per_node", 5))
    # load trained model bundle + metadata, saved by graph_ai_model() under
    # <model_mogonet.checkpoints_dir>/<portal>/<ct_slug>/<study_id>/ (older runs: <checkpoints_dir>/<ct>/)
    portal, study_id = _study_for_ct(cfg, ct)
    ckpt_root = Path((cfg.get("model_mogonet") or cfg.get("model") or {}).get("checkpoints_dir", "data/models"))
    candidates = [ckpt_root / portal / ct_slug / study_id, ckpt_root / ct_slug, ckpt_root / str(ct).replace(" ", "_")]
    ckpt_dir = next((c for c in candidates if (c / "meta.json").exists()), None)
    if ckpt_dir is None:
        print(f"[post] {ct}: checkpoints not found at {candidates[0]}; skipping selection.")
        return
    model_dict, meta = load_model_bundle(ckpt_dir)

    # ---- USE TRAINED VIEW ORDER STRICTLY ----
    trained_views = list(meta.get("views") or [])
    if not trained_views:
        raise RuntimeError("Checkpoint meta.json missing 'views'. Retrain to save 'views' in meta.")
    mods_to_load = trained_views
    print(f"[post] {ct}: portal={portal} study_id={study_id} trained_views={mods_to_load}")

    try:
        clin_cfg = ((cfg.get("model_mogonet") or {}).get("clinical_view") or {})
        X_tr, X_te, y_tr, y_te, filt_root = _load_proc_for_study(portal, ct, study_id, mods_to_load, clin_cfg)
    except FileNotFoundError as e:
        print(str(e))
        return
    classes = np.unique(np.concatenate([_coerce_int_labels(y_tr.values), _coerce_int_labels(y_te.values)]))
    y_idx = _class_index(_coerce_int_labels(y_te.values), classes)
    X_tr, X_te = _reorder_to_trained_order(X_tr, X_te, mods_to_load)

    base_prob, te_idx, X_tr_aln, X_te_aln, params = _baseline_probs(
        model_dict, mods_to_load, X_tr, X_te, edges_per_node, meta
    )
    base_pred = base_prob.argmax(1)
    base_auc = None
    if base_prob.shape[1] == 2:
        try:
            base_auc = roc_auc_score(y_idx, base_prob[:, 1])
        except Exception:
            pass
    print(f"[post] {ct}: baseline acc={accuracy_score(y_idx, base_pred):.4f}, "
          f"f1={f1_score(y_idx, base_pred, average='macro'):.4f}"
          + ("" if base_auc is None else f", auc={base_auc:.4f}") + f" on {len(y_idx)} test patients")

    graph_views = _graph_views(meta, mods_to_load)
    base_X, base_A = {}, {}
    for m in mods_to_load:
        base_X[m], base_A[m] = _trte_view(X_tr_aln[m].to_numpy(np.float32), X_te_aln[m].to_numpy(np.float32),
                                          params, graph=m in graph_views)

    out_dir = out_root / portal / ct_slug / study_id
    for m in mods_to_load:
        if m not in OMICS_VIEWS:
            print(f"[post] {portal}/{ct}/{study_id}/{m}: skip importance (non-omics).")
            continue
        Xt, Xe = X_tr_aln[m].to_numpy(np.float32), X_te_aln[m].to_numpy(np.float32)
        dprob, df1 = _ablate_view(model_dict, mods_to_load, m, base_X, base_A, Xt, Xe, te_idx, params,
                                  y_idx, base_prob, label=f"{ct}/")
        path = out_dir / f"{m}_selection.csv"
        out = _write_importance(path, X_tr_aln[m].columns, {"delta_prob": dprob.mean(axis=1),
                                                            "delta_f1": df1 * Xt.shape[1]}, score)
        _report(out, f"{ct}/{m} (test split)", path)


def _importance_cv(cfg: dict, ct: str, ab: dict, score: str, out_root: Path) -> None:
    """Retrain MOGONET on each benchmark fold (data_process features, all modeled patients) and
    average the ablation over the test folds, so every patient is scored once."""
    from sklearn.model_selection import StratifiedKFold
    from . import benchmark as bm
    from . import graph_ai as gai

    b, hp = bm._bench_cfg(cfg), bm._mogonet_cfg(cfg)
    hp["edges_per_node"] = int(ab.get("edges_per_node", hp["edges_per_node"]))
    clin_cfg = (cfg.get("model_mogonet", {}) or {}).get("clinical_view") or {}
    clinical_features = (list(clin_cfg.get("features", ["AGE", "CLINICAL_STAGE", "RACE"]))
                         if clin_cfg.get("enabled", False) else None)
    mods = [m for m in gai._normalize_modalities((cfg.get("expression_profiles") or {}).get(ct, []))
            if m in OMICS_VIEWS]
    portal, study_id = _study_for_ct(cfg, ct)
    try:
        X, y, clin = bm.load_cohort(cfg, ct, portal, study_id, mods, "precomputed")
    except FileNotFoundError as e:
        print(f"[post] {ct}: {e}")
        return
    classes = np.unique(y.to_numpy())
    params = MOGONETGraphParams(edges_per_node=hp["edges_per_node"])
    skf = StratifiedKFold(n_splits=b["n_splits"], shuffle=True, random_state=b["seed"])
    print(f"[post] {ct}: importance over {b['n_splits']} CV folds (seed {b['seed']}), {len(y)} patients, views {mods}")

    sums = {m: np.zeros(X[m].shape[1]) for m in mods}
    fold_means = {m: [] for m in mods}
    fold_f1 = {m: [] for m in mods}
    for k, (i_tr, i_te) in enumerate(skf.split(np.zeros(len(y)), y.to_numpy()), start=1):
        tr, te = y.index[i_tr], y.index[i_te]
        Xtr, Xte = bm.fold_views(X, y, clin, tr, te, mods, "precomputed", bm._label_cfg(cfg, ct), clinical_features)
        views = list(Xtr)
        metrics, model_dict = gai.train_with_split(views, dict(Xtr), dict(Xte), y.loc[tr], y.loc[te],
                                                   save_dir=None, seed=b["seed"], **hp)
        graph_views = gai.graph_view_names(views, hp["clinical_graph"])
        base_X, base_A = {}, {}
        for v in views:
            base_X[v], base_A[v] = _trte_view(Xtr[v].to_numpy(np.float32), Xte[v].to_numpy(np.float32),
                                              params, graph=v in graph_views)
        te_idx = list(range(len(tr), len(tr) + len(te)))
        data_trte, adj_trte = tensors_from_numpy_lists([base_X[v] for v in views], [base_A[v] for v in views],
                                                       device="cpu")
        base_prob = test_epoch(data_trte, adj_trte, te_idx, model_dict)
        y_idx = _class_index(y.loc[te].to_numpy(), classes)
        print(f"[post] {ct} fold {k}/{b['n_splits']}: acc={metrics.get('acc', float('nan')):.3f} "
              f"f1={metrics.get('f1', float('nan')):.3f} on {len(te)} test patients")
        for m in mods:
            dprob, df1 = _ablate_view(model_dict, views, m, base_X, base_A, Xtr[m].to_numpy(np.float32),
                                      Xte[m].to_numpy(np.float32), te_idx, params, y_idx, base_prob,
                                      label=f"{ct} fold {k}/")
            sums[m] += dprob.sum(axis=1)
            fold_means[m].append(dprob.mean(axis=1))
            fold_f1[m].append(df1 * X[m].shape[1])

    out_dir = out_root / portal / _slug(ct) / study_id
    for m in mods:
        per_fold = np.vstack(fold_means[m])
        path = out_dir / f"{m}_selection_cv.csv"
        out = _write_importance(path, X[m].columns, {
            "delta_prob": sums[m] / len(y),                      # mean over all patients (each in one test fold)
            "delta_prob_sd": per_fold.std(axis=0, ddof=1),       # SD of the fold means
            "folds_positive": (per_fold > 0).sum(axis=0),
            "delta_f1": np.vstack(fold_f1[m]).mean(axis=0),      # MOGONET score, mean over folds
        }, score)
        _report(out, f"{ct}/{m} ({b['n_splits']}-fold CV)", path)


def run_feature_selection():
    """
    Ablation importance per omics feature (post_analysis.feature_selection):
      score        prob (default): mean drop of the predicted probability of the true class when
                   the feature is zeroed; f1: drop in macro F1 x number of features in the view,
                   as in MOGONET (a feature scores 0 unless a test prediction flips). Both are
                   written; score sets the ranking.
      evaluate_on  test (default): the model trained by graph_ai_model, on its held-out test
                   split -> <mod>_selection.csv; cv: MOGONET retrained on each benchmark fold
                   (benchmark.n_splits / seed, data_process features) -> <mod>_selection_cv.csv.
    """
    cfg = _cfg()
    ab = (cfg.get("post_analysis") or {}).get("feature_selection") or {}
    if not ab or not ab.get("enabled", False):
        print("[post] Feature selection disabled or not configured.")
        return
    score = str(ab.get("score", "prob")).lower()
    evaluate_on = str(ab.get("evaluate_on", "test")).lower()
    if score not in ("prob", "f1") or evaluate_on not in ("test", "cv"):
        raise ValueError("post_analysis.feature_selection: score must be prob or f1, evaluate_on test or cv")
    out_root = Path(ab.get("out_dir", "data/analysis/feature_selection"))
    for ct in cfg.get("cancer_type_interest", []):
        if evaluate_on == "cv":
            _importance_cv(cfg, ct, ab, score, out_root)
        else:
            _importance_test_split(cfg, ct, ab, score, out_root)


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