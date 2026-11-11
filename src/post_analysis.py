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
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import yaml

# ---- graph / GNN utilities reused from graph_ai and graph_constructor ----
from .study_scope import is_modeled, modeling_studies
from .cohort import class_names, load_labels
from .graph_ai import (
    tensors_from_numpy_lists,
    test_epoch,
    encode,
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


# post_analysis.survival.style: publication. Font sizes, 600 dpi and titles (title, then the log-rank
# p-value) of PCGS's plot_km_curves_publish, drawn on a fixed canvas: every figure has the same size
# and the same axes box (inches), so KM figures placed side by side line up.
PUBLICATION = {"width": 7.0, "left": 1.35, "right": 0.35, "top": 1.15, "axes_height": 4.6,
               "bottom": 1.25, "legend_row": 0.5,
               "title": 22, "subtitle": 22, "label": 22, "ticks": 22, "legend": 20,
               "linewidth": 1.5, "dpi": 600, "font": "DejaVu Sans"}
# legend positions tried inside the axes with legend: inside (first clear of every curve and band)
_LEGEND_SPOTS = ["upper right", "lower right", "lower left", "upper left", "center right", "lower center",
                 "upper center", "center left", "center"]


def _logrank(df: pd.DataFrame, group_col: str, surv_time: str, surv_status: str, groups: Sequence) -> float:
    """Log-rank p-value for two groups, multivariate log-rank for more; NaN for one group."""
    if len(groups) == 2:
        g1 = df[df[group_col] == groups[0]]
        g2 = df[df[group_col] == groups[1]]
        return float(logrank_test(g1[surv_time], g2[surv_time], event_observed_A=g1[surv_status],
                                  event_observed_B=g2[surv_status]).p_value)
    if len(groups) > 2:
        return float(multivariate_logrank_test(df[surv_time], df[group_col], df[surv_status]).p_value)
    return np.nan


def _curve_paths(ax):
    """Survival curves (step lines) and confidence bands of an axes as display-space paths."""
    from matplotlib.path import Path as MPath
    out = []
    for line in ax.get_lines():
        x, y = (np.asarray(v, dtype=float) for v in line.get_data())
        if len(x) < 2:
            continue
        xy = np.column_stack([np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]])   # steps-post
        out.append((MPath(ax.transData.transform(xy)), False))
    for coll in ax.collections:
        for path in coll.get_paths():
            out.append((path.transformed(coll.get_transform()), True))
    return out


def _legend_inside(fig, ax, handles, labels, fontsize) -> bool:
    """Put the legend at the first of _LEGEND_SPOTS where its box touches no curve and no
    confidence band; False (no legend left on the axes) when every spot is taken."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    obstacles = _curve_paths(ax)
    for loc in _LEGEND_SPOTS:
        leg = ax.legend(handles, labels, loc=loc, fontsize=fontsize, frameon=True, framealpha=0.6)
        box = leg.get_window_extent(renderer).padded(4)
        if not any(path.intersects_bbox(box, filled=filled) for path, filled in obstacles):
            return True
        leg.remove()
    return False


def _km_figure_publication(fits, title: str, p: float, xlabel: Optional[str], filename: str,
                           legend: str = "below", xlim: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """KM figure for a figure panel. The canvas and the axes box are fixed, and the title, the
    log-rank p-value and the legend are centered on the axes, so figures with different titles,
    groups or follow-up still align. legend: below (default, under the x-axis label, never on a
    curve) or inside (first spot inside the axes clear of the curves and bands, else below).
    Returns the canvas size (inches), the axes box (figure fraction) and where the legend went."""
    st = PUBLICATION
    n_rows = 1
    with plt.rc_context({"font.family": st["font"]}):
        fig = plt.figure()
        W = st["width"]
        ax_w = W - st["left"] - st["right"]
        ax = fig.add_axes([0, 0, 1, 1])                     # placed below, once the height is known
        for kmf in fits:
            kmf.plot_survival_function(ax=ax, linewidth=st["linewidth"])
        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ncol = min(len(labels), 3)
        n_rows = int(np.ceil(len(labels) / ncol)) if labels else 1
        bottom = st["bottom"] + st["legend_row"] * n_rows   # reserved in every mode: same axes box
        H = st["top"] + st["axes_height"] + bottom
        fig.set_size_inches(W, H)
        ax.set_position([st["left"] / W, bottom / H, ax_w / W, st["axes_height"] / H])

        ax.set_ylim(-0.02, 1.04)
        ax.set_yticks(np.linspace(0, 1, 6))
        if xlim is not None:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.set_xlabel(xlabel or "Time", fontsize=st["label"])
        ax.set_ylabel("Survival Probability", fontsize=st["label"])
        ax.tick_params(axis="both", labelsize=st["ticks"])

        cx = (st["left"] + ax_w / 2) / W                   # center of the axes
        room = 2 * min(st["left"] + ax_w / 2, W - st["left"] - ax_w / 2) - 0.1
        renderer = fig.canvas.get_renderer()
        size = st["title"]
        t = fig.text(cx, 1 - 0.15 / H, title, ha="center", va="top", fontsize=size)
        while t.get_window_extent(renderer).width / fig.dpi > room and size > 12:
            size -= 1
            t.set_fontsize(size)
        if size < st["title"]:
            print(f"[post] KM title shortened to {size} pt to fit the figure: {title!r}")
        if not np.isnan(p):
            fig.text(cx, 1 - 0.62 / H, f"log-rank p-value = {p:.3e}", ha="center", va="top",
                     fontsize=st["subtitle"])

        where = "below"
        if str(legend).lower() == "inside" and _legend_inside(fig, ax, handles, labels, st["legend"]):
            where = "inside"
        if where == "below" and handles:
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(cx, 0.1 / H), ncol=ncol,
                       fontsize=st["legend"], frameon=True, framealpha=0.6)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=st["dpi"])               # no bbox_inches="tight": same canvas every time
        fig.savefig(str(Path(filename).with_suffix(".pdf")))
        info = {"size": (W, H), "axes": tuple(ax.get_position().bounds), "legend": where}
        plt.close(fig)
    return info


def plot_km_curves(clinical_df: pd.DataFrame, group_col: str, surv_time: str, surv_status: str, filename: str,
                   title: str, order: Optional[Sequence] = None, xlabel: Optional[str] = None,
                   publish: bool = False, legend: str = "below",
                   xlim: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """
    Plot Kaplan–Meier curves stratified by group_col and save to 'filename'.
    clinical_df columns: surv_time (e.g., OS_MONTHS), surv_status (0/1), group_col.
    order: groups in plotting (and color) order; default sorted. publish: figure-panel layout of
    PCGS (title and log-rank p-value above the axes, fixed canvas, 600 dpi PNG and a PDF) with the
    legend below the axes or, legend="inside", inside where it covers no curve; xlim: x-axis range.
    Returns one row per group (n, events, median survival) with the log-rank p-value.
    """
    present = list(clinical_df[group_col].dropna().unique())
    groups = [g for g in order if g in present] if order is not None else sorted(present, key=str)

    fits, stats = [], []
    for grp in groups:
        ix = clinical_df[group_col] == grp
        kmf = KaplanMeierFitter().fit(durations=clinical_df.loc[ix, surv_time],
                                      event_observed=clinical_df.loc[ix, surv_status], label=str(grp))
        fits.append(kmf)
        stats.append({"group": grp, "n": int(ix.sum()), "events": int(clinical_df.loc[ix, surv_status].sum()),
                      "median_survival": float(kmf.median_survival_time_)})
    p = _logrank(clinical_df, group_col, surv_time, surv_status, groups)

    if publish:
        _km_figure_publication(fits, title, p, xlabel, filename, legend=legend, xlim=xlim)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for kmf in fits:
            kmf.plot_survival_function(ax=ax)
        if xlim is not None:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        if not np.isnan(p):
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            label = "Log-rank" if len(groups) == 2 else "Multivariate log-rank"
            ax.text(x_min + 0.1 * (x_max - x_min), y_min + 0.1 * (y_max - y_min), f"{label} {_p_text(p)}",
                    fontsize=11, bbox=dict(facecolor="white", alpha=0.6))
        ax.set_title(title)
        ax.set_xlabel(xlabel or "Time")
        ax.set_ylabel("Survival Probability")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
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


def _patient_clinical(portal: str, ct_slug: str, study_id: str, need: Sequence[str] = ()) -> Optional[pd.DataFrame]:
    """Patient clinical table of a modeled study: clinical_filtered from data_process (the modeled
    patients), else clinical_patients.csv from data_extract, else the GDM subjects table. With
    need, the first table that has those columns."""
    filt_dir = Path("data/gdm/filtered") / portal / ct_slug / study_id
    cands = [filt_dir / "clinical_filtered.parquet", filt_dir / "clinical_filtered.csv",
             Path("data/fetch/cbioportal_api_request") / portal / ct_slug / study_id / "clinical_patients.csv",
             Path("data/gdm/clinical/clinical_subjects.parquet")]
    first = None
    for cand in cands:
        if not cand.exists():
            continue
        pat = pd.read_parquet(cand) if cand.suffix == ".parquet" else pd.read_csv(cand)
        if "patientId" not in pat.columns and "subject_id" in pat.columns:
            pat = pat.rename(columns={"subject_id": "patientId"})
        if all(c in pat.columns for c in need):
            return pat
        first = pat if first is None else first
    return first


def run_km_survival():
    """
    KM curves per modeled study (post_analysis.survival):
      group_col  clinical column, or "label" (the class modeled by data_process, named by
                 processor.labels.<type>.class_names); one value, or one per cancer type
      subset     all (train + test, default), train or test (the split of data_process)
      time_col, status_col, title, out_dir
      style      default, or publication: figure-panel layout of PCGS (title and log-rank p-value
                 above the axes, 22-pt type, 600 dpi PNG and a PDF) on a fixed canvas, so figures
                 line up; the title has no cancer type and study
      legend     below (default: under the x-axis label) or inside (where it covers no curve or
                 confidence band, else below); publication style only
      xlim       x-axis range, e.g. [0, 300] (default: automatic), to give figures the same axis
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
        publish = str(_for_type(pa.get("style"), ct, "default")).lower() == "publication"
        legend = str(_for_type(pa.get("legend"), ct, "below"))
        xlim = _for_type(pa.get("xlim"), ct)
        if not group_col:
            print(f"[post] {portal}/{ct}/{study_id}: no group_col for this cancer type; skipping KM.")
            continue
        if subset not in ("all", "train", "test"):
            print(f"[post] {portal}/{ct}/{study_id}: subset must be all, train or test (got {subset!r}); skipping KM.")
            continue
        pat = _patient_clinical(portal, ct_slug, study_id,
                                need=[time_col] + ([] if group_col == "label" else [group_col]))
        if pat is None:
            print(f"[post] No clinical table for {portal}/{ct}/{study_id}; skipping.")
            continue

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
        full_title = (title if publish else f"{title} — {ct} [{study_id}]") + ("" if subset == "all" else f" ({subset} set)")
        stats = plot_km_curves(df, group_col, time_col, "__status_bin", str(out_dir / "km.png"), full_title,
                               order=order, xlabel=f"Time ({unit})" if unit else time_col, publish=publish,
                               legend=legend, xlim=xlim)
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


def _load_trained(cfg: dict, ct: str, edges_per_node: Optional[int], task: str = "selection") -> Optional[dict]:
    """The model trained by graph_ai_model for a cancer type with its data_process train/test matrices,
    aligned to the trained views and features, the test-set probabilities and the train+test graph
    of every view (edges_per_node None: the value the model was trained with). None (with a
    message) when the checkpoint or the matrices are missing."""
    ct_slug = _slug(ct)
    # load trained model bundle + metadata, saved by graph_ai_model() under
    # <model_mogonet.checkpoints_dir>/<portal>/<ct_slug>/<study_id>/ (older runs: <checkpoints_dir>/<ct>/)
    portal, study_id = _study_for_ct(cfg, ct)
    ckpt_root = Path((cfg.get("model_mogonet") or cfg.get("model") or {}).get("checkpoints_dir", "data/models"))
    candidates = [ckpt_root / portal / ct_slug / study_id, ckpt_root / ct_slug, ckpt_root / str(ct).replace(" ", "_")]
    ckpt_dir = next((c for c in candidates if (c / "meta.json").exists()), None)
    if ckpt_dir is None:
        print(f"[post] {ct}: checkpoints not found at {candidates[0]}; skipping {task}.")
        return None
    model_dict, meta = load_model_bundle(ckpt_dir)
    if edges_per_node is None:
        edges_per_node = int(meta.get("edges_per_node") or (cfg.get("model_mogonet") or {}).get("edges_per_node", 5))

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
        return None
    classes = np.unique(np.concatenate([_coerce_int_labels(y_tr.values), _coerce_int_labels(y_te.values)]))
    X_tr, X_te = _reorder_to_trained_order(X_tr, X_te, mods_to_load)

    base_prob, te_idx, X_tr_aln, X_te_aln, params = _baseline_probs(
        model_dict, mods_to_load, X_tr, X_te, edges_per_node, meta
    )
    graph_views = _graph_views(meta, mods_to_load)
    base_X, base_A = {}, {}
    for m in mods_to_load:
        base_X[m], base_A[m] = _trte_view(X_tr_aln[m].to_numpy(np.float32), X_te_aln[m].to_numpy(np.float32),
                                          params, graph=m in graph_views)
    return {"portal": portal, "study_id": study_id, "ct_slug": ct_slug, "model_dict": model_dict, "meta": meta,
            "views": mods_to_load, "graph_views": graph_views, "X_tr": X_tr_aln, "X_te": X_te_aln,
            "y_tr": y_tr, "y_te": y_te, "classes": classes, "params": params, "base_prob": base_prob,
            "te_idx": te_idx, "base_X": base_X, "base_A": base_A}


def _importance_test_split(cfg: dict, ct: str, ab: dict, score: str, out_root: Path) -> None:
    """Ablation on the held-out test split with the model trained by graph_ai_model."""
    t = _load_trained(cfg, ct, int(ab.get("edges_per_node", 5)))
    if t is None:
        return
    portal, study_id, ct_slug = t["portal"], t["study_id"], t["ct_slug"]
    model_dict, mods_to_load, params = t["model_dict"], t["views"], t["params"]
    X_tr_aln, X_te_aln, base_prob, te_idx = t["X_tr"], t["X_te"], t["base_prob"], t["te_idx"]
    base_X, base_A = t["base_X"], t["base_A"]
    y_idx = _class_index(_coerce_int_labels(t["y_te"].values), t["classes"])
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
# Hierarchical clustering of the patients' latent features + KM per cluster
# =============================================================================
RISK_ORDER = ["Low-risk", "High-risk"]


def hierarchical_clustering_analysis(latent_features: np.ndarray, clinical_df: pd.DataFrame,
                                     time_col: str = "OS_MONTHS", status_col: str = "OS_STATUS_Binary",
                                     n_clusters: int = 2, filename: Optional[str] = None,
                                     title: str = "K-M Plot for Hierarchical Clusters",
                                     xlabel: Optional[str] = None, publish: bool = False,
                                     legend: str = "below", xlim: Optional[Sequence[float]] = None):
    """
    Ward hierarchical clustering (Euclidean) of latent_features (patients x dimensions, rows in the
    order of clinical_df) cut into n_clusters groups. With two clusters, the cluster with the lower
    median time_col (over its patients with a time) is "High-risk" and the other "Low-risk";
    otherwise each patient's group is "Cluster k". With filename, KM curves of the groups (patients
    with time_col and status_col) and the log-rank test (multivariate for more than two groups).
    Returns (cluster_labels, clinical_df with columns cluster and risk_group, KM stats or None).
    """
    Z = linkage(np.asarray(latent_features, dtype=np.float64), method="ward", metric="euclidean")
    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    df = clinical_df.copy()
    df["cluster"] = cluster_labels
    found = sorted(np.unique(cluster_labels))
    times = pd.to_numeric(df[time_col], errors="coerce") if time_col in df.columns else pd.Series(np.nan, df.index)
    medians = {c: float(np.nanmedian(times[df["cluster"] == c])) if times[df["cluster"] == c].notna().any()
               else np.nan for c in found}
    order = None
    if len(found) == 2 and all(np.isfinite(v) for v in medians.values()):
        high = min(found, key=lambda c: medians[c])           # lower median time -> High-risk
        df["risk_group"] = np.where(df["cluster"] == high, "High-risk", "Low-risk")
        order = RISK_ORDER
        print(f"[post] clusters: " + ", ".join(
            f"{c} ({'High' if c == high else 'Low'}-risk, n={int((df.cluster == c).sum())}, "
            f"median {time_col}={medians[c]:.1f})" for c in found))
    else:
        df["risk_group"] = [f"Cluster {c}" for c in df["cluster"]]
        order = [f"Cluster {c}" for c in found]
    stats = None
    if filename:
        surv = df.copy()
        surv[time_col] = times
        surv = surv[surv[time_col].notna() & surv[status_col].notna()]
        if surv["risk_group"].nunique() < 1:
            print("[post] clusters: no patient with survival data; no KM plot.")
        else:
            stats = plot_km_curves(surv, "risk_group", time_col, status_col, filename, title, order=order,
                                   xlabel=xlabel, publish=publish, legend=legend, xlim=xlim)
    return cluster_labels, df, stats


def _latent_features(t: dict, subset: str, views: Sequence[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Encoder output of the trained model for the given views, concatenated per patient. Training
    patients go through the training graph (as in training), test patients through the train-test
    graph (as in prediction). Returns latent (patients x dims), patient IDs and their split."""
    model_dict, all_views, params = t["model_dict"], t["views"], t["params"]
    pick = [i for i, m in enumerate(all_views) if m in views]
    blocks, ids, split = [], [], []
    if subset in ("train", "all"):
        X_list, A_list = [], []
        for m in all_views:
            Xtr = t["X_tr"][m].to_numpy(np.float32)
            A = None
            if m in t["graph_views"]:
                A = build_train_adjacency_mogonet(Xtr, radius=compute_adaptive_radius_parameter_mogonet(Xtr, params),
                                                  params=params)
            X_list.append(Xtr)
            A_list.append(A)
        z = encode(*tensors_from_numpy_lists(X_list, A_list, device="cpu"), model_dict)
        blocks.append(np.hstack([z[i] for i in pick]))
        ids += [str(i) for i in t["X_tr"][all_views[0]].index]
        split += ["train"] * len(t["X_tr"][all_views[0]])
    if subset in ("test", "all"):
        z = encode(*tensors_from_numpy_lists([t["base_X"][m] for m in all_views],
                                             [t["base_A"][m] for m in all_views], device="cpu"), model_dict)
        n_tr = len(t["X_tr"][all_views[0]])
        blocks.append(np.hstack([z[i][n_tr:] for i in pick]))
        ids += [str(i) for i in t["X_te"][all_views[0]].index]
        split += ["test"] * len(t["X_te"][all_views[0]])
    return np.vstack(blocks), ids, split


def run_clustering():
    """
    Hierarchical clustering of patients by the latent features of the model trained by
    graph_ai_model (post_analysis.clustering), then KM curves of the clusters:
      subset            test (default), train or all
      n_clusters        2 (default); with two, the cluster with the lower median time_col is
                        High-risk (lower median OS months, as in PCGS)
      include_clinical  false (default): omics views only; true also uses the clinical view
      standardize       false (default); true: z-score each latent dimension first
      time_col, status_col, title, style, legend, xlim (default: as post_analysis.survival),
      out_dir; cancer_types: the
      types to cluster (default: every type in cancer_type_interest with a trained model);
      edges_per_node: graph density (default: the value the model was trained with)
    The latent features are the GCN embedding (last encoder layer) of each view, concatenated.
    Writes clusters.tsv, cluster_vs_class.tsv, km_clusters.png and km_clusters_stats.tsv under
    <out_dir>/<portal>/<cancer type>/<study>/.
    """
    cfg = _cfg()
    cl = (cfg.get("post_analysis") or {}).get("clustering") or {}
    if not cl or not cl.get("enabled", False):
        print("[post] Clustering disabled or not configured.")
        return
    surv_cfg = (cfg.get("post_analysis") or {}).get("survival") or {}
    edges = cl.get("edges_per_node")                     # default: as trained (meta.json)
    edges = None if edges is None else int(edges)
    out_root = Path(cl.get("out_dir", "data/analysis/clusters"))
    types = cl.get("cancer_types") or cfg.get("cancer_type_interest", [])
    for ct in types:
        subset = str(_for_type(cl.get("subset"), ct, "test")).lower()
        if subset not in ("all", "train", "test"):
            print(f"[post] {ct}: clustering subset must be all, train or test (got {subset!r}); skipping.")
            continue
        n_clusters = int(_for_type(cl.get("n_clusters"), ct, 2))
        time_col = _for_type(cl.get("time_col"), ct, surv_cfg.get("time_col", "OS_MONTHS"))
        status_c = _for_type(cl.get("status_col"), ct, surv_cfg.get("status_col", "OS_STATUS"))
        title = _for_type(cl.get("title"), ct, "OS by Hierarchical Clusters")
        publish = str(_for_type(cl.get("style"), ct, _for_type(surv_cfg.get("style"), ct, "default"))
                      ).lower() == "publication"
        legend = str(_for_type(cl.get("legend"), ct, _for_type(surv_cfg.get("legend"), ct, "below")))
        xlim = _for_type(cl.get("xlim"), ct, _for_type(surv_cfg.get("xlim"), ct))

        t = _load_trained(cfg, ct, edges, task="clustering")
        if t is None:
            continue
        portal, study_id, ct_slug = t["portal"], t["study_id"], t["ct_slug"]
        views = [m for m in t["views"] if m != "clinical" or bool(cl.get("include_clinical", False))]
        latent, ids, split = _latent_features(t, subset, views)
        if bool(cl.get("standardize", False)):
            sd = latent.std(axis=0)
            latent = (latent - latent.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        if len(ids) < max(n_clusters, 2):
            print(f"[post] {portal}/{ct}/{study_id}: {len(ids)} patients; too few to cluster.")
            continue

        names = class_names(cfg, ct)
        y = pd.concat([t["y_tr"], t["y_te"]])
        y.index = y.index.astype(str)
        df = pd.DataFrame({"patientId": ids, "split": split})
        df["class"] = [names.get(int(k), f"label {k}") for k in _coerce_int_labels(y.loc[ids].values)]
        pat = _patient_clinical(portal, ct_slug, study_id, need=[time_col])
        if pat is not None and time_col in pat.columns:
            pat = pat.assign(patientId=pat["patientId"].astype(str).str.strip())
            pat = pat.drop_duplicates("patientId").set_index("patientId")
            df[time_col] = pd.to_numeric(pat[time_col].reindex(ids), errors="coerce").to_numpy()
            status = _derive_os_binary(pat, status_col=status_c) if status_c in pat.columns else None
            df["status"] = status.reindex(ids).to_numpy() if status is not None else np.nan
            df.loc[df[time_col].isna(), "status"] = np.nan
        else:
            print(f"[post] {portal}/{ct}/{study_id}: '{time_col}' not in the clinical table; clusters without KM.")
            df[time_col], df["status"] = np.nan, np.nan

        out_dir = out_root / portal / ct_slug / study_id
        out_dir.mkdir(parents=True, exist_ok=True)
        unit = "Months" if "MONTH" in str(time_col).upper() else ("Days" if "DAY" in str(time_col).upper() else None)
        full_title = (title if publish else f"{title} — {ct} [{study_id}]") + ("" if subset == "all" else f" ({subset} set)")
        has_surv = df[time_col].notna().any()
        print(f"[post] {portal}/{ct}/{study_id}: clustering {len(ids)} {subset} patients on "
              f"{latent.shape[1]} latent dimensions ({', '.join(views)})")
        _, df, stats = hierarchical_clustering_analysis(
            latent, df, time_col=time_col, status_col="status", n_clusters=n_clusters,
            filename=str(out_dir / "km_clusters.png") if has_surv else None, title=full_title,
            xlabel=f"Time ({unit})" if unit else time_col, publish=publish, legend=legend, xlim=xlim)
        df.rename(columns={"status": "status_binary"}).to_csv(out_dir / "clusters.tsv", sep="\t", index=False)
        pd.crosstab(df["risk_group"], df["class"]).to_csv(out_dir / "cluster_vs_class.tsv", sep="\t")
        if stats is not None:
            stats.insert(0, "subset", subset)
            stats.to_csv(out_dir / "km_clusters_stats.tsv", sep="\t", index=False)
            print(f"[post] {portal}/{ct}/{study_id}: " + "; ".join(
                f"{r.group}: n={r.n}, events={r.events}" for r in stats.itertuples())
                  + (f"; log-rank p={stats.logrank_p.iloc[0]:.3g}" if len(stats) > 1 else "") + f" -> {out_dir}")


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
    if (sec.get("clustering") or {}).get("enabled", False):
        run_clustering()
        did = True

    if not did:
        print("[post] Nothing to do. Configure post_analysis.survival/feature_selection/clustering in your YAML.")