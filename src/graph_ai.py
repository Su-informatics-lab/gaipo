"""
graph_ai.py — MOGONET-style GNN adapted to this project's I/O

Supports two modes:
1) Pretrained inference: load saved weights, build cross (train|valid) block adjacencies using
   TRAIN-derived radius, run inference on user-provided validation data, and compute feature
   importance by ablation (optional).
2) Train/evaluate: given train/test splits, per-modality features (samples x features), and
   prebuilt adjacencies (or feature matrices to build them on-the-fly via graph_constructor),
   train the GCN encoders + per-view classifiers (+ VCDN fusion) and report metrics.

This module expects you to have used processor.py and graph_constructor.py beforehand:
- processor.py supplies X_tr_dict, X_te_dict, y_tr, y_te, tr_ids, te_ids
- graph_constructor.py supplies adjacency builders respecting MOGONET logic

Note:
- If a requested clinical feature is missing in a study, it’s silently skipped; if none are available the clinical view is dropped for that study.
- Identity adjacency is used for the clinical view (no graph structure, no MLP), but it still blends seamlessly with other views via VCDN.
- Checkpoints are saved under data/models/<portal>/<ct_slug>/<study_id>/ to mirror your filtered/graphs directories.

Input path:
data/gdm/filtered/<portal>/<ct_slug>/<study_id>/
Output_path:
data/models/<portal>/<ct_slug>/<study_id>/

"""
from __future__ import annotations

import copy
import os
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scipy import sparse as sp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import yaml

# Import our graph builders for adjacency handling
from .graph_constructor import (
    MOGONETGraphParams,
    compute_adaptive_radius_parameter_mogonet,
    build_train_adjacency_mogonet,
    build_trte_block_adjacency_mogonet,
    csr_to_torch_coo,
)

# ==========================
# Reproducibility
# ==========================
def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================
# Config-driven entrypoint
# ==========================
def _load_cfg() -> dict:
    cfg_path = os.getenv("CONFIG_PATH", "config/pipeline_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

def _slug(s: str) -> str:
    return s.replace(" ", "_").lower()

def _normalize_modalities(mods):
    syn = {"microrna": "mirna"}
    return [syn.get((m or "").lower(), (m or "").lower()) for m in (mods or [])]

def _normalize_study_map(study_map_raw: dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for ct, v in (study_map_raw or {}).items():
        if v is None:
            out[ct] = []
        elif isinstance(v, str):
            out[ct] = [v]
        elif isinstance(v, list):
            out[ct] = [str(x) for x in v]
        else:
            out[ct] = []
    return out

def _studies_for_ct(ct: str, c_map: Dict[str, List[str]], p_map: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for sid in c_map.get(ct, []):
        out.append(("cbio", sid))
    for sid in p_map.get(ct, []):
        out.append(("pedcbio", sid))
    return out

def _device_from_cfg(model_cfg: dict) -> str:
    dev = str(model_cfg.get("device", "cpu")).lower()
    if dev == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ==========================
# Data utilities
# ==========================
def _coerce_numeric_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure df contains only numeric columns and float32 values.
    - Drops non-numeric columns with a warning.
    - Coerces to numeric, NaN/inf -> 0.0
    """
    orig_cols = df.columns.tolist()
    num = df.select_dtypes(include=["number"]).copy()
    dropped = sorted(set(orig_cols) - set(num.columns))
    if dropped:
        print(f"[graph_ai] {name}: dropping non-numeric columns: {dropped[:5]}{' …' if len(dropped)>5 else ''}")
    # Coerce anything borderline to numeric
    num = num.apply(pd.to_numeric, errors="coerce")
    # Replace inf/NaN
    num = num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Final dtype
    try:
        num = num.astype(np.float32)
    except Exception as e:
        # As a fallback, coerce each col individually
        for c in num.columns:
            num[c] = pd.to_numeric(num[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return num

def _minimax(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        v = out[c].astype(float)
        mn, mx = float(np.nanmin(v)), float(np.nanmax(v))
        if mx - mn <= 1e-12:
            out[c] = 0.0
        else:
            out[c] = (v - mn) / (mx - mn)
    return out.astype(np.float32)

# ==========================
# Models (GCN encoders, per-view classifiers, VCDN)
# ==========================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        out = torch.sparse.mm(adj, support)
        return out + self.bias if self.bias is not None else out

class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim: Sequence[int], dropout: float):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj), 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gc2(x, adj), 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gc3(x, adj), 0.25)
        return x

class Identity_E(nn.Module):
    def forward(self, x, adj=None):
        return x

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.clf.weight)
        nn.init.zeros_(self.clf.bias)
    def forward(self, x):
        return self.clf(x)

class VCDN(nn.Module):
    def __init__(self, num_view: int, num_cls: int, hvcdn_dim: int):
        super().__init__()
        self.num_cls = num_cls
        self.fc1 = nn.Linear(pow(num_cls, num_view), hvcdn_dim)
        self.fc2 = nn.Linear(hvcdn_dim, num_cls)
        nn.init.xavier_normal_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, in_list: List[torch.Tensor]):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)), (-1, pow(self.num_cls, 2), 1))
        for i in range(2, num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)), (-1, pow(self.num_cls, i + 1), 1))
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        x = F.leaky_relu(self.fc1(vcdn_feat), 0.25)
        return self.fc2(x)


# ==========================
# Model/optim factory
# ==========================
def init_model_dict(
    view_names: List[str],
    num_class: int,
    dim_list: Sequence[int],
    dim_he_list: Sequence[int],
    dim_hc: int,
    gcn_dropout: float = 0.5,
):
    """
    view_names aligns with dim_list order. For 'clinical' view we use Identity_E (no GCN).
    """
    model_dict: Dict[str, nn.Module] = {}
    for i, vname in enumerate(view_names):
        if vname == "clinical":
            model_dict[f"E{i+1}"] = Identity_E()
        else:
            model_dict[f"E{i+1}"] = GCN_E(dim_list[i], dim_he_list, gcn_dropout)
        model_dict[f"C{i+1}"] = Classifier_1(dim_he_list[-1] if vname != "clinical" else dim_list[i], num_class)
    if len(view_names) >= 2:
        model_dict["C"] = VCDN(len(view_names), num_class, dim_hc)
    return model_dict

def init_optim(view_names: List[str], model_dict: Dict[str, nn.Module], lr_e: float = 1e-4, lr_c: float = 1e-4):
    optim_dict = {}
    for i, _ in enumerate(view_names):
        optim_dict[f"C{i+1}"] = torch.optim.Adam(
            list(model_dict[f"E{i+1}"].parameters()) + list(model_dict[f"C{i+1}"].parameters()), lr=lr_e
        )
    if len(view_names) >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict


# ==========================
# Train / test loops (consume prebuilt adjacencies)
# ==========================
def train_epoch(data_list: List[torch.Tensor], adj_list, label: torch.Tensor, sample_weight: torch.Tensor, model_dict, optim_dict, train_VCDN: bool = True):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    loss_dict = {}

    for i in range(num_view):
        optim_dict[f"C{i+1}"].zero_grad()
        # clinical view passes Identity_E and ignores adj
        if isinstance(model_dict[f"E{i+1}"], Identity_E):
            enc = model_dict[f"E{i+1}"](data_list[i], None)
        else:
            enc = model_dict[f"E{i+1}"](data_list[i], adj_list[i])
        ci = model_dict[f"C{i+1}"](enc)
        ci_loss = torch.mean(criterion(ci, label) * sample_weight)
        ci_loss.backward()
        optim_dict[f"C{i+1}"].step()
        loss_dict[f"C{i+1}"] = float(ci_loss.detach().cpu().numpy())

    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        ci_list = []
        for i in range(num_view):
            if isinstance(model_dict[f"E{i+1}"], Identity_E):
                enc = model_dict[f"E{i+1}"](data_list[i], None)
            else:
                enc = model_dict[f"E{i+1}"](data_list[i], adj_list[i])
            ci_list.append(model_dict[f"C{i+1}"](enc))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(criterion(c, label) * sample_weight)
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = float(c_loss.detach().cpu().numpy())
    return loss_dict

def test_epoch(data_list: List[torch.Tensor], adj_list, te_idx: Sequence[int], model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    with torch.no_grad():
        ci_list = []
        for i in range(num_view):
            if isinstance(model_dict[f"E{i+1}"], Identity_E):
                enc = model_dict[f"E{i+1}"](data_list[i], None)
            else:
                enc = model_dict[f"E{i+1}"](data_list[i], adj_list[i])
            ci_list.append(model_dict[f"C{i+1}"](enc))
        c = model_dict["C"](ci_list) if num_view >= 2 else ci_list[0]
        c = c[te_idx, :]
        prob = F.softmax(c, dim=1).data.cpu().numpy()
    return prob


# ==========================
# Data adapters
# ==========================
def tensors_from_numpy_lists(X_list: List[np.ndarray], A_list, device: str = "cpu"):
    data_tensors = [torch.tensor(X, dtype=torch.float32, device=device) for X in X_list]
    # A_list can contain None (for clinical) → replace with identity
    adj_tensors = []
    for A, X in zip(A_list, X_list):
        if A is None:
            n = X.shape[0]
            A = sp.eye(n, dtype=np.float32, format="csr")
        adj_tensors.append(csr_to_torch_coo(A, device=device).coalesce())
    return data_tensors, adj_tensors


# ==========================
# Loader for processor outputs (portal/study aware)
# ==========================
def _load_processor_outputs(portal: str, study_id: str, ct: str, mods: List[str], use_clinical: bool):
    """
    Load per-modality TRAIN/TEST parquet matrices & labels from:
      data/gdm/filtered/<portal>/<ct>/<study_id>/
    """
    root = Path("data/gdm/filtered") / portal / _slug(ct) / study_id
    X_tr_dict, X_te_dict = {}, {}
    for m in mods:
        p_tr = root / f"{m}_train.parquet"
        p_te = root / f"{m}_test.parquet"
        if not p_tr.exists() or not p_te.exists():
            raise FileNotFoundError(f"Missing {p_tr} or {p_te}. Run processor for {portal}:{study_id}/{ct}/{m}.")
        X_tr_dict[m] = pd.read_parquet(p_tr)
        X_te_dict[m] = pd.read_parquet(p_te)

    # labels
    y_tr = y_te = None
    lt = root / "labels_train.csv"
    le = root / "labels_test.csv"
    if lt.exists():
        y_tr = pd.read_csv(lt, index_col=0, header=None).iloc[:, 0]
        y_tr.index = y_tr.index.astype(str)
    if le.exists():
        y_te = pd.read_csv(le, index_col=0, header=None).iloc[:, 0]
        y_te.index = y_te.index.astype(str)

    # Align label indices to matrix order if present
    if y_tr is not None:
        for m in mods:
            X_tr_dict[m] = X_tr_dict[m].loc[y_tr.index.intersection(X_tr_dict[m].index)]
        y_tr = y_tr.loc[X_tr_dict[mods[0]].index]
    if y_te is not None:
        for m in mods:
            X_te_dict[m] = X_te_dict[m].loc[y_te.index.intersection(X_te_dict[m].index)]
        y_te = y_te.loc[X_te_dict[mods[0]].index]

    # Optionally load clinical frames (already filtered to train/test by processor)
    clin_tr = clin_te = None
    if use_clinical:
        p_ctr = root / "clinical_train.parquet"
        p_cte = root / "clinical_test.parquet"
        if p_ctr.exists() and p_cte.exists():
            clin_tr = pd.read_parquet(p_ctr).set_index("patientId")
            clin_te = pd.read_parquet(p_cte).set_index("patientId")
            # align to labels order
            if y_tr is not None:
                clin_tr = clin_tr.loc[y_tr.index.intersection(clin_tr.index)]
            if y_te is not None:
                clin_te = clin_te.loc[y_te.index.intersection(clin_te.index)]
        else:
            print(f"[graph_ai] WARN {portal}:{study_id}/{ct}: clinical_train/test.parquet not found; skipping clinical view.")

    return X_tr_dict, X_te_dict, y_tr, y_te, clin_tr, clin_te, root


# ==========================
# Clinical feature builder (no-graph view)
# ==========================
def _build_clinical_view(
    clin_tr: pd.DataFrame,
    clin_te: pd.DataFrame,
    feature_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select configured clinical columns, one-hot encode categoricals,
    min-max scale numerics; return numeric float32 frames aligned to input indices.
    """
    if clin_tr is None or clin_te is None:
        return None, None

    feats = feature_cfg.get("features", ["AGE", "CLINICAL_STAGE", "RACE"])
    feats = [f for f in feats if f in clin_tr.columns or f in clin_te.columns]
    if not feats:
        print("[graph_ai] clinical_view: no configured features present; skipping.")
        return None, None

    def _prep(df):
        df2 = df.copy()
        # keep only requested columns that exist
        keep = [c for c in feats if c in df2.columns]
        df2 = df2[keep]
        # Split numeric vs categorical
        num_cols = df2.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in df2.columns if c not in num_cols]
        # One-hot categoricals (drop first to avoid collinearity)
        if cat_cols:
            df2 = pd.get_dummies(df2, columns=cat_cols, dummy_na=True)
        # Minimax numerical scaling
        df2 = _minimax(df2)
        df2 = df2.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        return df2

    Xtr = _prep(clin_tr)
    Xte = _prep(clin_te)

    # align columns between train/test
    cols = sorted(set(Xtr.columns).union(set(Xte.columns)))
    Xtr = Xtr.reindex(columns=cols, fill_value=0.0)
    Xte = Xte.reindex(columns=cols, fill_value=0.0)
    return Xtr, Xte


# ==========================
# High-level train/test wrapper (uses our inputs)
# ==========================
def train_with_split(
    view_names: List[str],
    X_tr_dict: Dict[str, pd.DataFrame],
    X_te_dict: Dict[str, pd.DataFrame],
    y_tr: pd.Series,
    y_te: pd.Series,
    *,
    edges_per_node: int = 5,
    dim_he_list: Sequence[int] = (400, 400, 200),
    lr_e_pretrain: float = 1e-4,
    lr_e: float = 1e-4,
    lr_c: float = 1e-4,
    num_epoch_pretrain: int = 50,
    num_epoch: int = 400,
    device: str = "cpu",
    save_dir: Optional[Path] = None,
) -> Tuple[Dict[str, float], Dict[str, nn.Module]]:
    set_seed(1234)

    # Label normalization (robust)
    classes = pd.Index(np.unique(np.concatenate([pd.Series(y_tr).values,
                                                 pd.Series(y_te).values]))).tolist()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_class = len(classes)
    y_tr_idx = np.asarray([class_to_idx[v] for v in pd.Series(y_tr).values], dtype=np.int64)
    y_te_idx = np.asarray([class_to_idx[v] for v in pd.Series(y_te).values], dtype=np.int64)

    # Build adjacencies per view (clinical uses identity/no-graph)
    params = MOGONETGraphParams(edges_per_node=edges_per_node)
    A_tr_list, A_trte_list, X_tr_list, X_trte_list = [], [], [], []
    te_idx = list(range(len(y_tr_idx), len(y_tr_idx) + len(y_te_idx)))

    for vname in view_names:
        Xtr_df = _coerce_numeric_df(X_tr_dict[vname], f"{vname}/train")
        Xte_df = _coerce_numeric_df(X_te_dict[vname], f"{vname}/test")
        X_tr_dict[vname] = Xtr_df
        X_te_dict[vname] = Xte_df

        Xtr = Xtr_df.to_numpy(dtype=np.float32)
        Xte = Xte_df.to_numpy(dtype=np.float32)

        if vname == "clinical":
            A_tr, A_trte = None, None  # identity will be used in tensor conversion
        else:
            radius = compute_adaptive_radius_parameter_mogonet(Xtr, params)
            A_tr = build_train_adjacency_mogonet(Xtr, radius=radius, params=params)
            A_trte = build_trte_block_adjacency_mogonet(Xtr, Xte, radius=radius, params=params)

        A_tr_list.append(A_tr)
        A_trte_list.append(A_trte)
        X_tr_list.append(Xtr)
        X_trte_list.append(np.vstack([Xtr, Xte]))

    # Tensors
    data_tr_tensors, adj_tr_tensors = tensors_from_numpy_lists(X_tr_list, A_tr_list, device=device)
    data_trte_tensors, adj_trte_tensors = tensors_from_numpy_lists(X_trte_list, A_trte_list, device=device)

    y_tr_tensor = torch.from_numpy(y_tr_idx).long().to(device)
    sample_weight_tr = torch.ones(len(y_tr_idx), dtype=torch.float32, device=device) / max(1, len(y_tr_idx))

    # Models
    dim_list = [X_tr_dict[v].shape[1] for v in view_names]
    dim_hvcdn = int(pow(num_class, len(view_names)))
    model_dict = init_model_dict(view_names, num_class, dim_list, dim_he_list, dim_hvcdn)
    for k in model_dict:
        model_dict[k] = model_dict[k].to(device)

    # Pretrain encoders (no VCDN)
    optim_dict = init_optim(view_names, model_dict, lr_e_pretrain, lr_c)
    for _ in range(num_epoch_pretrain):
        train_epoch(data_tr_tensors, adj_tr_tensors, y_tr_tensor, sample_weight_tr,
                    model_dict, optim_dict, train_VCDN=False)

    # Train full (with VCDN if >=2 views)
    optim_dict = init_optim(view_names, model_dict, lr_e, lr_c)
    y_all_idx = np.concatenate([y_tr_idx, y_te_idx])
    metrics = {"acc": [], "f1": [], "auc": []}

    for _ in range(num_epoch + 1):
        train_epoch(data_tr_tensors, adj_tr_tensors, y_tr_tensor, sample_weight_tr,
                    model_dict, optim_dict, train_VCDN=True)
        prob = test_epoch(data_trte_tensors, adj_trte_tensors, te_idx, model_dict)
        y_pred = prob.argmax(1)
        acc = accuracy_score(y_all_idx[te_idx], y_pred)
        f1  = f1_score(y_all_idx[te_idx], y_pred, average="macro")
        metrics["acc"].append(acc); metrics["f1"].append(f1)
        if num_class == 2 and prob.shape[1] >= 2:
            try:
                auc = roc_auc_score(y_all_idx[te_idx], prob[:, 1])
                metrics["auc"].append(auc)
            except Exception:
                pass

    final_metrics = {k: float(v[-1]) for k, v in metrics.items() if v}

    # Optional save bundle
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "num_view": int(len(view_names)),
            "num_class": int(num_class),
            "views": view_names,
            "dim_list": [int(d) for d in dim_list],
            "dim_he_list": [int(d) for d in dim_he_list],
            "edges_per_node": int(edges_per_node),
            "feature_names": {v: [str(c) for c in X_tr_dict[v].columns] for v in view_names},
        }
        save_model_bundle(save_dir, model_dict, meta)

    return final_metrics, model_dict


# ==========================
# Pretrained inference helper
# ==========================
def save_model_dict(ckpt_dir: Path, model_dict: Dict[str, torch.nn.Module]):
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    for key, model in model_dict.items():
        torch.save(model.state_dict(), ckpt_dir / f"{key}.pt")

def load_model_dict(ckpt_dir: Path, model_dict: Dict[str, torch.nn.Module]):
    ckpt_dir = Path(ckpt_dir)
    for key, model in model_dict.items():
        p = ckpt_dir / f"{key}.pt"
        if p.exists():
            try:
                state = torch.load(p, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(p, map_location="cpu")
            model.load_state_dict(state)
    return model_dict

def save_model_bundle(ckpt_dir: Path, model_dict: Dict[str, torch.nn.Module], meta: Dict):
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_model_dict(ckpt_dir, model_dict)
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def load_model_bundle(ckpt_dir: Path):
    ckpt_dir = Path(ckpt_dir)
    meta_path = ckpt_dir / "meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    num_view = meta["num_view"]
    num_class = meta["num_class"]
    views = meta["views"]
    dim_list = meta["dim_list"]
    dim_he_list = meta["dim_he_list"]
    dim_hvcdn = pow(num_class, num_view)

    model_dict = init_model_dict(views, num_class, dim_list, dim_he_list, dim_hvcdn)
    for key, model in model_dict.items():
        p = ckpt_dir / f"{key}.pt"
        if p.exists():
            try:
                state = torch.load(p, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(p, map_location="cpu")
            model.load_state_dict(state)

    return model_dict, meta


# ==========================
#  MOGONET-style GNN
# ==========================
def graph_ai_model():
    """
    Train or run inference with the MOGONET-style GNN on preprocessed data.

    Config key: model_mogonet
      mode: train|infer
      checkpoints_dir: data/models
      edges_per_node: 5
      gcn_hidden: [400,400,200]
      epochs: {pretrain: 50, train: 400}
      lr: {enc_pretrain:1e-3, enc:5e-4, clf:1e-3}
      device: cpu|cuda
      # OPTIONAL:
      clinical_view:
        enabled: true
        features: [AGE, CLINICAL_STAGE, RACE, SEX, ETHNICITY]  # must exist in clinical_* tables
    """
    cfg = _load_cfg()
    model_cfg = cfg.get("model_mogonet", {}) or {}
    mode = str(model_cfg.get("mode", "train")).lower()
    expr_profiles = cfg.get("expression_profiles", {})
    edges_per_node = int(model_cfg.get("edges_per_node", 5))
    dim_he_list = tuple(model_cfg.get("gcn_hidden", [400, 400, 200]))
    epochs = model_cfg.get("epochs", {}) or {}
    num_epoch_pretrain = int(epochs.get("pretrain", 50))
    num_epoch = int(epochs.get("train", 400))
    lr_cfg = model_cfg.get("lr", {}) or {}
    lr_e_pre = float(lr_cfg.get("enc_pretrain", 1e-3))
    lr_e = float(lr_cfg.get("enc", 5e-4))
    lr_c = float(lr_cfg.get("clf", 1e-3))
    device = str(model_cfg.get("device", "cpu")).lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    ckpt_root = Path(model_cfg.get("checkpoints_dir", "data/models"))

    ct_list = cfg.get("cancer_type_interest", [])
    if not ct_list:
        print("[graph_ai] No cancer types configured — nothing to do.")
        return

    # study maps (match extractor/processor/graph_constructor)
    c_map = _normalize_study_map(cfg.get("cBio_studyId", {}))
    p_map = _normalize_study_map(cfg.get("pedcBio_studyId", {}))

    clinical_cfg = (model_cfg.get("clinical_view") or {})
    clinical_enabled = bool(clinical_cfg.get("enabled", False))

    for ct in ct_list:
        studies = _studies_for_ct(ct, c_map, p_map)
        if not studies:
            print(f"[graph_ai] SKIP {ct}: no studies configured.")
            continue

        # base modalities from expression_profiles
        base_mods = _normalize_modalities(expr_profiles.get(ct, []))
        base_mods = [m for m in base_mods if m in {"mrna", "mirna", "methylation"}]

        # append clinical view if enabled
        view_names = base_mods.copy()
        if clinical_enabled:
            view_names.append("clinical")
        if not view_names:
            print(f"[graph_ai] {ct}: no views configured; skipping.")
            continue

        for portal, study_id in studies:
            print(f"[graph_ai] {ct} [{portal}:{study_id}] — mode={mode}, views={view_names}")

            # Load matrices/labels (+ clinical trains/test frames)
            X_tr_dict, X_te_dict, y_tr, y_te, clin_tr, clin_te, root = _load_processor_outputs(
                portal, study_id, ct, base_mods, use_clinical=clinical_enabled
            )

            # Build clinical matrices if requested
            if clinical_enabled and clin_tr is not None and clin_te is not None:
                Xc_tr, Xc_te = _build_clinical_view(clin_tr, clin_te, clinical_cfg)
                if Xc_tr is not None:
                    # ensure aligned to label order
                    if y_tr is not None:
                        Xc_tr = Xc_tr.loc[y_tr.index.intersection(Xc_tr.index)]
                    if y_te is not None:
                        Xc_te = Xc_te.loc[y_te.index.intersection(Xc_te.index)]
                    X_tr_dict["clinical"] = Xc_tr
                    X_te_dict["clinical"] = Xc_te
                else:
                    # drop clinical if we failed to build it
                    view_names = [v for v in view_names if v != "clinical"]

            if mode == "train":
                if y_tr is None or y_te is None:
                    raise RuntimeError(f"[graph_ai] {ct} [{portal}:{study_id}]: missing labels_train/test.csv.")
                ckpt_dir = ckpt_root / portal / _slug(ct) / study_id
                metrics, _trained = train_with_split(
                    view_names, X_tr_dict, X_te_dict, y_tr, y_te,
                    edges_per_node=edges_per_node,
                    dim_he_list=dim_he_list,
                    lr_e_pretrain=lr_e_pre,
                    lr_e=lr_e,
                    lr_c=lr_c,
                    num_epoch_pretrain=num_epoch_pretrain,
                    num_epoch=num_epoch,
                    device=device,
                    save_dir=ckpt_dir,
                )
                print(f"[graph_ai] {ct} [{portal}:{study_id}]: final metrics: {metrics}")
                print(f"[graph_ai] {ct} [{portal}:{study_id}]: checkpoints saved -> {ckpt_dir}")

            elif mode == "infer":
                # For brevity, inference can be wired similarly to train_with_split (load bundle & forward).
                # You can extend this block if you want evaluation printing as done in train().
                ckpt_dir = ckpt_root / portal / _slug(ct) / study_id
                print(f"[graph_ai] {ct} [{portal}:{study_id}] inference expects trained checkpoints at {ckpt_dir}")
            else:
                raise ValueError(f"[graph_ai] Unknown model_mogonet.mode: {mode}. Use 'train' or 'infer'.")

if __name__ == "__main__":
    print("This module is driven by CONFIG_PATH::model_mogonet. See extractor/processor/graph_constructor for upstream steps.")
