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

"""
from __future__ import annotations

import copy
import os
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import yaml
from scipy import sparse as sp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
import yaml
from scipy import sparse as sp

def _load_cfg() -> dict:
    cfg_path = os.getenv("CONFIG_PATH", "config/pipeline_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

def _slug(s: str) -> str:
    return s.replace(" ", "_")

def _normalize_modalities(mods):
    syn = {"microrna": "mirna"}
    return [syn.get(m.lower(), m.lower()) for m in (mods or [])]

def _load_processor_outputs(ct: str, mods: List[str]):
    """Load per-modality TRAIN/TEST parquet matrices and label CSVs from processor.py outputs."""
    root = Path("data/gdm/filtered") / _slug(ct)
    X_tr_dict, X_te_dict = {}, {}
    for m in mods:
        p_tr = root / f"{m}_train.parquet"
        p_te = root / f"{m}_test.parquet"
        if not p_tr.exists() or not p_te.exists():
            raise FileNotFoundError(f"Missing {p_tr} or {p_te}. Run 'process' first for {ct}/{m}.")
        X_tr_dict[m] = pd.read_parquet(p_tr)
        X_te_dict[m] = pd.read_parquet(p_te)
    # labels (optional but expected in train mode)
    y_tr = None; y_te = None
    lt = root / "labels_train.csv"
    le = root / "labels_test.csv"
    if lt.exists():
        y_tr = pd.read_csv(lt, index_col=0, header=None).iloc[:,0]
        y_tr.index = y_tr.index.astype(str)
    if le.exists():
        y_te = pd.read_csv(le, index_col=0, header=None).iloc[:,0]
        y_te.index = y_te.index.astype(str)
    # align label indices to matrix order if present
    if y_tr is not None:
        for m in mods:
            X_tr_dict[m] = X_tr_dict[m].loc[y_tr.index.intersection(X_tr_dict[m].index)]
        y_tr = y_tr.loc[X_tr_dict[mods[0]].index]
    if y_te is not None:
        for m in mods:
            X_te_dict[m] = X_te_dict[m].loc[y_te.index.intersection(X_te_dict[m].index)]
        y_te = y_te.loc[X_te_dict[mods[0]].index]
    return X_tr_dict, X_te_dict, y_tr, y_te

def _maybe_load_prebuilt_graphs(ct: str, mods: List[str]):
    """Optional: load prebuilt adjacencies written by graph_constructor.py (not strictly required)."""
    root = Path("data/gdm/graphs") / _slug(ct)
    A_tr_list, A_trte_list = [], []
    ok = True
    for m in mods:
        p_tr = root / f"{m}_train.adj.npz"
        p_trte = root / f"{m}_trte_block.adj.npz"
        if not p_tr.exists() or not p_trte.exists():
            ok = False
            break
        A_tr_list.append(sp.load_npz(p_tr))
        A_trte_list.append(sp.load_npz(p_trte))
    return (A_tr_list, A_trte_list) if ok else (None, None)

def _device_from_cfg(model_cfg: dict) -> str:
    dev = str(model_cfg.get("device", "cpu")).lower()
    if dev == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

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
        # concat via outer-product chain as in MOGONET
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

def init_model_dict(num_view: int, num_class: int, dim_list: Sequence[int], dim_he_list: Sequence[int], dim_hc: int, gcn_dropout: float = 0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict[f"E{i+1}"] = GCN_E(dim_list[i], dim_he_list, gcn_dropout)
        model_dict[f"C{i+1}"] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view: int, model_dict: Dict[str, nn.Module], lr_e: float = 1e-4, lr_c: float = 1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict[f"C{i+1}"] = torch.optim.Adam(
            list(model_dict[f"E{i+1}"].parameters()) + list(model_dict[f"C{i+1}"].parameters()), lr=lr_e
        )
    if num_view >= 2:
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
        ci = model_dict[f"C{i+1}"](model_dict[f"E{i+1}"](data_list[i], adj_list[i]))
        ci_loss = torch.mean(criterion(ci, label) * sample_weight)
        ci_loss.backward()
        optim_dict[f"C{i+1}"].step()
        loss_dict[f"C{i+1}"] = float(ci_loss.detach().cpu().numpy())

    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        ci_list = [model_dict[f"C{i+1}"](model_dict[f"E{i+1}"](data_list[i], adj_list[i])) for i in range(num_view)]
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
        ci_list = [model_dict[f"C{i+1}"](model_dict[f"E{i+1}"](data_list[i], adj_list[i])) for i in range(num_view)]
        c = model_dict["C"](ci_list) if num_view >= 2 else ci_list[0]
        c = c[te_idx, :]
        prob = F.softmax(c, dim=1).data.cpu().numpy()
    return prob


# ==========================
# Data adapters
# ==========================

def tensors_from_numpy_lists(X_list: List[np.ndarray], A_list, device: str = "cpu"):
    data_tensors = [torch.tensor(X, dtype=torch.float32, device=device) for X in X_list]
    adj_tensors = [csr_to_torch_coo(A, device=device).coalesce() for A in A_list]
    return data_tensors, adj_tensors


# ==========================
# High-level train/test wrapper (uses our inputs)
# ==========================

def train_with_split(
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
    """
    Train GNN with given split and return (final_metrics, model_dict).

    - Labels are normalized to contiguous integer indices across BOTH train & test,
      fixing dtype mismatches (e.g., '0'/'1' vs 0/1).
    - Feature frames are coerced to numeric float32 (non-numeric columns dropped).
    - If save_dir is provided, writes:
        <save_dir>/*.pt       (state_dicts for E{i}, C{i}, and C if fused)
        <save_dir>/meta.json  (architecture + preprocessing + label mapping)
    """
    set_seed(1234)
    views = list(X_tr_dict.keys())
    num_view = len(views)

    # -----------------------------
    # Label normalization (robust)
    # -----------------------------
    # Build a mapping across BOTH y_tr and y_te to ensure consistent indices
    classes = pd.Index(np.unique(np.concatenate([pd.Series(y_tr).values,
                                                 pd.Series(y_te).values]))).tolist()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_class = len(classes)

    y_tr_idx = np.asarray([class_to_idx[v] for v in pd.Series(y_tr).values], dtype=np.int64)
    y_te_idx = np.asarray([class_to_idx[v] for v in pd.Series(y_te).values], dtype=np.int64)

    # -----------------------------
    # Build adjacencies per view
    # -----------------------------
    params = MOGONETGraphParams(edges_per_node=edges_per_node)
    A_tr_list, A_trte_list, X_tr_list, X_trte_list = [], [], [], []
    # indices in the concatenated [train | test] order
    te_idx = list(range(len(y_tr_idx), len(y_tr_idx) + len(y_te_idx)))

    for m in views:
        # Ensure numeric float32 matrices; update dicts in-place so shapes are correct downstream
        Xtr_df = _coerce_numeric_df(X_tr_dict[m], f"{m}/train")
        Xte_df = _coerce_numeric_df(X_te_dict[m], f"{m}/test")
        X_tr_dict[m] = Xtr_df
        X_te_dict[m] = Xte_df

        Xtr = Xtr_df.to_numpy(dtype=np.float32)
        Xte = Xte_df.to_numpy(dtype=np.float32)

        # MOGONET threshold from TRAIN, applied to TRAIN and TRTE
        radius = compute_adaptive_radius_parameter_mogonet(Xtr, params)
        A_tr   = build_train_adjacency_mogonet(Xtr, radius=radius, params=params)
        A_trte = build_trte_block_adjacency_mogonet(Xtr, Xte, radius=radius, params=params)

        A_tr_list.append(A_tr); A_trte_list.append(A_trte)
        X_tr_list.append(Xtr); X_trte_list.append(np.vstack([Xtr, Xte]))

    # -----------------------------
    # Tensors
    # -----------------------------
    data_tr_tensors, adj_tr_tensors = tensors_from_numpy_lists(X_tr_list, A_tr_list, device=device)
    data_trte_tensors, adj_trte_tensors = tensors_from_numpy_lists(X_trte_list, A_trte_list, device=device)

    y_tr_tensor = torch.from_numpy(y_tr_idx).long().to(device)
    sample_weight_tr = torch.ones(len(y_tr_idx), dtype=torch.float32, device=device) / max(1, len(y_tr_idx))

    # -----------------------------
    # Models
    # -----------------------------
    dim_list  = [X_tr_dict[m].shape[1] for m in views]
    dim_hvcdn = int(pow(num_class, num_view))
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for k in model_dict:
        model_dict[k] = model_dict[k].to(device)

    # -----------------------------
    # Pretrain encoders (no VCDN)
    # -----------------------------
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for _ in range(num_epoch_pretrain):
        train_epoch(data_tr_tensors, adj_tr_tensors, y_tr_tensor, sample_weight_tr,
                    model_dict, optim_dict, train_VCDN=False)

    # -----------------------------
    # Train full (with VCDN if num_view >= 2)
    # -----------------------------
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
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
                # AUC can fail on degenerate predictions; ignore
                pass

    final_metrics = {k: float(v[-1]) for k, v in metrics.items() if v}

    # -----------------------------
    # Optional save bundle
    # -----------------------------
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "num_view": int(num_view),
            "num_class": int(num_class),
            "views": views,
            "dim_list": [int(d) for d in dim_list],
            "dim_he_list": [int(d) for d in dim_he_list],
            "edges_per_node": int(edges_per_node),
            "feature_names": {m: [str(c) for c in X_tr_dict[m].columns] for m in views},
        }
        save_model_bundle(save_dir, model_dict, meta)

    return final_metrics, model_dict


# ==========================
# Pretrained inference helper
# ==========================

def save_model_dict(ckpt_dir: Path, model_dict: Dict[str, torch.nn.Module]):
    """
    Save one .pt per submodule (E1, C1, ... , C) using state_dict (portable).
    """
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    for key, model in model_dict.items():
        torch.save(model.state_dict(), ckpt_dir / f"{key}.pt")

def load_model_dict(ckpt_dir: Path, model_dict: Dict[str, torch.nn.Module]):
    """
    Load state_dicts into an already-constructed model_dict.
    """
    ckpt_dir = Path(ckpt_dir)
    for key, model in model_dict.items():
        p = ckpt_dir / f"{key}.pt"
        if p.exists():
            try:
                state = torch.load(p, map_location="cpu", weights_only=True)
            except TypeError:
                # Older PyTorch that doesn't support weights_only yet
                state = torch.load(p, map_location="cpu")
            model.load_state_dict(state)
    return model_dict

def save_model_bundle(ckpt_dir: Path, model_dict: Dict[str, torch.nn.Module], meta: Dict):
    """
    Save the full bundle:
      - weights:  E1.pt, C1.pt, [E2.pt, C2.pt, ...], [C.pt (fusion)]
      - meta.json: architecture, label mapping, preprocessing params, etc.
    """
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_model_dict(ckpt_dir, model_dict)
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def load_model_bundle(ckpt_dir: Path):
    import json, torch
    ckpt_dir = Path(ckpt_dir)
    meta_path = ckpt_dir / "meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # reconstruct model skeleton from meta (same way you do during train)
    num_view    = meta["num_view"]
    num_class   = meta["num_class"]
    views       = meta["views"]
    dim_list    = meta["dim_list"]
    dim_he_list = meta["dim_he_list"]
    dim_hvcdn   = pow(num_class, num_view)

    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)

    # load weights safely
    for key, model in model_dict.items():
        p = ckpt_dir / f"{key}.pt"
        if p.exists():
            try:
                state = torch.load(p, map_location="cpu", weights_only=True)  # safe path
            except TypeError:
                state = torch.load(p, map_location="cpu")                    # PyTorch < 2.4 fallback
            model.load_state_dict(state)

    return model_dict, meta

# ==========================
#  MOGONET-style GNN
# ==========================
def graph_ai_model():
    """
    Train or run inference with the MOGONET-style GNN on preprocessed data.

    Config (pipeline_config.yaml):
      model:
        mode: train | infer
        cancer_types: null            # optional; defaults to cancer_type_interest
        modalities:
          "Wilms Tumor": [mrna, mirna, methylation]
        edges_per_node: 5
        gcn_hidden: [400, 400, 200]
        epochs:
          pretrain: 50
          train: 400
        lr:
          enc_pretrain: 1.0e-3
          enc: 5.0e-4
          clf: 1.0e-3
        device: cpu                   # or cuda (if available)
        checkpoints_dir: data/models  # per-cancer subfolders
    """
    cfg = _load_cfg()
    model_cfg = cfg.get("model", {}) or {}
    mode = str(model_cfg.get("mode", "train")).lower()
    ct_list = model_cfg.get("cancer_types") or cfg.get("cancer_type_interest", [])
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
    device = _device_from_cfg(model_cfg)
    ckpt_root = Path(model_cfg.get("checkpoints_dir", "data/models"))

    if not ct_list:
        print("[graph_ai] No cancer types configured — nothing to do.")
        return

    for ct in ct_list:
        # which modalities to use for this cancer type
        mods = _normalize_modalities(expr_profiles.get(ct, []))
        mods = [m for m in mods if m in {"mrna", "mirna", "methylation"}]
        if not mods:
            print(f"[graph_ai] {ct}: no valid bulk modalities configured; skipping.")
            continue

        print(f"[graph_ai] {ct} — mode={mode}, views={mods}")

        # Load data prepared by processor.py
        X_tr_dict, X_te_dict, y_tr, y_te = _load_processor_outputs(ct, mods)

        # Assemble / init model dict
        num_view = len(mods)
        if mode == "train":
            if y_tr is None or y_te is None:
                raise RuntimeError(f"[graph_ai] {ct}: labels_train.csv / labels_test.csv not found in filtered outputs.")
            ckpt_dir = ckpt_root / _slug(ct)
            metrics, trained = train_with_split(
                X_tr_dict, X_te_dict, y_tr, y_te,
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
            print(f"[graph_ai] {ct}: final metrics: {metrics}")
            print(f"[graph_ai] {ct}: checkpoints saved to {ckpt_dir}")
            
            # Save weights
            dim_list = [X_tr_dict[m].shape[1] for m in mods]
            num_class = len(np.unique(y_tr))
            dim_hvcdn = pow(num_class, num_view)
            model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
            for k in model_dict:
                model_dict[k] = model_dict[k].to(device)
          

        elif mode == "infer":
            # Build a model skeleton, load checkpoints if available, then run test forward pass.
            # If checkpoints are absent, we’ll just warn and exit.
            dim_list = [X_tr_dict[m].shape[1] for m in mods]
            # If no labels, assume binary for shape; otherwise derive from test labels
            if y_tr is not None:
                num_class = len(np.unique(y_tr))
            elif y_te is not None:
                num_class = len(np.unique(y_te))
            else:
                num_class = 2
            dim_hvcdn = pow(num_class, num_view)

            model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
            for k in model_dict:
                model_dict[k] = model_dict[k].to(device)

            ckpt_dir = ckpt_root / _slug(ct)
            if not ckpt_dir.exists():
                print(f"[graph_ai] {ct}: checkpoints not found at {ckpt_dir}; aborting inference.")
                continue
            load_model_dict(ckpt_dir, model_dict)

            # Build TR and TRTE adjacencies then run one inference pass
            params = MOGONETGraphParams(edges_per_node=edges_per_node)
            X_tr_list, X_trte_list, A_tr_list, A_trte_list = [], [], [], []
            for m in mods:
                Xtr = X_tr_dict[m].to_numpy(dtype=np.float32)
                Xte = X_te_dict[m].to_numpy(dtype=np.float32)
                radius = compute_adaptive_radius_parameter_mogonet(Xtr, params)
                A_tr = build_train_adjacency_mogonet(Xtr, radius=radius, params=params)
                A_trte = build_trte_block_adjacency_mogonet(Xtr, Xte, radius=radius, params=params)
                X_tr_list.append(Xtr); X_trte_list.append(np.vstack([Xtr, Xte]))
                A_tr_list.append(A_tr); A_trte_list.append(A_trte)

            data_trte_tensors, adj_trte_tensors = tensors_from_numpy_lists(X_trte_list, A_trte_list, device=device)
            te_idx = list(range(len(X_tr_dict[mods[0]]), len(X_tr_dict[mods[0]]) + len(X_te_dict[mods[0]])))
            # Fake a minimal train tensor to compile modules (we only need forward on TRTE)
            # But our test_epoch expects model_dict already trained – here we just forward.
            with torch.no_grad():
                ci_list = [model_dict[f"C{i+1}"](model_dict[f"E{i+1}"](data_trte_tensors[i], adj_trte_tensors[i]))
                           for i in range(num_view)]
                c = model_dict["C"](ci_list) if num_view >= 2 else ci_list[0]
                c = c[te_idx, :]
                prob = F.softmax(c, dim=1).data.cpu().numpy()

            if y_te is not None:
                y_true = y_te.values
                acc = accuracy_score(y_true, prob.argmax(1))
                f1 = f1_score(y_true, prob.argmax(1), average="macro")
                msg = {"acc": float(acc), "f1": float(f1)}
                if prob.shape[1] == 2:
                    try:
                        auc = roc_auc_score(y_true, prob[:, 1])
                        msg["auc"] = float(auc)
                    except Exception:
                        pass
                print(f"[graph_ai] {ct}: inference metrics: {msg}")
            else:
                print(f"[graph_ai] {ct}: inference probabilities shape: {prob.shape}")

        else:
            raise ValueError(f"[graph_ai] Unknown model.mode: {mode}. Use 'train' or 'infer'.")

if __name__ == "__main__":
    print("This module is imported by src/main.py; see processor.py and graph_constructor.py for data prep.")
