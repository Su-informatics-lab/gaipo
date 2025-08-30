"""
graph_constructor.py — Graph builders for multi-omics pipelines

Scope
-----
This module currently implements **patient-similarity graphs for bulk omics** (MOGONET-style).
It also defines **empty placeholders** for **spatial single-cell** graph construction; only
the step names are provided for now.

───────────────────────────────────────────────────────────────────────────────
A) Bulk omics (patient-similarity)
───────────────────────────────────────────────────────────────────────────────
Goal
  Build adjacency matrices over patients using cosine similarity on preprocessed,
  per-modality feature matrices (rows = patients, cols = features). This follows
  the spirit of MOGONET:
    • Fit selectors/scalers upstream on TRAIN only (done in processor.py)
    • TRAIN graph: edges selected by an adaptive “radius” so each node has about
      `edges_per_node` neighbors (cosine distance thresholding)
    • TRTE block graph: build cross edges (TRAIN↔TEST) using the TRAIN threshold,
      with no TEST↔TEST edges
    • Weights = cosine similarity (1 − cosine distance)
    • Symmetrize with max(A, Aᵀ), add self-loops, then row-normalize

Inputs (produced by processor.py)
  data/gdm/filtered/<Cancer_Type>/<mod>_train.parquet
  data/gdm/filtered/<Cancer_Type>/<mod>_test.parquet
  where <mod> ∈ {mrna, mirna, methylation}

Outputs
  data/gdm/graphs/<Cancer_Type>/<mod>_train.adj.npz         # CSR adjacency for TRAIN
  data/gdm/graphs/<Cancer_Type>/<mod>_trte_block.adj.npz    # CSR adjacency for TRAIN|TEST block
  (optional) .graphml exports for inspection

Config (config/pipeline_config.yaml)
  graph:
    edges_per_node: 15     # neighbor budget used to derive the cosine distance cutoff on TRAIN
    save_graphml: false    # write GraphML alongside .npz if true

  cancer_type_interest: ["Wilms Tumor", ...]
  expression_profiles:
    "Wilms Tumor": [mrna, mirna, methylation]

Pipeline entrypoint (config-driven)
  graph_construct()
    • Loops over configured cancer types and modalities
    • Loads *_train.parquet and *_test.parquet
    • Computes TRAIN threshold; writes <mod>_train.adj.npz and <mod>_trte_block.adj.npz
    • Optional GraphML if graph.save_graphml=true

CLI utilities (ad-hoc)
  python src/graph_constructor.py bulk-train \
    --matrix data/gdm/filtered/mrna_train.parquet \
    --format parquet \
    --edges-per-node 5 \
    --out data/gdm/graphs/mrna_train.adj.npz \
    --out-graphml data/gdm/graphs/mrna_train.graphml

  python src/graph_constructor.py bulk-split \
    --matrix-train data/gdm/filtered/mrna_train.parquet \
    --matrix-test  data/gdm/filtered/mrna_test.parquet \
    --format parquet \
    --edges-per-node 5 \
    --out-train data/gdm/graphs/mrna_train.adj.npz \
    --out-trte  data/gdm/graphs/mrna_trte_block.adj.npz

Run via main.py
  python -m src.main --call graph_construct
  # Or chained:
  python -m src.main --call data_fetch,data_extract,process,graph_construct

Notes
  • Modality synonyms are normalized (e.g., “microrna” → “mirna”).
  • If a modality’s train/test parquet is missing, that modality is skipped for that cancer type.
  • Adjacencies are standard SciPy CSR saved with scipy.sparse.save_npz; they can be read in
    PyTorch Geometric after converting to edge_index (helper provided).

Key functions (bulk)
  compute_adaptive_radius_parameter_mogonet(X_tr, params)
  build_train_adjacency_mogonet(X_tr, radius=None, params)
  build_trte_block_adjacency_mogonet(X_tr, X_te, radius, params)
  graph_construct()   # config-driven pipeline step

Helpers
  csr_to_edge_index(A)              # -> (edge_index, weights) numpy arrays
  csr_to_torch_coo(A, device=None)  # -> torch.sparse_coo_tensor
  csr_to_networkx(A, node_ids=None) # -> networkx Graph
  save_adjacency_npz(A, out_path)
  save_graph_graphml(G, out_path)

───────────────────────────────────────────────────────────────────────────────
B) Spatial single-cell graphs (placeholders) 
───────────────────────────────────────────────────────────────────────────────


"""

from __future__ import annotations
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances
import torch
import networkx as nx

ArrayLike = Union[np.ndarray, pd.DataFrame]


# ==========================
# Parameters
# ==========================
@dataclass
class MOGONETGraphParams:
    edges_per_node: int = 5
    include_self_loops: bool = True
    row_normalize: bool = True
    metric: str = "cosine"  # Only cosine supported for faithful MOGONET behavior


# ==========================
# Core helpers
# ==========================

def _ensure_samples_by_features(X: ArrayLike) -> Tuple[np.ndarray, Optional[Sequence[str]]]:
    """Ensure X is (n_samples, n_features). If DataFrame, rows are samples; return (array, index_labels)."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), list(map(str, X.index))
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    return X, None


def _cosine_dist(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    return pairwise_distances(X, Y=Y, metric="cosine")


def _similarity(D: np.ndarray) -> np.ndarray:
    return 1.0 - D


def _sym_max(A: sp.csr_matrix) -> sp.csr_matrix:
    return A.maximum(A.T)


def _row_norm(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(A.sum(axis=1)).ravel()
    d[d == 0] = 1.0
    return sp.diags(1.0 / d) @ A


def _postprocess(A: sp.csr_matrix, include_self_loops: bool, row_normalize: bool) -> sp.csr_matrix:
    A = _sym_max(A)
    if include_self_loops:
        A = A + sp.eye(A.shape[0], dtype=A.dtype, format="csr")
    if row_normalize:
        A = _row_norm(A)
    return A.tocsr()

def _load_cfg() -> dict:
    cfg_path = os.getenv("CONFIG_PATH", "config/pipeline_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

def _normalize_modalities(mods):
    # normalize synonyms from extractor/processor
    syn = {"microrna": "mirna"}
    return [syn.get(m.lower(), m.lower()) for m in (mods or [])]


# ==========================
# MOGONET-style threshold & graphs
# ==========================

def compute_adaptive_radius_parameter_mogonet(X_tr: ArrayLike, params: MOGONETGraphParams = MOGONETGraphParams()) -> float:
    Xtr, _ = _ensure_samples_by_features(X_tr)
    n = Xtr.shape[0]
    if n < 2:
        raise ValueError("Need >=2 training samples to compute a threshold")
    D = _cosine_dist(Xtr)
    flat = np.sort(D.reshape(-1))
    idx = int(np.clip(params.edges_per_node * n, 0, flat.size - 1))
    return float(flat[idx])


def build_train_adjacency_mogonet(
    X_tr: ArrayLike,
    *,
    radius: Optional[float] = None,
    params: MOGONETGraphParams = MOGONETGraphParams(),
) -> sp.csr_matrix:
    Xtr, _ = _ensure_samples_by_features(X_tr)
    n = Xtr.shape[0]
    if radius is None:
        radius = compute_adaptive_radius_parameter_mogonet(Xtr, params)

    D = _cosine_dist(Xtr)
    keep = (D <= radius)
    np.fill_diagonal(keep, False)

    S = _similarity(D)
    r, c = np.where(keep)
    data = S[r, c]
    A = sp.csr_matrix((data, (r, c)), shape=(n, n))

    return _postprocess(A, params.include_self_loops, params.row_normalize)


def build_trte_block_adjacency_mogonet(
    X_tr: ArrayLike,
    X_te: ArrayLike,
    *,
    radius: float,
    params: MOGONETGraphParams = MOGONETGraphParams(),
) -> sp.csr_matrix:
    Xtr, _ = _ensure_samples_by_features(X_tr)
    Xte, _ = _ensure_samples_by_features(X_te)
    n_tr, n_te = Xtr.shape[0], Xte.shape[0]
    if n_tr < 1 or n_te < 1:
        raise ValueError("Both train and test must be non-empty for block adjacency")

    D_tr_te = _cosine_dist(Xtr, Xte)
    S_tr_te = _similarity(D_tr_te)

    # threshold for cross edges only (train<->test)
    keep_tr_te = (D_tr_te <= radius)
    r1, c1 = np.where(keep_tr_te)
    d1 = S_tr_te[r1, c1]
    A_tr_te = sp.csr_matrix((d1, (r1, c1)), shape=(n_tr, n_te))

    # assemble block [[0, A_tr_te],[A_tr_te^T, 0]] then postprocess
    tl = sp.csr_matrix((n_tr, n_tr))
    br = sp.csr_matrix((n_te, n_te))
    A_block = sp.bmat([[tl, A_tr_te], [A_tr_te.T, br]], format="csr")

    return _postprocess(A_block, params.include_self_loops, params.row_normalize)


# ==========================
# Torch / PyG conversions
# ==========================

def csr_to_torch_coo(A: sp.csr_matrix, device: Optional[str] = None):
    if torch is None:
        raise ImportError("torch is not available. Install PyTorch to use csr_to_torch_coo().")
    coo = A.tocoo()
    idx = np.vstack([coo.row, coo.col])
    i = torch.as_tensor(idx, dtype=torch.long, device=device)
    v = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(i, v, size=coo.shape, device=device)


def csr_to_edge_index(A: sp.csr_matrix):
    coo = A.tocoo()
    return np.vstack([coo.row, coo.col]), coo.data


# ==========================
# NetworkX conversions
# ==========================

def csr_to_networkx(
    A: sp.csr_matrix,
    node_ids: Optional[Sequence[str]] = None,
    *,
    directed: bool = False,
):
    """Convert CSR adjacency to a NetworkX Graph (edge attr 'weight')."""
    if nx is None:
        raise ImportError("networkx is not available. Install networkx to use csr_to_networkx().")

    create_using = nx.DiGraph if directed else nx.Graph
    try:
        G = nx.from_scipy_sparse_array(A, edge_attribute="weight", create_using=create_using)
    except Exception:
        G = nx.from_scipy_sparse_matrix(A, edge_attribute="weight", create_using=create_using())

    if node_ids is not None:
        if len(node_ids) != A.shape[0]:
            raise ValueError("node_ids length must equal number of rows in A")
        mapping = {i: str(node_ids[i]) for i in range(len(node_ids))}
        G = nx.relabel_nodes(G, mapping)
    return G


# ==========================
# I/O helpers
# ==========================

def save_adjacency_npz(A: sp.csr_matrix, out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_path, A.tocsr())
    return out_path


def save_graph_graphml(G, out_path: Union[str, Path]) -> Path:
    import networkx as nx
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, out_path)
    return out_path


def _load_matrix(path: str, fmt: str) -> pd.DataFrame:
    fmt = fmt.lower()
    if fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "csv":
        return pd.read_csv(path, sep=",", index_col=0)
    elif fmt == "tsv":
        return pd.read_csv(path, sep="\t", index_col=0)
    else:
        raise ValueError("Unsupported format. Choose from: parquet, csv, tsv.")

def graph_construct():
    """
    Build MOGONET-style patient-similarity graphs from preprocessed matrices.

    Reads:
      data/gdm/filtered/<Cancer_Type>/<mod>_train.parquet
      data/gdm/filtered/<Cancer_Type>/<mod>_test.parquet

    Writes:
      data/gdm/graphs/<Cancer_Type>/<mod>_train.adj.npz
      data/gdm/graphs/<Cancer_Type>/<mod>_trte_block.adj.npz
      (optional) .graphml if enabled in config.graph.save_graphml
    """
    cfg = _load_cfg()

    ct_list = cfg.get("cancer_type_interest", [])
    expr = cfg.get("expression_profiles", {})
    graph_cfg = (cfg.get("graph", {}) or {})
    edges_per_node = int(graph_cfg.get("edges_per_node", 15))
    save_graphml = bool(graph_cfg.get("save_graphml", False))

    filtered_root = Path("data/gdm/filtered")
    graphs_root = Path("data/gdm/graphs")

    params = MOGONETGraphParams(edges_per_node=edges_per_node)

    if not ct_list:
        print("[graph] No cancer_type_interest configured — nothing to do.")
        return

    for ct in ct_list:
        ct_slug = ct.replace(" ", "_")
        ct_filtered = filtered_root / ct_slug
        if not ct_filtered.exists():
            print(f"[graph] SKIP {ct}: {ct_filtered} not found (run process step first).")
            continue

        # modalities configured for this cancer type (normalize)
        modalities = _normalize_modalities(expr.get(ct, []))
        # keep only bulk modalities we can build graphs for
        modalities = [m for m in modalities if m in {"mrna", "mirna", "methylation"}]
        if not modalities:
            print(f"[graph] SKIP {ct}: no valid bulk modalities configured.")
            continue

        out_dir = graphs_root / ct_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        for mod in modalities:
            tr_pq = ct_filtered / f"{mod}_train.parquet"
            te_pq = ct_filtered / f"{mod}_test.parquet"

            if not tr_pq.exists() or not te_pq.exists():
                print(f"[graph] SKIP {ct}/{mod}: missing {tr_pq.name} or {te_pq.name}.")
                continue

            # load matrices (rows = patients, cols = selected features)
            df_tr = pd.read_parquet(tr_pq)
            df_te = pd.read_parquet(te_pq)

            # compute adaptive threshold on TRAIN, then build graphs
            try:
                radius = compute_adaptive_radius_parameter_mogonet(df_tr, params)
                A_tr = build_train_adjacency_mogonet(df_tr, radius=radius, params=params)
                A_trte = build_trte_block_adjacency_mogonet(df_tr, df_te, radius=radius, params=params)
            except Exception as e:
                print(f"[graph] ERROR {ct}/{mod}: {type(e).__name__}: {e}")
                continue

            # save adjacency
            p_train = out_dir / f"{mod}_train.adj.npz"
            p_trte  = out_dir / f"{mod}_trte_block.adj.npz"
            save_adjacency_npz(A_tr, p_train)
            save_adjacency_npz(A_trte, p_trte)
            print(f"[graph] {ct}/{mod}: wrote {p_train.name}, {p_trte.name}")

            # optional GraphML (for inspection)
            if save_graphml:
                try:
                    g_train = out_dir / f"{mod}_train.graphml"
                    g_trte  = out_dir / f"{mod}_trte_block.graphml"
                    G_tr = csr_to_networkx(A_tr, node_ids=list(map(str, df_tr.index)))
                    save_graph_graphml(G_tr, g_train)
                    G_trte = csr_to_networkx(A_trte, node_ids=(list(map(str, df_tr.index)) + list(map(str, df_te.index))))
                    save_graph_graphml(G_trte, g_trte)
                    print(f"[graph] {ct}/{mod}: wrote {g_train.name}, {g_trte.name}")
                except Exception as e:
                    print(f"[graph] WARN {ct}/{mod}: GraphML export failed: {e}")

# ==========================
# Minimal CLI
# ==========================

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="MOGONET-style graph construction (bulk)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # TRAIN only
    btr = sub.add_parser("bulk-train", help="Build TRAIN adjacency from a TRAIN-only matrix (rows=samples/patients)")
    btr.add_argument("--matrix", type=str, required=True, help="Path to TRAIN matrix (parquet/csv/tsv)")
    btr.add_argument("--format", choices=["parquet", "csv", "tsv"], default="parquet")
    btr.add_argument("--edges-per-node", type=int, default=15)
    btr.add_argument("--out", type=str, required=True)
    btr.add_argument("--out-graphml", type=str, default=None, help="Optional: save NetworkX graph to this .graphml path")

    # TRAIN + TRTE block (either full matrix + id lists OR separate train/test matrices)
    bsp = sub.add_parser("bulk-split", help="Build TRAIN and TRTE adjacencies given train/test data")
    bsp.add_argument("--matrix", type=str, default=None, help="Path to FULL matrix (parquet/csv/tsv)")
    bsp.add_argument("--matrix-train", type=str, default=None, help="Path to TRAIN matrix (parquet/csv/tsv)")
    bsp.add_argument("--matrix-test", type=str, default=None, help="Path to TEST matrix (parquet/csv/tsv)")
    bsp.add_argument("--format", choices=["parquet", "csv", "tsv"], default="parquet")
    bsp.add_argument("--train-ids", type=str, default=None, help="Text file with one TRAIN id per line (used with --matrix)")
    bsp.add_argument("--test-ids", type=str, default=None, help="Text file with one TEST id per line (used with --matrix)")
    bsp.add_argument("--edges-per-node", type=int, default=15)
    bsp.add_argument("--out-train", type=str, required=True)
    bsp.add_argument("--out-trte", type=str, required=True)
    bsp.add_argument("--out-train-graphml", type=str, default=None, help="Optional: save TRAIN NetworkX graph (.graphml)")
    bsp.add_argument("--out-trte-graphml", type=str, default=None, help="Optional: save TRTE block NetworkX graph (.graphml)")

    args = p.parse_args()

    if args.cmd == "bulk-train":
        df = _load_matrix(args.matrix, args.format)
        params = MOGONETGraphParams(edges_per_node=args.edges_per_node)
        A_tr = build_train_adjacency_mogonet(df, params=params)
        save_adjacency_npz(A_tr, args.out)
        print(f"TRAIN adjacency saved -> {args.out}")
        if args.out_graphml is not None:
            G_tr = csr_to_networkx(A_tr, node_ids=list(map(str, df.index)))
            save_graph_graphml(G_tr, args.out_graphml)
            print(f"TRAIN NetworkX graph saved -> {args.out_graphml}")

    elif args.cmd == "bulk-split":
        params = MOGONETGraphParams(edges_per_node=args.edges_per_node)

        if args.matrix_train and args.matrix_test:
            # Case 1: separate train/test matrices provided (recommended with processor outputs)
            df_tr = _load_matrix(args.matrix_train, args.format)
            df_te = _load_matrix(args.matrix_test, args.format)
            tr_ids = list(map(str, df_tr.index))
            te_ids = list(map(str, df_te.index))
        elif args.matrix and args.train_ids and args.test_ids:
            # Case 2: one full matrix + explicit id files
            df_full = _load_matrix(args.matrix, args.format)
            with open(args.train_ids) as f:
                tr_ids = [line.strip() for line in f if line.strip()]
            with open(args.test_ids) as f:
                te_ids = [line.strip() for line in f if line.strip()]
            missing_tr = [i for i in tr_ids if i not in df_full.index]
            missing_te = [i for i in te_ids if i not in df_full.index]
            if missing_tr or missing_te:
                raise KeyError(f"Missing IDs — train:{missing_tr[:5]} test:{missing_te[:5]}")
            df_tr = df_full.loc[tr_ids]
            df_te = df_full.loc[te_ids]
        else:
            raise ValueError(
                "Provide either (--matrix-train & --matrix-test) OR (--matrix & --train-ids & --test-ids)."
            )

        radius = compute_adaptive_radius_parameter_mogonet(df_tr, params)
        A_tr = build_train_adjacency_mogonet(df_tr, radius=radius, params=params)
        A_trte = build_trte_block_adjacency_mogonet(df_tr, df_te, radius=radius, params=params)

        save_adjacency_npz(A_tr, args.out_train)
        save_adjacency_npz(A_trte, args.out_trte)
        print(f"TRAIN adjacency saved -> {args.out_train}")
        print(f"TRTE block adjacency saved -> {args.out_trte}")

        if args.out_train_graphml is not None:
            G_tr = csr_to_networkx(A_tr, node_ids=list(map(str, df_tr.index)))
            save_graph_graphml(G_tr, args.out_train_graphml)
            print(f"TRAIN NetworkX graph saved -> {args.out_train_graphml}")
        if args.out_trte_graphml is not None:
            G_trte = csr_to_networkx(A_trte, node_ids=(list(map(str, df_tr.index)) + list(map(str, df_te.index))))
            save_graph_graphml(G_trte, args.out_trte_graphml)
            print(f"TRTE NetworkX graph saved -> {args.out_trte_graphml}")


# ==========================
# Spatial placeholders (empty)
# ==========================
class SpatialGraphParams:
    pass

def build_spatial_knn_graph(*args, **kwargs):
    raise NotImplementedError("Spatial graph functions are placeholders and will be implemented later.")