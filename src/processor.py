"""
processor.py — Bulk omics preprocessing for Glioma & Wilms Tumor

Goals
-----
1) Load clinical tables and bulk omics matrices fetched from cBioPortal (CSV/TSV) that
   the extractor wrote under: data/fetch/cbioportal_api_request/<cbio|pedcbio>/<Cancer_Type>/.
2) Align sample-level omics with patient-level clinical labels.
   - Glioma: patient/sample IDs often align directly (or derive patient by trimming '-NN').
   - Wilms Tumor: map sampleId -> patientId using CCDI mapping
     data/fetch/ccdi_api_request/sample_ids/<cancer_type>.csv (columns: sample_id, subject_id).
   - If multiple samples per patient, aggregate to patient-level features (median per gene).
3) Pre-select top-K features per modality using a training-only fit (VarianceThreshold -> FDR(ANOVA F) -> KBest),
   then optionally scale (z-score / min–max) *after* selection to avoid leakage.
4) Output:
   - Clean feature matrices per modality (patients × features), aligned to the label index.
   - Encoded binary labels (y) with an inverse label map {0: classA, 1: classB}.
   - Train/test splits (and persisted files) compatible with downstream graph builder/GNN.

Notes
-----
- cBioPortal expression TSVs typically come as (genes × samples) with leading columns:
  ['hugoGeneSymbol', 'entrezGeneId']. We set 'hugoGeneSymbol' as index and transpose to (samples × genes).
- Without CCDI mapping:
    • If labels are patient-based, we attempt a light coercion from sampleId -> patientId by trimming a final
      '-NN' suffix (e.g., TARGET-xx-xxxx-01 -> TARGET-xx-xxxx). If that fails, provide the CCDI mapping CSV.
    • If labels are sample-based, keep sample indexing; however, this processor currently expects patient-level labels
      for multi-modal alignment and will error without a mapping.
- Feature selection pipeline:
    VarianceThreshold(threshold=0.01) -> SelectFdr(f_classif, alpha=0.05) -> SelectKBest(f_classif, k=top_k)
  After selection, optional normalization can be applied (z-score and/or min–max).
- Files written to: data/gdm/filtered/<Cancer_Type>/
    mrna.parquet, mirna.parquet, methylation.parquet              # full filtered matrices (patients × features)
    mrna_train.parquet, mrna_test.parquet, ...                    # split matrices
    labels_train.csv, labels_test.csv                              # encoded labels aligned to splits
  CSV mirrors (*_all.csv, *_train.csv, *_test.csv) are written for debugging.

Config-driven usage (recommended via main.py)
---------------------------------------------
pipeline_config.yaml:
  expression_profiles:
    "Wilms Tumor": [mrna, mirna, methylation]
  processor:
    labels:
      "Wilms Tumor":
        source: samples
        column: HISTOLOGY_CLASSIFICATION_IN_PRIMARY_TUMOR
        map: "FHWT=0,DAWT=1"
        top_k: 500
        test_size: 0.2
        seed: 42

Run:
  python -m src.main --call data_process
  # or as part of the chain:
  python -m src.main --call data_fetch,data_extract,data_process

Ad-hoc CLI usage
----------------
Wilms Tumor:
  python src/processor.py \
    --cbio-dir data/fetch/cbioportal_api_request/cbio/wilms_tumor/wt_target_2018_pub \
    --ccdi-csv data/fetch/ccdi_api_request/sample_ids/wilms_tumor.csv \
    --cancer-type "Wilms Tumor" \
    --modalities mrna mirna methylation \
    --label-source samples \
    --label-column HISTOLOGY_CLASSIFICATION_IN_PRIMARY_TUMOR \
    --label-map "FHWT=0,DAWT=1" \
    --top-k 500 --test-size 0.2 --seed 42

Glioma (example single-modality):
  python src/processor.py \
    --cbio-dir data/fetch/cbioportal_api_request/pedcbio/glioma/pbta_all \
    --cancer-type "Glioma" \
    --modalities mrna \
    --label-source samples \
    --label-column CANCER_TYPE \
    --label-map "low-grade=0,high-grade=1" \
    --top-k 500 --test-size 0.2 --seed 42
"""
# src/processor.py
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectFdr, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import yaml

# ==========================
# Feature selection helper
# ==========================

def feature_filter_scaler(
    data: pd.DataFrame,
    labels: pd.Series,
    var_threshold,               # float or List[float]
    alpha,                       # float or List[float]
    topFeature,                  # int   or List[int]
    *,
    standard: bool = False,
    minimax: bool = True,
    mod_idx: int = 0,            # NEW: which modality's hyperparams to use
) -> pd.DataFrame:
    """
    Apply VarianceThreshold + FDR(ANOVA F) + SelectKBest to 'data' using
    hyperparameters that can be per-modality lists or scalars.

    var_threshold: float or list of floats (len = #modalities)
    alpha:         float or list of floats (len = #modalities)
    topFeature:    int   or list of ints   (len = #modalities)
    mod_idx:       index of current modality in the configured list
    """
    # Select per-modality hyperparameters (or use scalars)
    vt  = float(_pick_for_mod(var_threshold, mod_idx))
    a   = float(_pick_for_mod(alpha, mod_idx))
    k   = int(_pick_for_mod(topFeature, mod_idx))

    tmp_filter = data.select_dtypes(include=["number"]).copy()
    # keep columns that are non-zero in >= 50% of rows
    tmp_filter = tmp_filter.loc[:, (tmp_filter != 0).sum(axis=0) >= (0.5 * tmp_filter.shape[0])]
    # drop constant-mean columns and NaN columns
    tmp_filter = tmp_filter.loc[:, tmp_filter.mean() != 0].dropna(axis=1)

    pipe = Pipeline(
        [
            ("var_thresh", VarianceThreshold(threshold=vt)),
            ("fdr", SelectFdr(score_func=f_classif, alpha=a)),
            ("kbest", SelectKBest(score_func=f_classif, k=min(k, max(1, tmp_filter.shape[1]-1)))),
        ]
    )
    pipe.fit(tmp_filter, labels)

    cols_vt = tmp_filter.columns[pipe.named_steps["var_thresh"].get_support()]
    cols_fdr = cols_vt[pipe.named_steps["fdr"].get_support()] if "fdr" in pipe.named_steps else cols_vt
    final_cols = cols_fdr[pipe.named_steps["kbest"].get_support()]

    X_sel = pipe.transform(tmp_filter)
    filtered_df = pd.DataFrame(X_sel, index=tmp_filter.index, columns=final_cols)

    if standard:
        filtered_df = filtered_df.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12), axis=0)
    if minimax:
        filtered_df = filtered_df.apply(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12), axis=0)

    filtered_df.columns.name = None
    return filtered_df

# ==========================
# Label & output helpers
# ==========================

def _pick_for_mod(x, mod_idx: int):
    """Return scalar for this modality even if x is a list/tuple."""
    if isinstance(x, (list, tuple)):
        if not x:
            raise ValueError("Empty list provided for a per-modality parameter.")
        if mod_idx >= len(x):
            raise IndexError(f"mod_idx={mod_idx} out of range for list of length {len(x)}")
        return x[mod_idx]
    return x

def _collapse_mirna_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse miRNA rows that share the same base symbol (substring before '/').
    Example:
      LET-7A-1/3P, LET-7A-1/5P, LET-7A-1/7A*, LET-7A-1/7A-1L  -> base 'LET-7A-1'
    Policy:
      - If all rows for a base are identical across samples, keep the first.
      - Otherwise, aggregate by median (row-wise).
    Returns a dataframe with index = base miRNA symbol, columns = sampleIds
    (i.e., still 'genes x samples' orientation — same as input), ready to transpose.
    """
    # Copy to avoid mutating caller
    df = df_raw.copy()

    # Drop non-expression columns if present
    df = df.drop(columns=[c for c in ["entrezGeneId"] if c in df.columns], errors="ignore")

    if "hugoGeneSymbol" not in df.columns:
        raise ValueError("Expected column 'hugoGeneSymbol' in miRNA TSV")

    # Build base symbol: keep everything before the first '/'
    base = df["hugoGeneSymbol"].astype(str).str.replace(r"/.*$", "", regex=True)
    df["_base"] = base

    # Expression-only matrix
    expr_cols = [c for c in df.columns if c not in ("hugoGeneSymbol", "_base")]
    expr = df[expr_cols]

    # Group by base and collapse
    pieces = []
    for b, g in df.groupby("_base"):
        g_expr = g[expr_cols]
        # If all rows in this group are identical across all samples, keep the first
        if g_expr.drop_duplicates().shape[0] == 1:
            rep = g_expr.iloc[0].to_frame().T
            rep.index = [b]
            pieces.append(rep)
        else:
            # Otherwise, take the median across the group
            med = g_expr.median(axis=0, numeric_only=True).to_frame().T
            med.index = [b]
            pieces.append(med)

    out = pd.concat(pieces, axis=0)
    out.index.name = "hugoGeneSymbol_base"
    return out

def _parse_label_map_str(label_map_str: Optional[str]) -> Optional[Dict[str, int]]:
    if not label_map_str:
        return None
    mp: Dict[str, int] = {}
    for tok in str(label_map_str).split(","):
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v not in {"0", "1"}:
            raise ValueError("Label map values must be 0 or 1, e.g., 'FHWT=0,DAWT=1'.")
        mp[k] = int(v)
    return mp or None


def _build_binary_labels_general(
    *,
    label_source: str,
    label_column: str,
    clin_pat: pd.DataFrame,
    clin_sam: pd.DataFrame,
    mapping_series: Optional[pd.Series],  # index=sampleId, values=patientId
    label_map_override: Optional[Dict[str, int]] = None,
    collapse_to_patient: bool = True,
) -> Tuple[pd.Series, Dict[int, str]]:
    if label_source not in {"patients", "samples"}:
        raise ValueError("label_source must be 'patients' or 'samples'")

    if label_source == "patients":
        ser = clin_pat.set_index("patientId")[label_column].astype(str).str.strip()
    else:
        ser = clin_sam.set_index("sampleId")[label_column].astype(str).str.strip()

    if label_map_override is not None:
        mapped = ser.map({k: int(v) for k, v in label_map_override.items()})
        if mapped.isna().any():
            missing = sorted(ser[mapped.isna()].unique().tolist())
            raise ValueError(f"Label values not covered by label_map: {missing}")
        labels = mapped.astype(int)
        inv = {int(v): k for k, v in label_map_override.items()}
    else:
        uniq = sorted([u for u in ser.dropna().unique().tolist()])
        if len(uniq) != 2:
            raise ValueError(
                f"Column '{label_column}' has {len(uniq)} unique values; provide --label-map like 'A=0,B=1'."
            )
        inferred = {uniq[0]: 0, uniq[1]: 1}
        labels = ser.map(inferred).astype(int)
        inv = {0: uniq[0], 1: uniq[1]}

    if mapping_series is not None and label_source == "samples" and collapse_to_patient:
        idx = labels.index.intersection(mapping_series.index)
        if idx.empty:
            raise ValueError("No overlap between labels (samples) and mapping (sampleId->patientId).")
        tmp = pd.DataFrame({"y": labels.loc[idx], "patientId": mapping_series.loc[idx].values})
        labels = tmp.groupby("patientId")["y"].first().astype(int)
        labels.index.name = "patientId"
    else:
        if label_source == "patients":
            labels.index.name = "patientId"

    return labels, inv

def _slug(s: str) -> str:
    return str(s).replace(" ", "_")

def _derive_os_binary(df_pat: pd.DataFrame, status_col: str = "OS_STATUS") -> pd.Series:
    """
    cBio OS_STATUS often looks like '1:DECEASED' or '0:LIVING'.
    Returns int 1 (deceased) / 0 (living/other). Missing -> 0.
    """
    if status_col not in df_pat.columns:
        return pd.Series([np.nan] * len(df_pat), index=df_pat.index).fillna(0).astype(int)
    s = df_pat[status_col].astype(str).str.strip().str.upper()
    out = pd.Series(0, index=s.index, dtype="int64")
    out = out.mask(s.str.startswith("1"), 1)
    out = out.mask(s.str.contains("DECEASED"), 1)
    return out.fillna(0).astype(int)

def _derive_patient_from_sample_index_like(series: pd.Series) -> pd.Series:
    """
    Convert sample IDs like TARGET-50-CAAAAC-01 -> TARGET-50-CAAAAC.
    Works on any string-ish series; leaves values unchanged if no '-NN' suffix.
    """
    s = series.astype(str)
    return s.str.replace(r"-\d+$", "", regex=True)

def _patient_aggregate_samples(clin_sam: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clinical_samples to patientId level:
      - Create patientId from sampleId (strip '-NN' suffix).
      - For duplicate patients, numeric columns -> mean; non-numeric -> first non-null.
    Returns a DataFrame indexed by patientId.
    """
    if "sampleId" not in clin_sam.columns:
        # Nothing to aggregate; return empty to avoid breaking joins downstream
        return pd.DataFrame()

    sam = clin_sam.copy()
    sam["patientId"] = _derive_patient_from_sample_index_like(sam["sampleId"])

    # Split numeric vs. non-numeric
    num_cols = sam.select_dtypes(include=["number"]).columns.tolist()
    # exclude 'sampleId' and 'patientId' from non-numeric pool
    non_num_cols = [c for c in sam.columns if c not in num_cols and c not in ("sampleId", "patientId")]

    agg_parts = []
    if num_cols:
        agg_num = sam.groupby("patientId")[num_cols].mean(numeric_only=True)
        agg_parts.append(agg_num)
    if non_num_cols:
        # first non-null per column
        agg_non = sam.groupby("patientId")[non_num_cols].agg(lambda x: x.dropna().iloc[0] if x.dropna().size else np.nan)
        agg_parts.append(agg_non)

    if not agg_parts:
        return pd.DataFrame(index=sam["patientId"].unique()).sort_index()

    agg = pd.concat(agg_parts, axis=1)
    agg.index.name = "patientId"
    return agg

def _build_clinical_filtered(clin_pat: pd.DataFrame,
                             clin_sam: pd.DataFrame,
                             keep_patient_ids: List[str]) -> pd.DataFrame:
    """
    Merge patient-level clinical (clinical_patients) with patient-aggregated sample clinical,
    then restrict to keep_patient_ids (the IDs used in features/splits).
    Columns are not hard-coded; we keep everything available from both tables.
    """
    # Ensure patientId is the index on patients
    if "patientId" not in clin_pat.columns:
        raise KeyError("clinical_patients.csv must have a 'patientId' column.")
    pat = clin_pat.copy().set_index("patientId")

    sam_agg = _patient_aggregate_samples(clin_sam)
    # Align: left join keeps all patients; right columns get suffix if collide
    merged = pat.join(sam_agg, how="left", rsuffix="_sam")

    # Restrict to patients actually present in filtered features
    ids = pd.Index([str(x) for x in keep_patient_ids])
    # Some pipelines normalize IDs; try to match loosely
    have = merged.index.astype(str)
    merged = merged.loc[have.intersection(ids)]
    merged.index.name = "patientId"

    # Ensure stable dtypes (floats/ints stay numeric; objects remain)
    # Nothing fancy needed here; the write step will handle.
    return merged.reset_index()

# ==========================
# Main orchestration
# ==========================

def _find_omics_file(rawData_dir: Path, mod: str) -> Path:
    """
    Find a TSV written by extractor for the requested modality.
    We accept common patterns per cBioPortal profile names.
    """
    patterns = {
        "mrna": [
            "*_rna_seq_mrna_median_all_sample_Zscores.tsv",
            "*mrna*.tsv",
            "*rna*seq*.tsv",
        ],
        "mirna": [
            "*mirna*.tsv",
            "*microrna*.tsv",
        ],
        "methylation": [
            "*methyl*.tsv",
            "*hm450*.tsv",
        ],
    }
    for pat in patterns.get(mod, []):
        hits = list(rawData_dir.glob(pat))
        if hits:
            return sorted(hits)[0]
    raise FileNotFoundError(f"No TSV found for modality '{mod}' in {rawData_dir}")

def load_and_prepare(
    rawData_dir: Path,
    ccdi_csv: Optional[Path],
    cancer_type: str,
    modalities: List[str],
    label_source: str,
    label_column: str,
    top_k: int = 500,
    test_size: float = 0.2,
    random_state: int = 42,
    label_map: Optional[str] = None,
    var_threshold=0.01,
    alpha=0.05,
):
    clin_pat = pd.read_csv(rawData_dir / "clinical_patients.csv")
    clin_sam = pd.read_csv(rawData_dir / "clinical_samples.csv")

    mapping_series = None
    if ccdi_csv is not None and Path(ccdi_csv).exists():
        ccdi = pd.read_csv(ccdi_csv)
        rename = {}
        if "sample_id" in ccdi.columns: rename["sample_id"] = "sampleId"
        if "subject_id" in ccdi.columns: rename["subject_id"] = "patientId"
        ccdi = ccdi.rename(columns=rename)
        ccdi = ccdi[["sampleId", "patientId"]].dropna().drop_duplicates()
        ccdi["sampleId"] = ccdi["sampleId"].astype(str).str.strip()
        ccdi["patientId"] = ccdi["patientId"].astype(str).str.strip()
        mapping_series = ccdi.set_index("sampleId")["patientId"]

    user_map = _parse_label_map_str(label_map)
    labels_enc, inv_map = _build_binary_labels_general(
        label_source=label_source,
        label_column=label_column,
        clin_pat=clin_pat,
        clin_sam=clin_sam,
        mapping_series=mapping_series,
        label_map_override=user_map,
        collapse_to_patient=True,
    )

    omics_dict: Dict[str, pd.DataFrame] = {}
    for mi, mod in enumerate(modalities):
        f = _find_omics_file(rawData_dir, mod)
        df_raw = pd.read_csv(f, sep="\t")

        if mod == "mirna":
            genes_by_samples = _collapse_mirna_rows(df_raw)
        else:
            df_tmp = df_raw.drop(columns=[c for c in ["entrezGeneId"] if c in df_raw.columns], errors="ignore")
            if "hugoGeneSymbol" not in df_tmp.columns:
                raise ValueError(f"Expected 'hugoGeneSymbol' in {f}")
            genes_by_samples = df_tmp.set_index("hugoGeneSymbol")

        df = genes_by_samples.T
        df.index = df.index.astype(str).str.strip()
        df.columns.name = None

        if mapping_series is not None:
            keep = df.index.intersection(mapping_series.index)
            if keep.empty:
                raise ValueError(f"[{cancer_type}] No overlapping sampleIds between {mod} and CCDI mapping.")
            df2 = df.loc[keep].copy()
            df2["patientId"] = mapping_series.loc[keep].values
            df = df2.groupby("patientId").aggregate("median")
            df.index.name = "patientId"
        else:
            if label_source == "patients":
                df.index = df.index.astype(str).str.replace(r"-\d+$", "", regex=True)
                df.index.name = "patientId"
            else:
                df.index.name = "sampleId"

        if df.index.name != "patientId":
            raise ValueError("To use patient-level labels with sample-indexed features, provide CCDI mapping.")

        common = df.index.intersection(labels_enc.index)
        if common.empty:
            raise ValueError(
                f"[{cancer_type}] No overlap between {mod} features and labels after mapping.\n"
                "Check CCDI mapping and label_source/label_column."
            )
        df = df.loc[common]
        labels_aligned = labels_enc.loc[common]

        df_filtered = feature_filter_scaler(
            df,
            labels_aligned,
            var_threshold=var_threshold,
            alpha=alpha,
            topFeature=top_k,
            standard=False,
            minimax=True,
            mod_idx=mi,
        )
        df_filtered = df_filtered.astype("float32")
        df_filtered.index.name = "patientId"
        omics_dict[mod] = df_filtered

    commons = set(labels_enc.index)
    for m in modalities:
        commons &= set(omics_dict[m].index)
    common_ids = sorted(list(commons))
    if not common_ids:
        raise ValueError(f"[{cancer_type}] After processing, no common patientIds across all modalities and labels.")

    clinical_filtered_df = _build_clinical_filtered(clin_pat, clin_sam, common_ids)

    for mod in modalities:
        omics_dict[mod] = omics_dict[mod].loc[common_ids]
    labels_final = labels_enc.loc[common_ids]

    tr_ids, te_ids = train_test_split(
        common_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_final.loc[common_ids],
    )
    X_tr_dict = {m: omics_dict[m].loc[tr_ids] for m in modalities}
    X_te_dict = {m: omics_dict[m].loc[te_ids] for m in modalities}
    y_tr = labels_final.loc[tr_ids]
    y_te = labels_final.loc[te_ids]
    clinical_tr = clinical_filtered_df.loc[clinical_filtered_df["patientId"].isin(tr_ids)]
    clinical_te = clinical_filtered_df.loc[clinical_filtered_df["patientId"].isin(te_ids)]

    return (
        X_tr_dict, X_te_dict, y_tr, y_te,
        tr_ids, te_ids, inv_map, omics_dict, clinical_filtered_df,
        clinical_tr, clinical_te
    )

# ==========================
# Config-driven pipeline entry
# ==========================

def _load_cfg() -> dict:
    cfg_path = os.getenv("CONFIG_PATH", "config/pipeline_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


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
    """
    Returns list of (portal, study_id) for a given cancer type.
    portal in {"cbio", "pedcbio"}.
    """
    out: List[Tuple[str, str]] = []
    for sid in c_map.get(ct, []):
        out.append(("cbio", sid))
    for sid in p_map.get(ct, []):
        out.append(("pedcbio", sid))
    return out


def data_process():
    """
    Read config and run preprocessing per (portal, study_id) under:
      rawData_dir: data/fetch/cbioportal_api_request/<portal>/<ct_slug>/<study_id>
      outputs   : data/gdm/filtered/<portal>/<ct_slug>/<study_id>/
    """
    cfg = _load_cfg()
    ct_list = cfg.get("cancer_type_interest", [])
    c_map = _normalize_study_map(cfg.get("cBio_studyId", {}))
    p_map = _normalize_study_map(cfg.get("pedcBio_studyId", {}))
    expr = cfg.get("expression_profiles", {})
    labels_cfg = (cfg.get("processor", {}) or {}).get("labels", {})

    fetch_root = Path("data/fetch/cbioportal_api_request")
    mapping_root = Path("data/fetch/ccdi_api_request/sample_ids")

    for ct in ct_list:
        ct_slug = ct.replace(" ", "_").lower()
        studies = _studies_for_ct(ct, c_map, p_map)
        if not studies:
            print(f"[processor] SKIP {ct}: no studies configured.")
            continue

        # CCDI mapping is per cancer type (shared across studies)
        ccdi_csv = None
        ccdi_path = mapping_root / f"{ct_slug}.csv"
        if ccdi_path.exists():
            ccdi_csv = ccdi_path

        # modalities (normalize microrna->mirna)
        raw_mods = [m.lower() for m in (expr.get(ct, []) or [])]
        syn = {"microrna": "mirna"}
        modalities = [syn.get(m, m) for m in raw_mods if m in {"mrna","mirna","methylation"}]
        if not modalities:
            print(f"[processor] SKIP {ct}: no valid modalities configured.")
            continue

        # labels config
        lc = labels_cfg.get(ct, {})
        label_source = lc.get("source", "samples")
        label_column = lc.get("column", None)
        label_map = lc.get("map", None)
        var_threshold = lc.get("var_threshold", 0.01)
        alpha = lc.get("alpha", 0.05)
        top_k = lc.get("top_k", 500)
        test_size = float(lc.get("test_size", 0.2))
        seed = int(lc.get("seed", 42))

        for portal, study_id in studies:
            rawData_dir = fetch_root / portal / ct_slug / study_id
            if not rawData_dir.exists():
                print(f"[processor] SKIP {ct} [{portal}:{study_id}]: {rawData_dir} not found (run extractor first).")
                continue

            print(f"[processor] {ct} [{portal}:{study_id}]: mods={modalities} label={label_source}:{label_column} top_k={top_k}")

            (X_tr, X_te, y_tr, y_te, tr_ids, te_ids, inv_map,
             omics_dict, clinical_filtered_df, clinical_tr, clinical_te) = load_and_prepare(
                rawData_dir=rawData_dir,
                ccdi_csv=ccdi_csv,
                cancer_type=ct,
                modalities=modalities,
                label_source=label_source,
                label_column=label_column,
                top_k=top_k,
                test_size=test_size,
                random_state=seed,
                label_map=label_map,
                var_threshold=var_threshold,
                alpha=alpha,
            )

            out_dir = Path("data/gdm/filtered") / portal / ct_slug / study_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # write clinical subsets
            if clinical_filtered_df is not None and not clinical_filtered_df.empty:
                clinical_filtered_df.to_parquet(out_dir / "clinical_filtered.parquet", index=False)
                clinical_filtered_df.to_csv(out_dir / "clinical_filtered.csv", index=False)
                clinical_tr.to_parquet(out_dir / "clinical_train.parquet", index=False)
                clinical_tr.to_csv(out_dir / "clinical_train.csv", index=False)
                clinical_te.to_parquet(out_dir / "clinical_test.parquet", index=False)
                clinical_te.to_csv(out_dir / "clinical_test.csv", index=False)

            # write per-modality full filtered + splits
            for mod, df in omics_dict.items():
                df.to_parquet(out_dir / f"{mod}.parquet")
                df.to_csv(out_dir / f"{mod}_all.csv")

            for mod, df in X_tr.items():
                df.to_parquet(out_dir / f"{mod}_train.parquet")
                df.to_csv(out_dir / f"{mod}_train.csv")

            for mod, df in X_te.items():
                df.to_parquet(out_dir / f"{mod}_test.parquet")
                df.to_csv(out_dir / f"{mod}_test.csv")

            y_tr.to_csv(out_dir / "labels_train.csv", header=True)
            y_te.to_csv(out_dir / "labels_test.csv", header=True)

            print(f"[processor] {ct} [{portal}:{study_id}]: wrote {len(tr_ids)} train / {len(te_ids)} test -> {out_dir}")

if __name__ == "__main__":
    # Optional: keep CLI for ad-hoc runs by pointing directly to a rawData_dir
    import argparse
    p = argparse.ArgumentParser(description="Process bulk omics data and output feature matrices + labels.")
    p.add_argument("--rawData-dir", type=str, required=True, help="Path like data/fetch/cbioportal_api_request/<portal>/<ct>/<study>")
    p.add_argument("--ccdi-csv", type=str, default=None)
    p.add_argument("--cancer-type", type=str, required=True)
    p.add_argument("--modalities", nargs="+", required=True)
    p.add_argument("--label-source", choices=["patients", "samples"], required=True)
    p.add_argument("--label-column", type=str, required=True)
    p.add_argument("--label-map", type=str, default=None, help="Binary mapping like 'FHWT=0,DAWT=1'")
    p.add_argument("--top-k", type=int, default=500)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    rawData_dir = Path(args.rawData_dir)
    ccdi_csv = Path(args.ccdi_csv) if args.ccdi_csv else None

    (X_tr_dict, X_te_dict, y_tr, y_te, tr_ids, te_ids, inv_map,
     omics_dict, clinical_filtered_df, clinical_tr, clinical_te) = load_and_prepare(
        rawData_dir=rawData_dir,
        ccdi_csv=ccdi_csv,
        cancer_type=args.cancer_type,
        modalities=args.modalities,
        label_source=args.label_source,
        label_column=args.label_column,
        top_k=args.top_k,
        test_size=args.test_size,
        random_state=args.seed,
        label_map=args.label_map,
    )

    out_dir = Path("data/gdm/filtered") / rawData_dir.parts[-3] / rawData_dir.parts[-2] / rawData_dir.parts[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    if clinical_filtered_df is not None and not clinical_filtered_df.empty:
        clinical_filtered_df.to_parquet(out_dir / "clinical_filtered.parquet", index=False)
        clinical_filtered_df.to_csv(out_dir / "clinical_filtered.csv", index=False)
        clinical_tr.to_csv(out_dir / "clinical_train.csv", index=False)
        clinical_tr.to_parquet(out_dir / "clinical_train.parquet", index=False)
        clinical_te.to_csv(out_dir / "clinical_test.csv", index=False)
        clinical_te.to_parquet(out_dir / "clinical_test.parquet", index=False)

    for mod, df in omics_dict.items():
        df.to_parquet(out_dir / f"{mod}.parquet")
        df.to_csv(out_dir / f"{mod}_all.csv")
    for mod, df in X_tr_dict.items():
        df.to_parquet(out_dir / f"{mod}_train.parquet")
        df.to_csv(out_dir / f"{mod}_train.csv")
    for mod, df in X_te_dict.items():
        df.to_parquet(out_dir / f"{mod}_test.parquet")
        df.to_csv(out_dir / f"{mod}_test.csv")
    y_tr.to_csv(out_dir / "labels_train.csv", header=True)
    y_te.to_csv(out_dir / "labels_test.csv", header=True)

    print(f"[processor CLI] wrote {len(tr_ids)} train / {len(te_ids)} test -> {out_dir}")
