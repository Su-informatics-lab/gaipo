# src/extractor.py
"""
extractor.py — Pull clinical + molecular profiles from cBioPortal for configured studies

What it does
------------
1) Loads cancer types of interest and studyId mapping from config.
2) Loads CCDI subject/sample CSVs produced by data_source.py.
   - Subjects: data/fetch/ccdi_api_request/subject_ids/<ct>.csv
   - Samples: data/fetch/ccdi_api_request/sample_ids/<ct>.csv (optional; if missing,
               sample IDs are fetched per patient from cBioPortal)
3) Fetches clinical tables via cBioPortal Swagger (bravado):
   - Patient clinical -> pivot to one row per patientId
   - Sample clinical -> pivot to one row per sampleId
   Writes CSV mirrors and GDM clinical Parquets.
4) Discovers molecular profiles in the study, then for each requested expression category
   (mrna, mirna, methylation, linear_cna) fetches molecular data via REST:
   POST /api/molecular-profiles/{profileId}/molecular-data/fetch
   - Pivots long data to wide (genes × samples) TSV per profile.
   - Optionally maps into Zarr + Parquet row/col indexes under data/gdm/features + data/gdm/indexes.

Key inputs (from config/pipeline_config.yaml)
---------------------------------------------
services:
  cbioportal_api:    https://www.cbioportal.org/api
  cbioportal_swagger:https://www.cbioportal.org/api/v2/api-docs
cancer_type_interest: ["Wilms Tumor", ...]
studyId:             { "Wilms Tumor": "wt_target_2018_pub", ... }
expression_profiles: { "Wilms Tumor": [mrna, mirna, methylation], ... }
timeout: 60

Outputs
-------
data/fetch/cbioportal_api_request/<Cancer_Type>/
  clinical_patients.csv
  clinical_samples.csv
  <profileId>.tsv                        # gene × sample matrix per discovered profile
data/gdm/clinical/
  clinical_subjects.parquet              # patient-level fields
  clinical_samples.parquet               # sample-level fields
data/gdm/features/
  mrna.zarr, mirna.zarr, methylation.zarr  (when profile recognized)
data/gdm/indexes/
  mrna_rows.parquet, mrna_cols.parquet, ... (row/col indices for feature stores)

Environment
-----------
CONFIG_PATH : path to pipeline_config.yaml (default: config/pipeline_config.yaml)
(OPTIONAL) CBIO token : if you need auth on a private portal, add header logic where marked

CLI / Pipeline
--------------
- Recommended:
    python -m src.main --call data_extract
- Ad-hoc:
    python -m src.extractor   # uses config; or call data_extract() inside Python

Notes & tips
------------
- If you don’t have a precomputed samples CSV, extractor will call
  /Samples.getAllSamplesOfPatientInStudyUsingGET for each patientId.
- Profile discovery is tolerant: it matches by profile name keywords and alteration type.
- Zarr writing is version-flexible: tries v3 (with zarr-codecs Zstd) then v3 (uncompressed) then v2 (Blosc).
- Clinical Parquets are written both to data/gdm/clinical/ and as GDC nodes later by data_model.py.
"""

# extractor.py
import os
import time
import requests
import yaml
import pandas as pd
from pathlib import Path
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from bravado.exception import HTTPError as BravadoHTTPError
from bravado_core.exception import SwaggerMappingError
from numcodecs import Blosc  # v2 path fallback
import numpy as np
import zarr

# zarr v3 codec detection (optional)
try:
    from zarr.codecs import ZstdCodec  # v3 path
    _HAVE_V3_CODEC = True
except Exception:
    _HAVE_V3_CODEC = False

# ─── 1) Load config (honor CONFIG_PATH) ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
cfg_env = os.getenv("CONFIG_PATH")
config_path = Path(cfg_env) if cfg_env else ROOT / "config" / "pipeline_config.yaml"
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

services = cfg["services"]
CBIO_API_BASE   = services["cbioportal_api"].rstrip("/")
CBIO_SWAGGER    = services["cbioportal_swagger"]
PED_API_BASE    = services["pedcbioportal_api"].rstrip("/")
PED_SWAGGER     = services["pedcbioportal_swagger"]

# study maps (can be string or list per CT)
cBio_studyId    = cfg.get("cBio_studyId", {})      # {"Wilms Tumor": "wt_target_2018_pub", "Glioma": ["brain_cptac_2020", ...]}
pedcBio_studyId = cfg.get("pedcBio_studyId", {})   # {"Glioma": "pbta_all", ...}
expression_profiles = cfg.get("expression_profiles", {})  # {"Glioma": ["mrna","linear_cna"], "Wilms Tumor": ["mrna","methylation",...]}
cancer_types = cfg["cancer_type_interest"]
TIMEOUT  = cfg.get("timeout", 60)

# optional auth tokens (env overrides)
CBIO_TOKEN = (cfg.get("auth", {}) or {}).get("cbioportal_api_token") or os.getenv("CBIOPORTAL_API_TOKEN")
PED_TOKEN  = (cfg.get("auth", {}) or {}).get("pedcbioportal_api_token") or os.getenv("PEDCBIOPORTAL_API_TOKEN")

# Raw data output (match your tree)
OUTPUT_ROOT = ROOT / "data" / "fetch" / "cbioportal_api_request"
SUBJECT_DIR = ROOT / "data" / "fetch" / "ccdi_api_request" / "subject_ids"
SAMPLE_DIR  = ROOT / "data" / "fetch" / "ccdi_api_request" / "sample_ids"

# Parquet + Zarr outputs
GDM_ROOT = ROOT / "data" / "gdm"
GDM_CLIN = GDM_ROOT / "clinical"
GDM_FEAT = GDM_ROOT / "features"
GDM_IDX  = GDM_ROOT / "indexes"
for p in [GDM_ROOT, GDM_CLIN, GDM_FEAT, GDM_IDX]:
    p.mkdir(parents=True, exist_ok=True)

# ─── 2) Init portal clients ─────────────────────────────────────────────────────
def _make_client(swagger_url: str, token: str | None):
    http_client = RequestsClient()
    if token:
        http_client.session.headers.update({"Authorization": f"Bearer {token}"})
    http_client.session.headers.update({"User-Agent": "gaipo/0.1 (+extractor.py)"})
    cli = SwaggerClient.from_url(
        swagger_url,
        http_client=http_client,
        config={
            "validate_requests": False,
            "validate_responses": False,
            "validate_swagger_spec": False,
        }
    )
    # normalize namespace keys (defensive)
    for name in dir(cli):
        setattr(cli, name.replace(" ", "_").lower(), getattr(cli, name))
    return cli, http_client

cbioportal, cbio_http = _make_client(CBIO_SWAGGER, CBIO_TOKEN)      # token may be None (that's fine)
pedcbioportal, ped_http = _make_client(PED_SWAGGER, PED_TOKEN)       # token typically required

def _portal_env(portal: str):
    if portal == "cbio":
        return cbioportal, CBIO_API_BASE, cbio_http.session.headers.copy()
    elif portal == "pedcbio":
        return pedcbioportal, PED_API_BASE, ped_http.session.headers.copy()
    else:
        raise ValueError(f"Unknown portal '{portal}'")

# ─── 3) Helpers ────────────────────────────────────────────────────────────────
def _sleep(s: float = 0.02):
    if s > 0:
        time.sleep(s)

def _normalize_study_map(study_map_raw: dict[str, object]) -> dict[str, list[str]]:
    out = {}
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

def load_ids(path: Path, id_col: str) -> list[str]:
    df = pd.read_csv(path, usecols=[id_col])
    return df[id_col].dropna().astype(str).unique().tolist()

def _needs_ccdi_normalization(x: str) -> bool:
    """Return True if the ID looks like CCDI-style (has lowercase letters)."""
    return any(c.islower() for c in x)

def norm_patient_id_for_cbio(pid: str) -> str:
    """
    - CCDI-like: 'pt-z4a9dmas' -> 'PT_Z4A9DMAS'
    - Already portal-style (e.g., 'TARGET-50-CAAAAC'): leave hyphens as-is
    """
    if pid is None:
        return None
    pid = str(pid).strip()
    if not pid:
        return pid
    if _needs_ccdi_normalization(pid):
        return pid.upper().replace("-", "_")
    return pid

def norm_sample_id_for_cbio(sid: str) -> str:
    """
    - CCDI-like: 'bs-01tjtdm6' -> 'BS_01TJTDM6'
    - Portal-style (e.g., 'TARGET-50-CAAAAC-01'): leave hyphens as-is
    """
    if not isinstance(sid, str) or not sid:
        return sid
    sid = sid.strip()
    if _needs_ccdi_normalization(sid):
        return sid.upper().replace("-", "_")
    return sid

def subject_from_sample(sid: str) -> str:
    if not isinstance(sid, str):
        return sid
    if "-" in sid:
        return sid.rsplit("-", 1)[0]
    if "_" in sid:
        return sid.rsplit("_", 1)[0]
    return sid

def is_specimen_id(s: str) -> bool:
    """
    Heuristic for CCDI/cBio/PedcBio specimen/biospecimen identifiers.
    Treat as SPECIMEN if:
      - Starts with BS_/bs- (biospecimen)
      - Contains a '.' with known assay suffixes (e.g., .WGS, .WES, .RNA_SEQ, .MULTIPLE)
      - In practice, ANY id containing a '.' is not a cBio 'SAMPLE_ID'
    """
    if not isinstance(s, str) or not s:
        return False
    s = s.strip()
    up = s.upper()
    # Common biospecimen prefixes
    if up.startswith("BS_") or up.startswith("BS-") or s.lower().startswith("bs-"):
        return True
    # Dot-separated assay suffixes (e.g., SJBT030078_D1.WGS / .RNA_SEQ / .MULTIPLE)
    if "." in s:
        suffix = s.rsplit(".", 1)[-1].upper()
        if suffix in {"WGS", "WES", "RNA_SEQ", "RNA-SEQ", "RNA", "MULTIPLE", "RRBS", "HM450", "HM27", "EXOME", "GENOME"}:
             return True
    return False

def is_probable_sample_id(s: str) -> bool:
    """Probable cBio/PedcBio SAMPLE_ID = anything that is not a specimen id."""
    return isinstance(s, str) and bool(s.strip()) and (not is_specimen_id(s))

# ─── 4) Portal-aware clinical fetchers ─────────────────────────────────────────
def fetch_patient_clinical(client, study_id: str, patient_ids: list[str]) -> tuple[pd.DataFrame, set[str]]:
    """
    Returns (patient_clinical_df, ok_patients_set)
    ok_patients_set are those patient IDs for which the API actually returned clinical rows.
    """
    recs = []
    ok_patients = set()
    for raw_pid in patient_ids:
        pid = norm_patient_id_for_cbio(raw_pid)
        try:
            out = client.Clinical_Data.getAllClinicalDataOfPatientInStudyUsingGET(
                studyId=study_id,
                patientId=pid,
                projection="DETAILED",
                pageSize=10_000_000
            ).result()
            if out:
                ok_patients.add(pid)
            for mdl in (out or []):
                recs.append({
                    "patientId": getattr(mdl, "patientId", None),
                    "clinicalAttributeId": getattr(mdl, "clinicalAttributeId", None),
                    "value": getattr(mdl, "value", None),
                })
        except BravadoHTTPError as e:
            if e.response.status_code in (404, 500):
                continue
            raise
        except SwaggerMappingError:
            continue
        _sleep(0.01)

    df = pd.DataFrame(recs)
    if df.empty or "patientId" not in df.columns:
        return pd.DataFrame(), ok_patients
    dfp = (df.pivot(index="patientId", columns="clinicalAttributeId", values="value")
             .reset_index()
             .rename_axis(None, axis=1))
    return dfp, ok_patients

def fetch_samples_for_patients(client, study_id: str, patient_ids: list[str]) -> list[str]:
    sample_ids = []
    for pid in patient_ids:
        try:
            out = client.Samples.getAllSamplesOfPatientInStudyUsingGET(
                studyId=study_id,
                patientId=pid,
                pageSize=10_000_000
            ).result()
            for r in (out or []):
                sid = getattr(r, "sampleId", None) if hasattr(r, "sampleId") else (r or {}).get("sampleId")
                if sid:
                    sample_ids.append(str(sid))
        except BravadoHTTPError as e:
            if e.response.status_code in (404, 500):
                continue
            raise
        except SwaggerMappingError:
            continue
        _sleep(0.005)
    return sorted(set(sample_ids))

def fetch_sample_clinical(client, study_id: str, sample_ids: list[str]) -> pd.DataFrame:
    recs = []
    for raw_sid in sample_ids:
        sid = norm_sample_id_for_cbio(raw_sid)
        try:
            out = client.Clinical_Data.getAllClinicalDataOfSampleInStudyUsingGET(
                studyId=study_id,
                sampleId=sid,
                projection="DETAILED",
                pageSize=10_000_000
            ).result()
            for mdl in (out or []):
                recs.append({
                    "sampleId": getattr(mdl, "sampleId", None),
                    "clinicalAttributeId": getattr(mdl, "clinicalAttributeId", None),
                    "value": getattr(mdl, "value", None),
                })
        except BravadoHTTPError as e:
            if e.response.status_code in (404, 500):
                continue
            raise
        except SwaggerMappingError:
            continue
        _sleep(0.003)

    df = pd.DataFrame(recs)
    if df.empty or "sampleId" not in df.columns:
        return pd.DataFrame()
    dfs = (df.pivot(index="sampleId", columns="clinicalAttributeId", values="value")
             .reset_index()
             .rename_axis(None, axis=1))
    return dfs

# ─── 5) Molecular profiles (client/portal aware) ───────────────────────────────
def _discover_expression_profiles(api_base: str, headers: dict, study_id: str, categories: list[str]) -> dict[str, list[dict]]:
    profiles = requests.get(
        f"{api_base}/studies/{study_id}/molecular-profiles",
        headers=headers, timeout=TIMEOUT
    ).json()

    syn = {
        "microrna": "mirna",
        "mirna": "mirna",
        "mrna": "mrna",
        "methyl": "methylation",
        "methylation": "methylation",
        "linear_cna": "linear_cna",
        "cna_linear": "linear_cna",
    }
    norm_cats = [syn.get((c or "").lower(), (c or "").lower()) for c in categories]

    catmap = {
        "mrna": {"alter_types": {"MRNA_EXPRESSION"}, "name_keys": ["mrna", "rna seq", "rna-seq", "rna_seq", "rna expression"]},
        "mirna": {"alter_types": {"MIRNA_EXPRESSION"}, "name_keys": ["mirna", "microrna"]},
        "methylation": {"alter_types": {"METHYLATION"}, "name_keys": ["methyl"]},
        "linear_cna":  {"alter_types": {"COPY_NUMBER_ALTERATION"}, "name_keys": ["gistic", "cna linear", "linear cna"]},
    }

    out = {c: [] for c in norm_cats}
    for p in profiles:
        name_l = (p.get("name") or "").lower()
        mtype  = (p.get("molecularAlterationType") or "").upper()
        for c in norm_cats:
            spec = catmap.get(c, {"alter_types": set(), "name_keys": [c]})
            if any(k in name_l for k in spec["name_keys"]) or (mtype in spec["alter_types"]):
                out[c].append(p)
    return out

def fetch_profile_data(api_base: str, headers: dict, profile_id: str, sample_ids: list[str]) -> pd.DataFrame:
    """
    POST /api/molecular-profiles/{profile_id}/molecular-data/fetch
    Returns long-form DataFrame:
      [molecularProfileId, hugoGeneSymbol, entrezGeneId, sampleId, patientId, studyId, value]
    """
    if not sample_ids:
        return pd.DataFrame()
    url  = f"{api_base}/molecular-profiles/{profile_id}/molecular-data/fetch"
    body = {"sampleIds": [norm_sample_id_for_cbio(s) for s in sample_ids]}
    params = {"projection": "DETAILED"}
    resp = requests.post(url, params=params, json=body, timeout=120, headers=headers)
    resp.raise_for_status()
    data = resp.json() or []
    if not data:
        return pd.DataFrame()

    records = []
    for rec in data:
        gene = rec.get("gene") or {}
        records.append({
            "molecularProfileId": rec.get("molecularProfileId"),
            "hugoGeneSymbol": gene.get("hugoGeneSymbol"),
            "entrezGeneId": rec.get("entrezGeneId"),
            "sampleId": rec.get("sampleId"),
            "patientId": rec.get("patientId"),
            "studyId": rec.get("studyId"),
            "value": rec.get("value"),
        })
    df = pd.DataFrame.from_records(records)
    return df.drop_duplicates(subset=["molecularProfileId","hugoGeneSymbol","entrezGeneId","sampleId"])

def pivot_profile_data(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return df_long
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    wide = (df_long
            .pivot(index=["hugoGeneSymbol","entrezGeneId"], columns="sampleId", values="value")
            .reset_index()
            .rename_axis(None, axis=1))
    return wide

# ─── 6) Parquet/Zarr writers ───────────────────────────────────────────────────
def write_subjects_parquet_from_pat_df(pat_df: pd.DataFrame, out_path: Path):
    df = pat_df.rename(columns={"patientId":"subject_id"})
    if "OS_STATUS" in df:
        df["os_status_raw"] = df["OS_STATUS"].astype(str)
        df["os_event"] = df["OS_STATUS"].astype(str).str.startswith("1").astype("int8")
    df.rename(columns={
        "AGE":"age_years", "AGE_IN_DAYS":"age_days",
        "OS_DAYS":"os_days", "OS_MONTHS":"os_months",
        "RACE":"race", "ETHNICITY":"ethnicity", "SEX":"sex",
        "CLINICAL_STAGE":"clinical_stage", "EVENT_TYPE":"event_type",
        "PROTOCOL":"protocol"
    }, inplace=True, errors="ignore")
    df.to_parquet(out_path, compression="zstd", index=False)
    print(f"[GDM] wrote {out_path}")

def write_samples_parquet_from_samp_df(samp_df: pd.DataFrame, out_path: Path):
    df = samp_df.rename(columns={"sampleId":"sample_id"})
    if "sample_id" in df and "subject_id" not in df:
        df["subject_id"] = df["sample_id"].map(subject_from_sample)
    df.rename(columns={
        "CANCER_TYPE":"cancer_type",
        "CANCER_TYPE_DETAILED":"cancer_type_detailed",
        "HISTOLOGY_CLASSIFICATION_IN_PRIMARY_TUMOR":"histology_in_primary",
        "ONCOTREE_CODE":"oncotree_code",
        "TMB_NONSYNONYMOUS":"tmb_nonsyn"
    }, inplace=True, errors="ignore")
    df.to_parquet(out_path, compression="zstd", index=False)
    print(f"[GDM] wrote {out_path}")

# v2/v3 zarr helpers
def _put_array(g, name, data, chunks):
    shape = data.shape
    dtype = data.dtype
    try:
        if _HAVE_V3_CODEC:
            arr = g.create_array(name, shape=shape, dtype=dtype, chunks=chunks, codecs=[ZstdCodec(level=5)])
            arr[:] = data
            return arr
    except TypeError:
        pass
    try:
        arr = g.create_array(name, shape=shape, dtype=dtype, chunks=chunks)
        arr[:] = data
        return arr
    except TypeError:
        pass
    comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    arr = g.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, compressor=comp)
    arr[:] = data
    return arr

def _as_fixed_bytes(arr: list[str]) -> np.ndarray:
    """Encode a list of strings to fixed-length UTF-8 bytes dtype S{N} (Zarr v3 safe)."""
    max_len = max((len(s.encode("utf-8")) for s in arr), default=1)
    dt = np.dtype(f"S{max_len}")
    out = np.empty(len(arr), dtype=dt)
    out[:] = [s.encode("utf-8") for s in arr]
    return out

def wide_to_zarr(df_wide, zarr_dir, rows_index_path, cols_index_path, value_dtype="float32"):
    if df_wide.empty:
        return
    df = (df_wide.rename(columns={"hugoGeneSymbol":"gene_symbol","entrezGeneId":"entrezGeneId"})
                  .dropna(subset=["gene_symbol"])
                  .drop_duplicates(subset=["gene_symbol"])
                  .set_index("gene_symbol"))
    entrez = df.pop("entrezGeneId").astype("Int64") if "entrezGeneId" in df_wide else None

    samples = df.columns.astype(str).tolist()
    genes   = df.index.astype(str).tolist()

    # Matrix: samples × genes
    X = df.to_numpy(dtype=value_dtype).T
    chunks = (min(4096, X.shape[0]), min(256, X.shape[1]))

    g = zarr.open_group(str(zarr_dir), mode="w")
    _put_array(g, "X", X, chunks=chunks)

    # Encode labels as fixed-length UTF-8 bytes (avoid Zarr v3 warnings)
    samp_bytes = _as_fixed_bytes(samples)
    gene_bytes = _as_fixed_bytes(genes)

    _put_array(g, "axis0", samp_bytes, chunks=(min(8192, len(samples)),))
    _put_array(g, "axis1", gene_bytes, chunks=(min(8192, len(genes)),))
    g.attrs.update({"axis0":"samples","axis1":"genes","created_from":"cbioportal","encoding":"utf-8"})

    # row/col index sidecars
    pd.DataFrame({"sample_id": samples, "row": np.arange(len(samples), dtype=np.int32)}).to_parquet(
        rows_index_path, compression="zstd", index=False
    )
    cols_df = {"gene_symbol": genes, "col": np.arange(len(genes), dtype=np.int32)}
    if entrez is not None:
        cols_df["entrezGeneId"] = pd.Series(entrez.reindex(genes)).astype("Int64").tolist()
    pd.DataFrame(cols_df).to_parquet(cols_index_path, compression="zstd", index=False)

# ─── 7) Orchestration with dual-portal routing ─────────────────────────────────
def _studies_for_ct(ct: str) -> list[tuple[str, str]]:
    """
    Returns list of tuples (portal_tag, study_id)
      portal_tag in {"cbio", "pedcbio"}
    """
    out = []
    c_map  = _normalize_study_map(cBio_studyId)
    p_map  = _normalize_study_map(pedcBio_studyId)
    for sid in c_map.get(ct, []):
        out.append(("cbio", sid))
    for sid in p_map.get(ct, []):
        out.append(("pedcbio", sid))
    return out

def data_extract():
    for ct in cancer_types:
        studies = _studies_for_ct(ct)
        if not studies:
            print(f"[WARN] No studies configured for cancer type '{ct}', skipping.")
            continue

        ct_key = ct.replace(" ", "_").lower()

        for portal, study_id in studies:
            client, api_base, base_headers = _portal_env(portal)
            outdir = OUTPUT_ROOT / portal / ct_key / study_id
            outdir.mkdir(parents=True, exist_ok=True)

            # headers for requests.*
            headers = base_headers.copy()
            headers.setdefault("User-Agent", "gaipo/0.1 (+extractor.py)")

            # subject/sample CSVs (study-specific first; CT-level fallback)
            subj_file_study   = SUBJECT_DIR / f"{ct_key}_{study_id}.csv"
            sample_file_study = SAMPLE_DIR  / f"{ct_key}_{study_id}.csv"
            subj_file_ct      = SUBJECT_DIR / f"{ct_key}.csv"
            sample_file_ct    = SAMPLE_DIR  / f"{ct_key}.csv"

            # subjects
            if subj_file_study.exists():
                subj_file = subj_file_study
            elif subj_file_ct.exists():
                subj_file = subj_file_ct
            else:
                print(f"[ERROR] Missing subject CSV for {ct} / {portal}:{study_id}")
                continue

            subj_df  = pd.read_csv(subj_file, usecols=["subject_id"])
            subj_ids_raw  = sorted(subj_df["subject_id"].astype(str).unique())
            subj_ids_norm = [norm_patient_id_for_cbio(x) for x in subj_ids_raw]
            print(f"[{ct} / {portal}:{study_id}] Loaded {len(subj_ids_norm)} subject IDs")

            # 1) Patient clinical + ok-patient set
            print(f"[{ct} / {portal}:{study_id}] fetching patient clinical…")
            pat_df, ok_patients = fetch_patient_clinical(client, study_id, subj_ids_norm)
            pat_out = outdir / "clinical_patients.csv"
            pat_df.to_csv(pat_out, index=False)
            print(f"[{ct} / {portal}:{study_id}] wrote {pat_out} (ok patients: {len(ok_patients)})")

            # 2) Samples: derive from OK patients; only keep CSV entries that look like real SAMPLE_IDs
            sample_ids_csv = None
            if sample_file_study.exists():
                sample_file = sample_file_study
            elif sample_file_ct.exists():
                sample_file = sample_file_ct
            else:
                sample_file = None

            if sample_file:
                s_df = pd.read_csv(sample_file, usecols=["sample_id"])
                raw_csv = sorted(s_df["sample_id"].astype(str).unique())
                # Drop CCDI biospecimen ids (BS_/bs-) and dot-suffixed assay identifiers
                sample_ids_csv_filtered = [norm_sample_id_for_cbio(s) for s in raw_csv if is_probable_sample_id(s)]
                kept = len(sample_ids_csv_filtered)
                dropped = len(raw_csv) - kept
                print(f"[{ct} / {portal}:{study_id}] Loaded {len(raw_csv)} from CSV, "
                    f"kept {kept} probable SAMPLE_IDs, dropped {dropped} SPECIMEN_IDs.")
                # If we kept zero, mark as None so we will fall back to API-derived sample IDs
                sample_ids_csv = sample_ids_csv_filtered if kept > 0 else None

            # Always derive sample IDs from OK patients (prevents 0-row molecular fetch)
            if ok_patients:
                print(f"[{ct} / {portal}:{study_id}] deriving sample IDs for ok patients ({len(ok_patients)})…")
                sample_ids_from_ok = fetch_samples_for_patients(client, study_id, sorted(ok_patients))
            else:
                sample_ids_from_ok = []

            # Final selection:
            #  - If CSV had usable sample ids, intersect with API-derived samples (if available).
            #  - Otherwise, just use API-derived samples.
            if sample_ids_csv is not None:
                sample_ids = sorted(set(sample_ids_csv).intersection(sample_ids_from_ok) if sample_ids_from_ok else set(sample_ids_csv))
            else:
                sample_ids = sample_ids_from_ok

            print(f"[{ct} / {portal}:{study_id}] using {len(sample_ids)} samples for sample-clinical & omics fetch")
            
            # 3) Sample clinical for selected samples
            samp_df = fetch_sample_clinical(client, study_id, sample_ids)
            samp_out = outdir / "clinical_samples.csv"
            samp_df.to_csv(samp_out, index=False)
            print(f"[{ct} / {portal}:{study_id}] wrote {samp_out}")

            # parquet mirrors (per-study)
            write_subjects_parquet_from_pat_df(pat_df, GDM_CLIN / f"clinical_subjects__{portal}__{ct_key}__{study_id}.parquet")
            write_samples_parquet_from_samp_df(samp_df, GDM_CLIN / f"clinical_samples__{portal}__{ct_key}__{study_id}.parquet")

            # 4) Omics profiles (mRNA / methylation / CNA / mirna if present)
            wanted_cats = expression_profiles.get(ct, [])
            if wanted_cats:
                found = _discover_expression_profiles(api_base, headers, study_id, wanted_cats)
                for category in wanted_cats:
                    profs = found.get(category, [])
                    if not profs:
                        print(f"[{ct} / {portal}:{study_id}] no '{category}' profiles, skipping")
                        continue

                    for p in profs:
                        pid = p["molecularProfileId"]
                        print(f"[{ct} / {portal}:{study_id}] fetching {pid}…")
                        df_long = fetch_profile_data(api_base, headers, pid, sample_ids)
                        if df_long.empty:
                            print(f"   – {pid} returned 0 rows (samples passed: {len(sample_ids)})")
                            continue

                        df_wide = pivot_profile_data(df_long)
                        outfn = outdir / f"{pid}.tsv"
                        df_wide.to_csv(outfn, sep="\t", index=False)
                        print(f"[{ct} / {portal}:{study_id}] wrote {len(df_wide):,} genes × {len(df_wide.columns)-2:,} samples -> {outfn}")

                        # canonical -> zarr/parquet (per study)
                        canon = None
                        name_l = (pid or "").lower()
                        if ("mrna" in name_l) or ("rna_seq" in name_l) or ("rna-seq" in name_l) or ("rna " in name_l):
                            canon = "mrna"
                        elif ("mirna" in name_l) or ("microrna" in name_l):
                            canon = "mirna"
                        elif "methyl" in name_l:
                            canon = "methylation"
                        elif "linear" in name_l and "cna" in name_l:
                            canon = "linear_cna"

                        if canon:
                            zarr_dir = GDM_FEAT / f"{canon}__{portal}__{ct_key}__{study_id}.zarr"
                            rows_idx = GDM_IDX  / f"{canon}__{portal}__{ct_key}__{study_id}__rows.parquet"
                            cols_idx = GDM_IDX  / f"{canon}__{portal}__{ct_key}__{study_id}__cols.parquet"
                            wide_to_zarr(df_wide, zarr_dir, rows_idx, cols_idx)

                        _sleep(0.2)

# allow `python -m src.extractor` for ad-hoc testing
if __name__ == "__main__":
    data_extract()