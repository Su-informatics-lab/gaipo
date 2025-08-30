# src/extractor.py
"""
extractor.py — Pull clinical + molecular profiles from cBioPortal for configured studies

What it does
------------
1) Loads cancer types of interest and studyId mapping from config.
2) Loads CCDI subject/sample CSVs produced by data_source.py.
   - Subjects: data/fetch/ccdi_api_request/subject_ids/<ct>.csv
   - Samples : data/fetch/ccdi_api_request/sample_ids/<ct>.csv (optional; if missing,
               sample IDs are fetched per patient from cBioPortal)
3) Fetches clinical tables via cBioPortal Swagger (bravado):
   - Patient clinical  → pivot to one row per patientId
   - Sample clinical   → pivot to one row per sampleId
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
CONFIG_PATH           : path to pipeline_config.yaml (default: config/pipeline_config.yaml)
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

API_BASE = cfg["services"]["cbioportal_api"].rstrip("/")
SWAGGER_URL = cfg["services"]["cbioportal_swagger"]

study_ids = cfg["studyId"]                      # e.g. {"Wilms Tumor": "wt_target_2018_pub"}
expression_profiles = cfg.get("expression_profiles", {})  # mapping cancer_type -> list[str]
cancer_types = cfg["cancer_type_interest"]
TIMEOUT  = cfg.get("timeout", 60)

# optional auth token support (no-op if not set)
CBIO_TOKEN = (cfg.get("auth", {}) or {}).get("cbioportal_api_token") or os.getenv("CBIOPORTAL_API_TOKEN")

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

# ─── 2) Init cBioPortal bravado client ──────────────────────────────────────────
http_client = RequestsClient()
if CBIO_TOKEN:
    http_client.session.headers.update({"Authorization": f"Bearer {CBIO_TOKEN}"})
http_client.session.headers.update({"User-Agent": "gaipo/0.1 (+extractor.py)"})

cbioportal = SwaggerClient.from_url(
    SWAGGER_URL,
    http_client=http_client,
    config={
        "validate_requests": False,
        "validate_responses": False,
        "validate_swagger_spec": False,
    }
)
# normalize namespace keys (defensive no-op in most cases)
for name in dir(cbioportal):
    setattr(cbioportal, name.replace(" ", "_").lower(), getattr(cbioportal, name))

# ─── 3) Helpers ────────────────────────────────────────────────────────────────
def _sleep(s: float = 0.05):
    if s > 0:
        time.sleep(s)

def load_ids(path: Path, id_col: str) -> list[str]:
    df = pd.read_csv(path, usecols=[id_col])
    return df[id_col].dropna().astype(str).unique().tolist()

def fetch_patient_clinical(study_id: str, patient_ids: list[str]) -> pd.DataFrame:
    recs = []
    for pid in patient_ids:
        try:
            out = cbioportal.Clinical_Data.getAllClinicalDataOfPatientInStudyUsingGET(
                studyId=study_id,
                patientId=pid,
                projection="DETAILED",
                pageSize=10_000_000
            ).result()
            for mdl in (out or []):
                recs.append({
                    "patientId":           getattr(mdl, "patientId", None),
                    "clinicalAttributeId": getattr(mdl, "clinicalAttributeId", None),
                    "value":               getattr(mdl, "value", None),
                })
        except BravadoHTTPError as e:
            code = e.response.status_code
            if code in (404, 500):
                print(f"[WARN] skipping patient {pid} (HTTP {code})")
                continue
            raise
        except SwaggerMappingError:
            print(f"[WARN] malformed response for patient {pid}, skipping")
            continue
        _sleep(0.05)

    df = pd.DataFrame(recs)
    if df.empty or "patientId" not in df.columns:
        print(f"[ERROR] No patient clinical for study {study_id}.")
        return pd.DataFrame()
    return (df.pivot(index="patientId", columns="clinicalAttributeId", values="value")
              .reset_index()
              .rename_axis(None, axis=1))

def fetch_samples_for_patients(study_id: str, patient_ids: list[str]) -> list[str]:
    sample_ids = []
    for pid in patient_ids:
        try:
            out = cbioportal.Samples.getAllSamplesOfPatientInStudyUsingGET(
                studyId=study_id,
                patientId=pid,
                pageSize=10_000_000
            ).result()
            # bravado objects sometimes act like dicts; support both
            for r in (out or []):
                sid = getattr(r, "sampleId", None) if hasattr(r, "sampleId") else r.get("sampleId")
                if sid:
                    sample_ids.append(str(sid))
        except BravadoHTTPError as e:
            code = e.response.status_code
            if code in (404, 500):
                print(f"[WARN] skipping patient {pid} (HTTP {code})")
                continue
            raise
        except SwaggerMappingError:
            print(f"[WARN] malformed response for patient {pid}, skipping")
            continue
        _sleep(0.05)
    return sorted(set(sample_ids))

def fetch_sample_clinical(study_id: str, sample_ids: list[str]) -> pd.DataFrame:
    recs = []
    for sid in sample_ids:
        try:
            out = cbioportal.Clinical_Data.getAllClinicalDataOfSampleInStudyUsingGET(
                studyId=study_id,
                sampleId=sid,
                projection="DETAILED",
                pageSize=10_000_000
            ).result()
            for mdl in (out or []):
                recs.append({
                    "sampleId":           getattr(mdl, "sampleId", None),
                    "clinicalAttributeId": getattr(mdl, "clinicalAttributeId", None),
                    "value":               getattr(mdl, "value", None),
                })
        except BravadoHTTPError as e:
            code = e.response.status_code
            if code in (404, 500):
                print(f"[WARN] skipping sample {sid} (HTTP {code})")
                continue
            raise
        except SwaggerMappingError:
            print(f"[WARN] malformed response for sample {sid}, skipping")
            continue
        _sleep(0.05)

    df = pd.DataFrame(recs)
    if df.empty or "sampleId" not in df.columns:
        print(f"[ERROR] No sample clinical for study {study_id}.")
        return pd.DataFrame()
    return (df.pivot(index="sampleId", columns="clinicalAttributeId", values="value")
              .reset_index()
              .rename_axis(None, axis=1))

def fetch_profile_data(profile_id: str, sample_ids: list[str]) -> pd.DataFrame:
    """
    POST /api/molecular-profiles/{profile_id}/molecular-data/fetch
    """
    url  = f"{API_BASE}/molecular-profiles/{profile_id}/molecular-data/fetch"
    body = {"sampleIds": sample_ids}
    params = {"projection": "DETAILED"}
    headers = {"User-Agent": "gaipo/0.1 (+extractor.py)"}
    if CBIO_TOKEN:
        headers["Authorization"] = f"Bearer {CBIO_TOKEN}"
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
            "hugoGeneSymbol":     gene.get("hugoGeneSymbol"),
            "entrezGeneId":       rec.get("entrezGeneId"),
            "sampleId":           rec.get("sampleId"),
            "patientId":          rec.get("patientId"),
            "studyId":            rec.get("studyId"),
            "value":              rec.get("value"),
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

def subject_from_sample(sid: str) -> str:
    return sid.rsplit("-", 1)[0]

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

# --- cross zarr v2/v3 writer helpers ---
def _put_array(g, name, data, chunks):
    shape = data.shape
    dtype = data.dtype
    # v3 with zstd
    try:
        if _HAVE_V3_CODEC:
            arr = g.create_array(name, shape=shape, dtype=dtype, chunks=chunks, codecs=[ZstdCodec(level=5)])
            arr[:] = data
            return arr
    except TypeError:
        pass
    # v3 no compression
    try:
        arr = g.create_array(name, shape=shape, dtype=dtype, chunks=chunks)
        arr[:] = data
        return arr
    except TypeError:
        pass
    # v2 fallback
    comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    arr = g.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, compressor=comp)
    arr[:] = data
    return arr

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

    dt_samp = np.dtype(f"U{max(1, max((len(s) for s in samples), default=1))}")
    dt_gene = np.dtype(f"U{max(1, max((len(s) for s in genes),   default=1))}")
    _put_array(g, "axis0", np.asarray(samples, dtype=dt_samp), chunks=(min(8192, len(samples)),))
    _put_array(g, "axis1", np.asarray(genes, dtype=dt_gene),   chunks=(min(8192, len(genes)),))
    g.attrs.update({"axis0":"samples","axis1":"genes","created_from":"cbioportal"})

    pd.DataFrame({"sample_id": samples, "row": np.arange(len(samples), dtype=np.int32)}).to_parquet(
        rows_index_path, compression="zstd", index=False
    )
    cols_df = {"gene_symbol": genes, "col": np.arange(len(genes), dtype=np.int32)}
    if entrez is not None:
        cols_df["entrezGeneId"] = pd.Series(entrez.reindex(genes)).astype("Int64").tolist()
    pd.DataFrame(cols_df).to_parquet(cols_index_path, compression="zstd", index=False)

# --- profile discovery helpers ---
def _discover_expression_profiles(study_id: str, categories: list[str]) -> dict[str, list[dict]]:
    headers = {"User-Agent": "gaipo/0.1 (+extractor.py)"}
    if CBIO_TOKEN:
        headers["Authorization"] = f"Bearer {CBIO_TOKEN}"

    profiles = requests.get(
        f"{API_BASE}/studies/{study_id}/molecular-profiles",
        headers=headers, timeout=TIMEOUT
    ).json()

    # normalize requested category names
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
        "mrna":        {"alter_types": {"MRNA_EXPRESSION"},        "name_keys": ["mrna", "rna seq", "rna-seq", "rna_seq", "rna expression"]},
        "mirna":       {"alter_types": {"MIRNA_EXPRESSION"},       "name_keys": ["mirna", "microrna"]},
        "methylation": {"alter_types": {"METHYLATION"},            "name_keys": ["methyl"]},
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

# ─── 4) Orchestration ───────────────────────────────────────────────────────────
def data_extract():
    for ct in cancer_types:
        study_id = study_ids[ct]
        outdir = OUTPUT_ROOT / ct.replace(" ", "_")
        outdir.mkdir(parents=True, exist_ok=True)

        # input CSVs from data_fetch()
        subj_file   = SUBJECT_DIR / f"{ct.replace(' ', '_').lower()}.csv"
        sample_file = SAMPLE_DIR  / f"{ct.replace(' ', '_').lower()}.csv"
        if (not subj_file.exists()) and (not sample_file.exists()):
            print(f"[ERROR] missing both subject and sample files for {ct}:\n  {subj_file}\n  {sample_file}")
            continue

        # subjects/patients
        if not subj_file.exists():
            print(f"[WARN] missing subject file for {ct}: {subj_file}")
            continue
        subj_df = pd.read_csv(subj_file, usecols=["subject_id"])
        subj_ids = sorted(subj_df["subject_id"].astype(str).unique())
        print(f"[{ct}] Loaded {len(subj_ids)} subject IDs")

        # samples
        if sample_file.exists():
            s_df = pd.read_csv(sample_file, usecols=["sample_id"])
            sample_ids = sorted(s_df["sample_id"].astype(str).unique())
            print(f"[{ct}] Loaded {len(sample_ids)} sample IDs")
        else:
            print(f"[{ct}] No sample CSV; fetching via API…")
            sample_ids = fetch_samples_for_patients(study_id, subj_ids)
            print(f"[{ct}] Fetched {len(sample_ids)} sample IDs")

        # patient clinical
        print(f"[{ct}] fetching patient clinical ({len(subj_ids)} patients)…")
        pat_df = fetch_patient_clinical(study_id, subj_ids)
        pat_out = outdir / "clinical_patients.csv"
        pat_df.to_csv(pat_out, index=False)
        print(f"[{ct}] wrote {pat_out}")

        # sample clinical
        print(f"[{ct}] fetching sample clinical ({len(sample_ids)} samples)…")
        samp_df = fetch_sample_clinical(study_id, sample_ids)
        samp_out = outdir / "clinical_samples.csv"
        samp_df.to_csv(samp_out, index=False)
        print(f"[{ct}] wrote {samp_out}")

        # GDM parquet mirrors
        write_subjects_parquet_from_pat_df(pat_df, GDM_CLIN / "clinical_subjects.parquet")
        write_samples_parquet_from_samp_df(samp_df, GDM_CLIN / "clinical_samples.parquet")

        # discover expression profiles for this cancer type
        wanted_cats = expression_profiles.get(ct, [])
        found = _discover_expression_profiles(study_id, wanted_cats)

        # fetch + write omics
        for category in wanted_cats:
            profs = found.get(category, [])
            if not profs:
                print(f"[{ct}] no '{category}' profiles, skipping")
                continue

            for p in profs:
                pid = p["molecularProfileId"]
                print(f"[{ct}] fetching {pid}…")
                df_long = fetch_profile_data(pid, sample_ids)
                if df_long.empty:
                    print(f"   – {pid} returned 0 rows")
                    continue

                df_wide = pivot_profile_data(df_long)
                outfn = outdir / f"{pid}.tsv"
                df_wide.to_csv(outfn, sep="\t", index=False)
                print(f"[{ct}] wrote {len(df_wide):,} genes × {len(df_wide.columns)-2:,} samples -> {outfn}")

                # canonical mapping -> zarr/parquet
                canon = None
                name_l = (pid or "").lower()
                if ("mrna" in name_l) or ("rna_seq" in name_l) or ("rna-seq" in name_l) or ("rna" in name_l):
                    canon = "mrna"
                elif ("mirna" in name_l) or ("microrna" in name_l):
                    canon = "mirna"
                elif "methyl" in name_l:
                    canon = "methylation"
                elif "linear" in name_l and "cna" in name_l:
                    canon = "linear_cna"

                if canon:
                    zarr_dir = GDM_FEAT / f"{canon}.zarr"
                    rows_idx = GDM_IDX  / f"{canon}_rows.parquet"
                    cols_idx = GDM_IDX  / f"{canon}_cols.parquet"
                    wide_to_zarr(df_wide, zarr_dir, rows_idx, cols_idx)

                _sleep(0.2)

# allow `python -m src.extractor` for ad-hoc testing
if __name__ == "__main__":
    data_extract()