#!/usr/bin/env python3
'''
Description:
	1.	Load CCDI subject/sample ID CSVs.
	2.	Fetch patient and sample clinical tables via the bravado client.
	3.	Automatically fall back to pulling sample IDs from patients if no provided sample ids.
	4.	Discover built-in sample lists for each expression category.
	5.	Pull molecular profiles per category/profile ID via HTTP GET.
	6.	Pivot each profile to a wide gene×sample matrix and write out one TSV per profile.
'''

import sys
import time
import requests
import yaml
import pandas as pd
from pathlib import Path
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from bravado.exception import HTTPError as BravadoHTTPError
from bravado_core.exception import SwaggerMappingError

# ─── 1. Load config ─────────────────────────────────────────────────────────────
config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

API_BASE = cfg["services"]["cbioportal_api"].rstrip("/")
SWAGGER_URL = cfg["services"]["cbioportal_swagger"]
study_ids = cfg["studyId"]
expression_profiles = cfg["expression_profiles"]
cancer_types = cfg["cancer_type_interest"]
TIMEOUT  = cfg.get("timeout", 60)
OUTPUT_ROOT = Path(__file__).parent.parent / "data" / "processed"
SUBJECT_DIR = Path(__file__).parent.parent / "data" / "ccdi_api_request" / "subject_ids"
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "ccdi_api_request" / "sample_ids"

# ─── Init cBioPortal client ──────────────────────────────────────────────────
http_client = RequestsClient()
cbioportal  = SwaggerClient.from_url(
    SWAGGER_URL,
    http_client=http_client,
    config={
        "validate_requests": False,
        "validate_responses": False,
        "validate_swagger_spec": False,
    }
)
# normalize resource names
for name in dir(cbioportal):
    setattr(cbioportal, name.replace(" ", "_").lower(), getattr(cbioportal, name))


# ─── 2. Helper Functions ─────────────────────────────────────────────────────────
def load_ids(path: Path, id_col: str) -> list[str]:
    df = pd.read_csv(path, usecols=[id_col])
    return df[id_col].dropna().astype(str).unique().tolist()

# ─── 2.1 Clinical extraction ────────────────────────────────────────────────────────
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
                # pull only the bits we need
                recs.append({
                    "patientId":            getattr(mdl, "patientId", None),
                    "clinicalAttributeId":  getattr(mdl, "clinicalAttributeId", None),
                    "value":                getattr(mdl, "value", None),
                })
        except BravadoHTTPError as e:
            code = e.response.status_code
            if code in (404, 500):             # skip on Not Found or server error
                print(f"[WARN] skipping patient {pid} (HTTP {code})")
                continue
            raise
        except SwaggerMappingError:
            # server sometimes returns an error-object instead of a list → skip
            print(f"[WARN] malformed response for patient {pid}, skipping")
            continue

        time.sleep(0.05)

    df = pd.DataFrame(recs)
    if df.empty or "patientId" not in df.columns:
        print(f"ERROR: No clinical data fetching for provided subject/patient IDs for study {study_id}. Output empty dataset.")
        return pd.DataFrame()
    return (
        df
        .pivot(index="patientId", columns="clinicalAttributeId", values="value")
        .reset_index()
        .rename_axis(None, axis=1)
    )

def fetch_samples_for_patients(study_id: str, patient_ids: list[str]) -> list[str]:
    sample_ids = []
    for pid in patient_ids:
        try:
            out = cbioportal.Samples.getAllSamplesOfPatientInStudyUsingGET(
                studyId=study_id,
                patientId=pid,
                pageSize=10_000_000
            ).result()
            sample_ids.extend([r["sampleId"] for r in out])
        except BravadoHTTPError as e:
            code = e.response.status_code
            if code in (404, 500):             # skip on Not Found or server error
                print(f"[WARN] skipping patient {pid} (HTTP {code})")
                continue
            raise
        except SwaggerMappingError:
            # server sometimes returns an error-object instead of a list → skip
            print(f"[WARN] malformed response for patient {pid}, skipping")
            continue

        time.sleep(0.05)
    return sample_ids

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
                    "sampleId": getattr(mdl, "sampleId", None),
                    "clinicalAttributeId": getattr(mdl, "clinicalAttributeId", None),
                    "value": getattr(mdl, "value", None),
                })
        except BravadoHTTPError as e:
            code = e.response.status_code
            if code in (404, 500):             # skip on Not Found or server error
                print(f"[WARN] skipping patient {sid} (HTTP {code})")
                continue
            raise
        except SwaggerMappingError:
            # server sometimes returns an error-object instead of a list → skip
            print(f"[WARN] malformed response for patient {sid}, skipping")
            continue

        time.sleep(0.05)

    df = pd.DataFrame(recs)
    if df.empty or "sampleId" not in df.columns:
        print(f"ERROR: No clinical data fetching for provided sample IDs for study {study_id}.")
        return pd.DataFrame()
    return (
        df
        .pivot(index="sampleId", columns="clinicalAttributeId", values="value")
        .reset_index()
        .rename_axis(None, axis=1)
    )


# ─── 2.2 Molecular prodfiles & data ──────────────────────────────────────────────────────────────
'''
def get_sample_list_for_profile(study_id: str, suffix: str) -> str | None:
    all_lists = cbioportal.Sample_Lists.getAllSampleListsInStudyUsingGET(studyId=study_id).result()
    for sl in all_lists:
        if sl.sampleListId.endswith(suffix):
            return sl.sampleListId
    return None
'''

def fetch_profile_data(profile_id: str, sample_ids: list[str]) -> pd.DataFrame:
    """
    POST /api/molecular-profiles/{profile_id}/molecular-data/fetch
    Returns a long-form DataFrame with columns:
      [molecularProfileId, hugoGeneSymbol, entrezGeneId,
       sampleId, patientId, studyId, value]
    """
    url  = f"{API_BASE}/molecular-profiles/{profile_id}/molecular-data/fetch"
    body = {"sampleIds": sample_ids}
    params = {"projection": "DETAILED"}
    resp = requests.post(url, params=params, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()  # list of dicts
    
    if not data:
        return pd.DataFrame()

    records = []
    for rec in data:
        gene = rec.get("gene") or {}
        records.append({
            "molecularProfileId": rec.get("molecularProfileId"),
            "hugoGeneSymbol":     gene.get("hugoGeneSymbol"),
            "entrezGeneId":       gene.get("entrezGeneId"),
            "sampleId":           rec.get("sampleId"),
            "patientId":          rec.get("patientId"),
            "studyId":            rec.get("studyId"),
            "value":              rec.get("value"),
        })

    df = pd.DataFrame.from_records(records)
    # drop any exact duplicates
    return df.drop_duplicates(subset=["molecularProfileId", "hugoGeneSymbol", "entrezGeneId", "sampleId"])

def pivot_profile_data(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long-form profile DataFrame into:
      index: [hugoGeneSymbol, entrezGeneId]
      columns: sampleId
      values: value
    """
    if df_long.empty:
        return df_long

    wide = (
        df_long
        .pivot(index=["hugoGeneSymbol", "entrezGeneId"],
               columns="sampleId",
               values="value")
        .reset_index()
        # flatten the columns so sampleIds become normal column names
        .rename_axis(None, axis=1)
    )
    return wide


# ─── 3. Main ─────────────────────────────────────────────────────────────────────
def main():
    for ct in cancer_types:
        study_id = study_ids[ct]
        outdir = OUTPUT_ROOT / ct.replace(" ", "_")
        outdir.mkdir(parents=True, exist_ok=True)

        # 1) Load subject/pateint IDs and sample IDs
        subj_file = SUBJECT_DIR / f"{ct.replace(' ', '_').lower()}.csv"
        sample_file = SAMPLE_DIR / f"{ct.replace(' ', '_').lower()}.csv"
        if (not subj_file.exists()) & (not sample_file.exists()):
            print(f"ERROR: missing both subject file for {ct}: {subj_file} & sample file for {ct}: {sample_file}")
            break

        # Pateint/subject IDs
        if not subj_file.exists():
            print(f"Warning: missing subject file for {ct}: {subj_file}")
            continue
        subj_df = pd.read_csv(subj_file, usecols=["subject_id"])
        subj_ids = sorted(subj_df["subject_id"].astype(str).unique())
        print(f"[{ct}] Loaded {len(subj_ids)} subject IDs from {subj_file}")

        # Sample IOs
        if sample_file.exists():
            sample_df = pd.read_csv(sample_file, usecols=["sample_id"])
            sample_ids = sorted(sample_df["sample_id"].astype(str).unique())
            print(f"[{ct}] Loaded {len(sample_ids)} sample IDs from {sample_file}")
        else:
            print(f"[{ct}] No sample CSV; fetching sample IDs via API…")
            sample_ids = fetch_samples_for_patients(study_id, subj_ids)
            print(f"[{ct}] Fetched {len(sample_ids)} sample IDs")
       
        # 2) Fetch patient clinical data
        print(f"[{ct}] fetching patient/subject clinical data ({len(subj_ids)} patients)")
        pat_df = fetch_patient_clinical(study_id, subj_ids)
        pat_out = outdir / "clinical_patients.csv"
        pat_df.to_csv(pat_out, index=False)
        print(f"[{ct}] Wrote patient clinical to {pat_out}")

        # 3) Fetch sample clinical data (either pre-loaded or via patients)
        print(f"[{ct}] fetching sample clinical data ({len(sample_ids)} samples)")
        sample_df = fetch_sample_clinical(study_id, sample_ids)
        sample_out = outdir / "clinical_samples.csv"
        sample_df.to_csv(outdir / "clinical_samples.csv", index=False)
        print(f"[{ct}] Wrote sample clinical to {sample_out}")
        
        # 4) Discover and fetch molecular profiles & data
        # Discover expression profiles on this study
        profiles = requests.get(f"{API_BASE}/studies/{study_id}/molecular-profiles").json()
        for category in expression_profiles.get(ct, []):
            # find matching profileIds by name prefix
            prof_ids = [
                p["molecularProfileId"]
                for p in profiles
                if p["name"].lower().startswith(category.lower())
            ]
            if not prof_ids:
                print(f"[{ct}] no '{category}' profiles, skipping")
                continue

            for pid in prof_ids:
                print(f"[{ct}] fetching {pid}…")
                df_long = fetch_profile_data(pid, sample_ids)
                if df_long.empty:
                    print(f"   – {pid} returned 0 rows")
                    continue

                df_wide = pivot_profile_data(df_long)
                outfn = outdir / f"{pid}.tsv"
                df_wide.to_csv(outfn, sep="\t", index=False)
                print(f"[{ct}] wrote {len(df_wide):,} genes × {len(sample_ids)} samples -> {outfn}")

                # be gentle on the API
                time.sleep(0.2)
        
if __name__ == '__main__':
    main()
