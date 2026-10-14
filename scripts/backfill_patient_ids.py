"""
backfill_patient_ids.py - add the missing patientId column to existing clinical_samples.csv

Older runs of src/extractor.py wrote clinical_samples.csv without patientId. For TARGET
studies the patient can be derived from the sample barcode, but PBTA / CPTAC studies use
unrelated ids (BS_... samples, PT_... patients), so samples could not be linked to patients.
This script fetches the sample -> patient table once per study from the portal API
(GET /studies/{studyId}/samples) and adds it as a patientId column, without re-running
the full data_extract stage.

    python scripts/backfill_patient_ids.py                 # every study folder found
    python scripts/backfill_patient_ids.py --portal pedcbio
    python scripts/backfill_patient_ids.py --dry-run

Layout: data/fetch/cbioportal_api_request/<cbio|pedcbio>/<ct_slug>/<study_id>/clinical_samples.csv
A copy of each original file is kept as clinical_samples.csv.bak.
Tokens: config/pipeline_config.yaml auth.* or env CBIOPORTAL_API_TOKEN / PEDCBIOPORTAL_API_TOKEN.
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]


def portal_settings(cfg: dict, portal: str):
    services, auth = cfg.get("services", {}), (cfg.get("auth", {}) or {})
    if portal == "cbio":
        base = services.get("cbioportal_api", "https://www.cbioportal.org/api")
        token = auth.get("cbioportal_api_token") or os.getenv("CBIOPORTAL_API_TOKEN")
    else:
        base = services.get("pedcbioportal_api", "https://pedcbioportal.kidsfirstdrc.org/api")
        token = auth.get("pedcbioportal_api_token") or os.getenv("PEDCBIOPORTAL_API_TOKEN")
    headers = {"Accept": "application/json", "User-Agent": "gaipo/0.1 (+backfill_patient_ids.py)"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return base.rstrip("/"), headers


class PortalResponseError(RuntimeError):
    pass


def sample_to_patient(base: str, headers: dict, study_id: str, timeout: int) -> pd.Series:
    url = f"{base}/studies/{study_id}/samples"
    r = requests.get(url, params={"projection": "SUMMARY", "pageSize": 100_000},
                     headers=headers, timeout=timeout, allow_redirects=False)
    ctype = r.headers.get("Content-Type", "")
    if r.is_redirect or r.status_code in (401, 403):
        raise PortalResponseError(
            f"HTTP {r.status_code} from {url} (redirect to: {r.headers.get('Location', '-')}). "
            "The portal wants you to log in: the API token is missing, expired or revoked.")
    if r.status_code != 200 or "json" not in ctype.lower():
        raise PortalResponseError(
            f"HTTP {r.status_code}, Content-Type '{ctype}' from {url}; expected JSON. "
            f"First bytes: {r.text[:200]!r}")
    df = pd.DataFrame(r.json())
    if df.empty:
        return pd.Series(dtype=object)
    return df.drop_duplicates("sampleId").set_index("sampleId")["patientId"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-root", default="data/fetch/cbioportal_api_request")
    ap.add_argument("--portal", choices=("all", "cbio", "pedcbio"), default="all")
    ap.add_argument("--config", default=str(ROOT / "config/pipeline_config.yaml"))
    ap.add_argument("--force", action="store_true", help="refetch even if patientId is present")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}
    timeout = int(cfg.get("timeout", 60))
    portals = ("cbio", "pedcbio") if args.portal == "all" else (args.portal,)

    for portal in portals:
        base, headers = portal_settings(cfg, portal)
        for sam in sorted((Path(args.fetch_root) / portal).glob("*/*/clinical_samples.csv")):
            study_id = sam.parent.name
            samples = pd.read_csv(sam)
            has = "patientId" in samples.columns and samples["patientId"].notna().all()
            if has and not args.force:
                print(f"[ok]   {sam}: patientId already present")
                continue
            if portal == "pedcbio" and "Authorization" not in headers:
                print(f"[warn] no PedcBioPortal token found (config auth.pedcbioportal_api_token "
                      f"or env PEDCBIOPORTAL_API_TOKEN)")
            try:
                s2p = sample_to_patient(base, headers, study_id, timeout)
            except (PortalResponseError, requests.RequestException) as e:
                print(f"[fail] {sam}: {e}")
                continue
            ids = samples["sampleId"].astype(str).map(s2p)
            n_miss = int(ids.isna().sum())
            print(f"[fill] {sam}: {len(samples) - n_miss}/{len(samples)} samples linked to a patientId"
                  + (f"; {n_miss} not found in study {study_id}" if n_miss else ""))
            if args.dry_run:
                continue
            shutil.copy2(sam, sam.with_name(sam.name + ".bak"))
            samples = samples.drop(columns=["patientId"], errors="ignore")
            samples.insert(1, "patientId", ids)
            samples.to_csv(sam, index=False)


if __name__ == "__main__":
    main()
