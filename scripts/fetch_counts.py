"""
fetch_counts.py - How many subjects and samples data_fetch and data_extract produced, and what
changed since an earlier run. Read-only: it only reads the fetch folders and writes a report.

Counts (current run, under data/fetch/):
  ccdi_search    per cancer type x CCDI source: subject and sample IDs found by data_fetch
                 (data/fetch/ccdi_api_request/{subject_ids,sample_ids}/<type>.csv), plus "ALL" =
                 unique IDs across sources after the ID normalization data_extract uses
                 ("pt-ab12" and "PT_AB12" are the same patient)
  extracted      per cancer type x portal:study: patients and samples data_extract kept
                 (clinical_patients.csv / clinical_samples.csv), samples per omics file, and
                 patients with at least one sample in every omics file
Comparison (--compare DIR): the same tables for an earlier run kept in DIR (a folder holding
ccdi_api_request/ and, optionally, cbioportal_api_request/), and the IDs added or removed.

Run from the repository root:
    python scripts/fetch_counts.py
    python scripts/fetch_counts.py --compare data/fetch/fixed_search_terms
Outputs in --out (default data/metadata/fetch_counts/):
    ccdi_search.tsv, extracted.tsv, changes.tsv (with --compare), added_<type>_<entity>.tsv,
    removed_<type>_<entity>.tsv, summary.md
"""
import argparse
from pathlib import Path

import pandas as pd

ENTITIES = (("subject", "subject_ids", "subject_id"), ("sample", "sample_ids", "sample_id"))


def md(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    cols = [str(c) for c in df.columns]
    cells = [["" if pd.isna(v) else str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(r) + " |" for r in cells])


def norm(x: str) -> str:
    """Same as extractor.norm_patient_id_for_cbio / norm_sample_id_for_cbio."""
    x = str(x).strip()
    return x.upper().replace("-", "_") if any(c.islower() for c in x) else x


def ccdi_ids(root: Path) -> dict:
    """{(type_slug, entity): DataFrame[source, id, nid]} from <root>/ccdi_api_request/."""
    out = {}
    for entity, folder, col in ENTITIES:
        for f in sorted((root / "ccdi_api_request" / folder).glob("*.csv")):
            df = pd.read_csv(f, usecols=["source", col], dtype=str).dropna()
            df = df.rename(columns={col: "id"}).drop_duplicates()
            df["nid"] = df["id"].map(norm)
            out[(f.stem, entity)] = df
    return out


def ccdi_table(ids: dict) -> pd.DataFrame:
    rows = []
    for (ct, entity), df in ids.items():
        for src, g in df.groupby("source"):
            rows.append({"cancer_type": ct, "entity": entity, "source": src, "ids": g["id"].nunique()})
        rows.append({"cancer_type": ct, "entity": entity, "source": "ALL (unique)", "ids": df["nid"].nunique()})
    return pd.DataFrame(rows, columns=["cancer_type", "entity", "source", "ids"])


def extracted_table(root: Path) -> pd.DataFrame:
    rows = []
    for pat in sorted((root / "cbioportal_api_request").glob("*/*/*/clinical_patients.csv")):
        d = pat.parent
        portal, ct, study = d.parts[-3:]
        p = pd.read_csv(pat, dtype=str)
        s = pd.read_csv(d / "clinical_samples.csv", dtype=str) if (d / "clinical_samples.csv").exists() else pd.DataFrame()
        s2p = (s.dropna(subset=["patientId"]).set_index("sampleId")["patientId"]
               if {"sampleId", "patientId"} <= set(s.columns) else pd.Series(dtype=str))
        row = {"cancer_type": ct, "portal": portal, "study_id": study,
               "patients": p["patientId"].nunique() if "patientId" in p else len(p),
               "samples": s["sampleId"].nunique() if "sampleId" in s else len(s)}
        with_all = None
        for tsv in sorted(d.glob("*.tsv")):
            cols = pd.read_csv(tsv, sep="\t", nrows=0).columns
            samples = [c for c in cols if c not in ("hugoGeneSymbol", "entrezGeneId")]
            row[f"samples_in:{tsv.stem}"] = len(samples)
            pts = set(s2p.reindex(samples).dropna())
            with_all = pts if with_all is None else with_all & pts
        row["patients_with_every_omics_file"] = len(with_all) if with_all is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def compare(new: dict, old: dict, out: Path) -> pd.DataFrame:
    rows = []
    for key in sorted(set(new) | set(old)):
        ct, entity = key
        n = new.get(key, pd.DataFrame(columns=["source", "id", "nid"]))
        o = old.get(key, pd.DataFrame(columns=["source", "id", "nid"]))
        added, removed = set(n["nid"]) - set(o["nid"]), set(o["nid"]) - set(n["nid"])
        rows.append({"cancer_type": ct, "entity": entity, "earlier": o["nid"].nunique(),
                     "now": n["nid"].nunique(), "added": len(added), "removed": len(removed)})
        for name, ids, df in (("added", added, n), ("removed", removed, o)):
            sub = df[df["nid"].isin(ids)].sort_values(["nid", "source"])
            sub.to_csv(out / f"{name}_{ct}_{entity}.tsv", sep="\t", index=False)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-root", default="data/fetch", help="folder with ccdi_api_request/ and cbioportal_api_request/")
    ap.add_argument("--compare", metavar="DIR", help="earlier run, e.g. data/fetch/fixed_search_terms")
    ap.add_argument("--out", default="data/metadata/fetch_counts")
    args = ap.parse_args()
    root, out = Path(args.fetch_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    new = ccdi_ids(root)
    ccdi = ccdi_table(new)
    ext = extracted_table(root)
    ccdi.to_csv(out / "ccdi_search.tsv", sep="\t", index=False)
    ext.to_csv(out / "extracted.tsv", sep="\t", index=False)
    lines = ["# data_fetch / data_extract counts", "", "## CCDI search (data_fetch)", "",
             md(ccdi) if len(ccdi) else "(no subject_ids/sample_ids CSVs found)", "",
             "## Extracted from cBioPortal / PedcBioPortal (data_extract)", "",
             md(ext) if len(ext) else "(no clinical_patients.csv found)"]

    if args.compare:
        prev = Path(args.compare)
        ch = compare(new, ccdi_ids(prev), out)
        ch.to_csv(out / "changes.tsv", sep="\t", index=False)
        lines += ["", f"## Compared with {prev}", "", "IDs are unique across sources after normalization; "
                  "added_*/removed_*.tsv list them with their source.", "", md(ch)]
        if (prev / "cbioportal_api_request").exists():
            pe = extracted_table(prev)
            if len(pe) and len(ext):
                m = pe.merge(ext, on=["cancer_type", "portal", "study_id"], how="outer", suffixes=("_earlier", "_now"))
                m = m[["cancer_type", "portal", "study_id", "patients_earlier", "patients_now",
                       "samples_earlier", "samples_now"]]
                m.to_csv(out / "changes_extracted.tsv", sep="\t", index=False)
                lines += ["", "### Extracted patients and samples, earlier vs now", "", md(m)]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)
    print(f"[fetch_counts] wrote {out}/")


if __name__ == "__main__":
    main()
