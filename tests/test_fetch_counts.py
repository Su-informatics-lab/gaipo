"""scripts/fetch_counts.py on a small fetch tree (no network)."""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_counts", ROOT / "scripts/fetch_counts.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)


def _ccdi(root, subjects, samples):
    for folder, col, rows in (("subject_ids", "subject_id", subjects), ("sample_ids", "sample_id", samples)):
        d = root / "ccdi_api_request" / folder
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["source", col, "diagnosis"]).to_csv(d / "glioma.csv", index=False)


def _study(root):
    d = root / "cbioportal_api_request/pedcbio/glioma/pbta_all"
    d.mkdir(parents=True)
    pd.DataFrame({"patientId": ["PT_A", "PT_B", "PT_C"]}).to_csv(d / "clinical_patients.csv", index=False)
    pd.DataFrame({"sampleId": ["BS_1", "BS_2", "BS_3", "BS_4"],
                  "patientId": ["PT_A", "PT_A", "PT_B", "PT_C"]}).to_csv(d / "clinical_samples.csv", index=False)
    for name, samples in (("pbta_all_rna_seq_mrna.tsv", ["BS_1", "BS_3"]), ("pbta_all_linear_CNA.tsv", ["BS_2", "BS_4"])):
        pd.DataFrame(columns=["hugoGeneSymbol", "entrezGeneId"] + samples).to_csv(d / name, sep="\t", index=False)


def test_counts_and_comparison(tmp_path, monkeypatch):
    now, before = tmp_path / "fetch", tmp_path / "fetch/fixed_search_terms"
    _ccdi(now, [("KidsFirst", "pt-a", "x"), ("CCDI-DCC", "PT_A", "x"), ("KidsFirst", "pt-b", "x"),
                ("StJude", "SJ1", "x")],
          [("KidsFirst", "bs-1", "x"), ("KidsFirst", "bs-2", "x")])
    _ccdi(before, [("KidsFirst", "pt-a", "x"), ("StJude", "SJ0", "x")], [("KidsFirst", "bs-1", "x")])
    _study(now)
    monkeypatch.setattr("sys.argv", ["fetch_counts.py", "--fetch-root", str(now), "--compare", str(before),
                                     "--out", str(tmp_path / "out")])
    fc.main()

    c = pd.read_csv(tmp_path / "out/ccdi_search.tsv", sep="\t").set_index(["entity", "source"])
    assert c.loc[("subject", "ALL (unique)"), "ids"] == 3          # pt-a and PT_A are one patient
    assert c.loc[("subject", "KidsFirst"), "ids"] == 2

    e = pd.read_csv(tmp_path / "out/extracted.tsv", sep="\t").iloc[0]
    assert (e.patients, e.samples) == (3, 4)
    assert e.patients_with_every_omics_file == 1                    # PT_A (BS_1 + BS_2); PT_B no CNA, PT_C no mRNA
    ch = pd.read_csv(tmp_path / "out/changes.tsv", sep="\t").set_index("entity")
    assert tuple(ch.loc["subject", ["earlier", "now", "added", "removed"]]) == (2, 3, 2, 1)
    added = pd.read_csv(tmp_path / "out/added_glioma_subject.tsv", sep="\t")
    assert set(added.nid) == {"PT_B", "SJ1"}
    assert (tmp_path / "out/summary.md").read_text().count("|") > 10
