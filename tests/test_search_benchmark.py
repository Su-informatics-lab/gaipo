"""scripts/search_benchmark.py with a fake CCDI API (case-insensitive substring search, as in its spec)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORDS = [  # (source, sample_id, subject_id, diagnosis)
    ("A", "SA1", "SU1", "9380/3 : Glioma, malignant"),
    ("A", "SA2", "SU2", "Low-grade glioma/astrocytoma (WHO grade I/II)"),
    ("A", "SA3", "SU3", "9421/1 : Pilocytic astrocytoma"),
    ("B", "SB1", "SV1", "Glioblastoma"),
    ("B", "SB2", "SV2", "Ganglioglioma"),
    ("B", "SB3", "SV3", "Desmoplastic infantile astrocytoma and ganglioglioma, RTK"),
    ("B", "SB4", "SV4", "Wilms tumor"),
]


@pytest.fixture()
def bench(monkeypatch):
    from src import data_source as ds
    spec = importlib.util.spec_from_file_location("search_benchmark", ROOT / "scripts/search_benchmark.py")
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)

    def fake_get_page(base, term, page, per_page=100):
        out = []
        for src in ("A", "B"):
            hits = [r for r in RECORDS if r[0] == src and term.casefold() in r[3].casefold()]
            if base == ds.BASE_SUBJECT:
                data = [{"id": {"name": su}, "metadata": {"associated_diagnoses": [{"value": d}]}}
                        for _, _, su, d in hits]
            else:
                data = [{"id": {"name": sa}, "subject": {"name": su}, "metadata": {"diagnosis": {"value": d}}}
                        for _, sa, su, d in hits]
            out.append({"source": src, "summary": {"counts": {"all": len(data)}}, "data": data})
        return out

    monkeypatch.setattr(ds, "_get_page", fake_get_page)
    monkeypatch.setattr(ds, "VERBOSE", False)
    return sb


def test_benchmark_counts_gains_runtime_and_precision(bench, tmp_path):
    bench.benchmark(["Glioma"], tmp_path)
    sel = pd.read_csv(tmp_path / "selected.tsv", sep="\t")
    samples = sel[sel.entity == "sample"].set_index("id")
    # default list: whole-word "glioma" etc., so ganglioglioma is retrieved but not selected
    assert set(samples.index[samples.selected_default]) == {"SA1", "SA2"}
    # ontology adds glioblastoma and pilocytic astrocytoma; DIA/ganglioglioma resolve outside the cohort
    assert set(samples.index[samples.selected_ontology]) == {"SA1", "SA2", "SA3", "SB1"}
    assert "glioblastoma" in samples.loc["SB1", "retrieved_by"]

    gq = pd.read_csv(tmp_path / "gains_by_query.tsv", sep="\t")
    s = gq[(gq.entity == "sample")].groupby("query").ontology_only.sum()
    assert s["glioblastoma"] == 1 and s["astrocytoma"] == 1      # SB1 and SA3 (SA2 is in both)
    gc = pd.read_csv(tmp_path / "gains_by_concept.tsv", sep="\t")
    assert set(gc[gc.entity == "sample"].concept) == {"Glioblastoma", "Pilocytic Astrocytoma"}

    rt = pd.read_csv(tmp_path / "runtime.tsv", sep="\t").set_index(["search", "endpoint"])
    assert rt.loc[("default", "both"), "queries"] == 3 and rt.loc[("ontology", "both"), "queries"] == 3
    assert rt.loc[("ontology", "both"), "requests"] == 6                 # 3 queries x 2 endpoints, 1 page each
    q = pd.read_csv(tmp_path / "queries.tsv", sep="\t")
    assert len(q) == 2 * 5 and q.loc[q["query"] == "glioma", "used_by"].iloc[0] == "default+ontology"
    assert (tmp_path / "cohort_concepts.tsv").exists() and (tmp_path / "summary.md").exists()

    rv = pd.read_csv(tmp_path / "review.tsv", sep="\t", dtype=str).fillna("")
    assert set(rv.diagnosis) == {r[3] for r in RECORDS[:4]}
    rv["relevant"] = ["yes"] * len(rv)
    rv.to_csv(tmp_path / "review.tsv", sep="\t", index=False)
    bench.score(tmp_path / "review.tsv")
    p = pd.read_csv(tmp_path / "precision.tsv", sep="\t").set_index(["entity", "search"])
    assert p.loc[("sample", "ontology"), "precision"] == 1.0
    assert p.loc[("sample", "default"), "recall_vs_union"] == 0.5


def test_terms_only_lists_the_cohort_concepts(bench, tmp_path):
    concepts, terms = bench.ontology_tables(["Glioma"])
    assert len(concepts) == 12 and concepts.n_terms.sum() == len(terms) == 93
