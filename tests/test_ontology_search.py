"""Ontology-derived CCDI search (src/ontology_search.py, data_source.search_plan/data_fetch), offline."""
import pandas as pd
import pytest

from src.ontology import OntologyLayer
from src.ontology_search import CohortMatcher, cohort_terms, minimal_queries, term_regex


@pytest.fixture(scope="module")
def onto():
    return OntologyLayer("config/ontology", "src/gdcdictionary/schemas")


def test_wilms_terms_come_from_the_ontology_files():
    c = cohort_terms("wilms_tumor")
    assert c.concepts == {"NCIT:C3267": "Wilms Tumor"}
    assert {"Wilms Tumor", "Nephroblastoma", "Nephroblastoma, NOS", "8960/3", "Wilms tumour"} <= set(c.terms)
    assert "WT" not in c.terms                                     # OncoTree code, not a label
    assert set(c.table.concept_id) == {"NCIT:C3267"}
    q = c.table[c.table.queried].term.tolist()
    assert "Nephroblastoma" in q and "Nephroblastoma, NOS" not in q  # covered by the shorter query


def test_glioma_queries_are_minimal_and_skip_codes():
    c = cohort_terms("glioma")
    assert "LGGNOS" not in c.terms and "DIPG" not in c.terms
    q = [t.casefold() for t in c.table[c.table.queried].term]
    assert "glioma" in q and "glioblastoma" in q and "9421/1" in q
    assert not any(a != b and a in b for a in q for b in q)


def test_minimal_queries_and_regex():
    assert minimal_queries(["Glioma", "Low-grade glioma", "Glioblastoma"]) == ["Glioma", "Glioblastoma"]
    rx = term_regex(["Low-grade glioma/astrocytoma (WHO grade I/II)", "8960/3"])
    assert rx.search("low-grade  glioma/astrocytoma (WHO grade I/II)")
    assert rx.search("8960/3 : Nephroblastoma, NOS") and not rx.search("18960/3")


@pytest.mark.parametrize("group,value,decision,concept", [
    ("wilms_tumor", "8960/3 : Nephroblastoma, NOS", "concept", "NCIT:C3267"),
    ("wilms_tumor", "Wilms tumor", "concept", "NCIT:C3267"),
    ("wilms_tumor", "Wilms Tumor, favorable histology", "term_only", None),
    ("wilms_tumor", "Clear cell sarcoma of kidney", "no_match", None),
    ("glioma", "9421/1 : Pilocytic astrocytoma", "concept", "NCIT:C4047"),
    ("glioma", "Low-grade glioma/astrocytoma (WHO grade I/II)", "concept", "NCIT:C3059"),
    ("glioma", "Ganglioglioma", "outside_cohort", "NCIT:C3788"),
    ("glioma", "Ependymoma", "outside_cohort", "NCIT:C3017"),
])
def test_cohort_membership(onto, group, value, decision, concept):
    m = CohortMatcher(onto, cohort_terms(group))
    got = m.explain(value)
    assert got[0] == decision and got[1] == concept
    assert m.search(value) == (decision in ("concept", "term_only"))


# --- data_fetch with a fake CCDI API (case-insensitive substring search, as in its spec) ---------
RECORDS = [  # (sample_id, subject_id, diagnosis)
    ("SA1", "SU1", "8960/3 : Nephroblastoma, NOS"),
    ("SA2", "SU2", "Wilms tumor"),
    ("SA3", "SU3", "Wilms Tumor, favorable histology"),
    ("SA4", "SU4", "Clear cell sarcoma of kidney"),
    ("SA5", "SU5", "9380/3 : Glioma, malignant"),
    ("SA6", "SU6", "9421/1 : Pilocytic astrocytoma"),
    ("SA7", "SU7", "Glioblastoma"),
    ("SA8", "SU8", "Ganglioglioma"),
    ("SA9", "SU9", "Low-grade glioma/astrocytoma (WHO grade I/II)"),
    ("SA10", "SU10", "Desmoplastic infantile astrocytoma and ganglioglioma, RTK"),
]


@pytest.fixture()
def ds(monkeypatch, tmp_path):
    from src import data_source as ds

    def fake_get_page(base, term, page, per_page=100):
        hits = [r for r in RECORDS if term.casefold() in r[2].casefold()]
        if base == ds.BASE_SUBJECT:
            data = [{"id": {"name": su}, "metadata": {"associated_diagnoses": [{"value": d}]}} for _, su, d in hits]
        else:
            data = [{"id": {"name": sa}, "subject": {"name": su}, "metadata": {"diagnosis": {"value": d}}}
                    for sa, su, d in hits]
        return [{"source": "FAKE", "summary": {"counts": {"all": len(data)}}, "data": data}]

    monkeypatch.setattr(ds, "_get_page", fake_get_page)
    monkeypatch.setattr(ds, "SUBJECT_IDS_DIR", tmp_path / "ccdi/subject_ids")
    monkeypatch.setattr(ds, "SAMPLE_IDS_DIR", tmp_path / "ccdi/sample_ids")
    monkeypatch.setattr(ds, "GDM_INDEX", tmp_path / "gdm")
    monkeypatch.setattr(ds, "VERBOSE", False)
    (tmp_path / "gdm").mkdir()
    return ds


TYPES = ["Wilms Tumor", "Glioma"]


def test_data_fetch_ontology_mode_writes_cohorts_and_report(ds, tmp_path):
    ds.data_fetch(source="ontology", compare_with_config=True, types=TYPES)
    samples = {ct: set(pd.read_csv(tmp_path / f"ccdi/sample_ids/{ct}.csv").sample_id)
               for ct in ("wilms_tumor", "glioma")}
    assert samples["wilms_tumor"] == {"SA1", "SA2", "SA3"}
    assert samples["glioma"] == {"SA5", "SA6", "SA7", "SA9"}       # astrocytomas in, ganglioglioma/DIA out
    rep = pd.read_csv(tmp_path / "ccdi/search_report.tsv", sep="\t").set_index(["cancer_type", "entity"])
    g = rep.loc[("Glioma", "sample")]
    assert (g.n_ontology, g.n_fixed_list, g.n_both, g.n_ontology_only, g.n_fixed_list_only) == (4, 2, 2, 2, 0)
    w = rep.loc[("Wilms Tumor", "subject")]
    assert (w.n_ontology, w.n_fixed_list, w.n_both) == (3, 3, 3)
    vals = pd.read_csv(tmp_path / "ccdi/search_values.tsv", sep="\t")
    dia = vals[(vals.cancer_type == "Glioma") & vals.diagnosis.str.startswith("Desmoplastic")].iloc[0]
    assert dia.ontology_decision == "outside_cohort" and not dia.selected_by_ontology
    terms = pd.read_csv(tmp_path / "ccdi/search_terms.tsv", sep="\t")
    assert {"cancer_type", "cohort", "term", "origin", "concept_id", "predicate", "queried"} <= set(terms.columns)


def test_report_only_run_leaves_cohort_files_alone(ds, tmp_path):
    ds.data_fetch(write_cohorts=False, report_dir=tmp_path / "rep", source="ontology",
                  compare_with_config=True, types=TYPES)
    assert (tmp_path / "rep/search_report.tsv").exists()
    assert not (tmp_path / "ccdi/sample_ids").exists() or not any((tmp_path / "ccdi/sample_ids").iterdir())
    assert not (tmp_path / "gdm/idmap.parquet").exists()


def test_config_mode_is_the_previous_fixed_list(ds, tmp_path):
    ds.data_fetch(source="config", types=TYPES)
    glioma = set(pd.read_csv(tmp_path / "ccdi/sample_ids/glioma.csv").sample_id)
    assert glioma == {"SA5", "SA9"}                                # \bglioma\b only
    assert not (tmp_path / "ccdi/search_report.tsv").exists()


def test_unusable_source_blocks_are_logged_not_fatal(ds, tmp_path, monkeypatch):
    good = ds._get_page

    def flaky(base, term, page, per_page=100):
        return good(base, term, page, per_page) + [
            {"source": "BROKEN", "message": "upstream timeout"},           # no summary / data
            {"source": "ERR", "errors": [{"kind": "InvalidParameters"}]},
        ]

    monkeypatch.setattr(ds, "_get_page", flaky)
    ds.data_fetch(write_cohorts=False, report_dir=tmp_path / "rep", source="ontology",
                  compare_with_config=True, types=TYPES)
    sk = pd.read_csv(tmp_path / "rep/search_skipped.tsv", sep="\t")
    assert set(sk.source) == {"BROKEN", "ERR"}
    assert {"subject-diagnosis", "sample-diagnosis"} == set(sk.endpoint)
    rep = pd.read_csv(tmp_path / "rep/search_report.tsv", sep="\t").set_index(["cancer_type", "entity"])
    assert rep.loc[("Glioma", "sample")].n_ontology == 4                 # the good source still counts
