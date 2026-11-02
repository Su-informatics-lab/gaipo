"""
search_benchmark.py - Default (fixed-list) CCDI search vs the ontology-based search, on the live API.

For each cancer type, both searches are run against the CCDI Data Federation endpoints
/subject-diagnosis and /sample-diagnosis exactly as data_fetch() runs them (same paging, same
classification), and the selected subjects and samples are compared. Read-only: nothing under
data/fetch/ or data/gdm/ is written, so the cohorts used by the processor stay as they are.

Every distinct query is sent once (compared case-insensitively, as the API matches); a query used
by both searches (e.g. "glioma") counts toward the runtime of both. Queries run one at a time, as data_fetch does with search.max_workers: 1.

Outputs (in --out, default data/metadata/search_benchmark/):
  cohort_concepts.tsv   NCIt concepts of each cohort and the number of search terms per concept
  search_terms.tsv      every ontology term: origin, NCIt concept, predicate, sent as a query or not
  queries.tsv           per query x endpoint: searches using it, API requests, seconds, records per source
  runtime.tsv           per search x endpoint: queries, API requests, seconds; plus totals
  selected.tsv          every subject/sample either search selected: source, id, diagnoses,
                        selected_default, selected_ontology, retrieved_by (queries), ontology concepts
  gains_by_query.tsv    ontology-only / default-only / both, per cancer type x entity x source x query
  gains_by_concept.tsv  ontology-only records per cancer type x entity x source x NCIt concept
  review.tsv            every diagnosis value of a selected record, with counts, the ontology decision
                        and which search selects it; fill in `relevant` (yes/no) for precision
  summary.md            the main counts in one table

After filling in review.tsv:
  python scripts/search_benchmark.py --score data/metadata/search_benchmark/review.tsv
  -> precision.tsv: per cancer type x entity x search: selected, relevant, precision, and recall
     relative to all relevant records found by either search (record = relevant if any of its
     diagnosis values is labelled yes).

Run from the repository root (needs network access to federation.ccdi.cancer.gov):
  python scripts/search_benchmark.py Glioma "Wilms Tumor"
  python scripts/search_benchmark.py --terms-only          # only cohort_concepts.tsv and search_terms.tsv
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_source as ds  # noqa: E402
from src.ontology_search import load_groups  # noqa: E402

ENTITIES = (("subject", "subject-diagnosis"), ("sample", "sample-diagnosis"))


# ─── terms (offline) ─────────────────────────────────────────────────────────────

def ontology_tables(types):
    """(cohort_concepts, search_terms) for the given cancer types, from config/ontology."""
    _, _, _, terms = ds.search_plan("ontology", compare_with_config=False, types=types)
    rows = []
    groups = load_groups(ROOT / ds.SEARCH_CFG.get("ontology_dir", "config/ontology"))
    for (ct, cohort), t in terms.groupby(["cancer_type", "cohort"], sort=False):
        concepts = groups[cohort]
        for cid, label in concepts.items():
            mine = t[t.concept_id == cid]
            rows.append({"cancer_type": ct, "cohort": cohort, "concept_id": cid, "concept_label": label,
                         "n_terms": len(mine), "n_queried": int(mine.queried.sum()),
                         "terms": "; ".join(mine.term)})
    return pd.DataFrame(rows), terms


# ─── live search ─────────────────────────────────────────────────────────────────

class Recorder:
    """Wraps data_source._get_page to count API requests and time per (endpoint, query)."""

    def __init__(self):
        self.requests = defaultdict(int)
        self._orig = ds._get_page

    def __enter__(self):
        def counted(base, term, page, per_page=ds.PER_PAGE):
            self.requests[(base, term)] += 1
            return self._orig(base, term, page, per_page)
        ds._get_page = counted
        return self

    def __exit__(self, *exc):
        ds._get_page = self._orig


def run_queries(queries):
    """{(endpoint, query): (per_src, seconds)} - each distinct query once per endpoint."""
    out = {}
    for entity, endpoint in ENTITIES:
        fn = ds.harvest_subjects_for_term if entity == "subject" else ds.harvest_samples_for_term
        for q in queries:
            t0 = time.perf_counter()
            per_src = fn(q)
            out[(endpoint, q)] = (per_src, time.perf_counter() - t0)
    return out


def _ids(per_src, entity):
    key = "subjects" if entity == "subject" else "samples"
    return {(src, eid) for src, info in per_src.items() for eid in info[key]}


def benchmark(types, out):
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    fixed_terms, fixed_rx = ds._fixed_lists(types)
    missing = [t for t in types if t not in fixed_terms]
    if missing:
        raise SystemExit(f"No default_type_terms/custom_type_terms for {missing}")
    onto_queries, matchers, _, terms = ds.search_plan("ontology", compare_with_config=False, types=types)
    default_queries = [x for t in types for x in fixed_terms[t]]
    plans = {"default": default_queries, "ontology": onto_queries}

    concepts, _ = ontology_tables(types)
    concepts.to_csv(out / "cohort_concepts.tsv", sep="\t", index=False)
    terms.to_csv(out / "search_terms.tsv", sep="\t", index=False)

    # the API search is case-insensitive, so "glioma" and "Glioma" are sent once and shared
    sent = list(dict.fromkeys(q.casefold() for q in default_queries + onto_queries))
    print(f"[bench] {len(sent)} distinct queries: default {default_queries}; ontology {onto_queries}")
    ds.SKIPPED.clear()
    with Recorder() as rec:
        res = run_queries(sent)
    ds.write_skipped(out)
    plans = {m: [q.casefold() for q in qs] for m, qs in plans.items()}
    distinct = sent

    # per query x endpoint
    qrows = []
    for (endpoint, q), (per_src, sec) in res.items():
        base = ds.BASE_SUBJECT if endpoint == "subject-diagnosis" else ds.BASE_SAMPLE
        qrows.append({"endpoint": endpoint, "query": q,
                      "used_by": "+".join(m for m, qs in plans.items() if q in qs),
                      "requests": rec.requests[(base, q)], "seconds": round(sec, 2),
                      **{f"records_{s}": i["seen"] for s, i in sorted(per_src.items())}})
    qdf = pd.DataFrame(qrows).fillna(0)
    qdf.to_csv(out / "queries.tsv", sep="\t", index=False)

    # runtime per search
    rrows = []
    for mode, qs in plans.items():
        for _, endpoint in ENTITIES:
            sub = qdf[(qdf.endpoint == endpoint) & qdf["query"].isin(qs)]
            rrows.append({"search": mode, "endpoint": endpoint, "queries": len(sub),
                          "requests": int(sub.requests.sum()), "seconds": round(sub.seconds.sum(), 1)})
        sub = qdf[qdf["query"].isin(qs)]
        rrows.append({"search": mode, "endpoint": "both", "queries": len(qs),
                      "requests": int(sub.requests.sum()), "seconds": round(sub.seconds.sum(), 1)})
    rdf = pd.DataFrame(rrows)
    rdf.to_csv(out / "runtime.tsv", sep="\t", index=False)

    # selection by each search (same merge + classification as data_fetch)
    sel_rows, review = [], defaultdict(lambda: {"n": 0, "sources": set()})
    for entity, endpoint in ENTITIES:
        merge = ds.merge_subjects if entity == "subject" else ds.merge_samples
        classify = ds.classify_entities if entity == "subject" else ds.classify_samples
        dkey = "diagnosis" if entity == "subject" else "diagnoses"
        selected = {}
        merged_all = merge([(q, res[(endpoint, q)][0]) for q in distinct])
        for mode, qs in plans.items():
            merged = merge([(q, res[(endpoint, q)][0]) for q in qs])
            rx = fixed_rx if mode == "default" else matchers
            cls = classify(merged, rx)
            selected[mode] = {ct: {(src, eid) for src, emap in cls.items() for eid, info in emap.items()
                                   if ct in info["matched_types"]} for ct in types}
        retrieved = {q: _ids(res[(endpoint, q)][0], entity) for q in distinct}
        for ct in types:
            m = matchers[ct]
            for key in sorted(selected["default"][ct] | selected["ontology"][ct]):
                src, eid = key
                diags = sorted(merged_all[src][eid][dkey])
                decisions = [m.explain(v) for v in diags]
                sel_rows.append({
                    "cancer_type": ct, "entity": entity, "source": src, "id": eid,
                    "diagnoses": " | ".join(diags),
                    "selected_default": key in selected["default"][ct],
                    "selected_ontology": key in selected["ontology"][ct],
                    "retrieved_by": "; ".join(q for q in distinct if key in retrieved[q]),
                    "ontology_concepts": "; ".join(sorted({d[2] for d in decisions if d[0] == "concept"})),
                    "ontology_decisions": "; ".join(sorted({d[0] for d in decisions})),
                })
                for v, d in zip(diags, decisions):
                    r = review[(ct, v)]
                    r["n"] += 1
                    r["sources"].add(src)
                    r[f"n_{entity}s"] = r.get(f"n_{entity}s", 0) + 1
                    r["decision"], r["concept"] = d[0], d[2]
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(out / "selected.tsv", sep="\t", index=False)

    # gains per query and per concept
    def status(r):
        return ("both" if r.selected_default and r.selected_ontology
                else "ontology_only" if r.selected_ontology else "default_only")
    gq, gc = [], []
    if not sel.empty:
        sel["status"] = sel.apply(status, axis=1)
        for _, r in sel.iterrows():
            for q in (r.retrieved_by.split("; ") if r.retrieved_by else ["(none)"]):
                gq.append({"cancer_type": r.cancer_type, "entity": r.entity, "source": r.source,
                           "query": q, "status": r.status})
            if r.status == "ontology_only":
                for c in (r.ontology_concepts.split("; ") if r.ontology_concepts else ["(unresolved)"]):
                    gc.append({"cancer_type": r.cancer_type, "entity": r.entity, "source": r.source,
                               "concept": c})
    keys = ["cancer_type", "entity", "source", "query"]
    gq = pd.DataFrame(gq, columns=keys + ["status"])
    gq = pd.crosstab([gq[k] for k in keys], gq.status).reset_index() if len(gq) else pd.DataFrame(columns=keys)
    for c in ("ontology_only", "default_only", "both"):
        if c not in gq:
            gq[c] = 0
    gq = gq[keys + ["ontology_only", "default_only", "both"]]
    gq.to_csv(out / "gains_by_query.tsv", sep="\t", index=False)
    gc = pd.DataFrame(gc, columns=["cancer_type", "entity", "source", "concept"])
    gc = gc.groupby(list(gc.columns)).size().rename("ontology_only").reset_index()
    gc.to_csv(out / "gains_by_concept.tsv", sep="\t", index=False)

    # review sheet (one row per cancer type x diagnosis value)
    rv = []
    for (ct, v), r in review.items():
        rv.append({"cancer_type": ct, "diagnosis": v, "n_subjects": r.get("n_subjects", 0),
                   "n_samples": r.get("n_samples", 0), "sources": ", ".join(sorted(r["sources"])),
                   "ontology_decision": r["decision"], "ontology_concept": r["concept"],
                   "selects_default": bool(fixed_rx[ct].search(v)), "selects_ontology": matchers[ct].search(v),
                   "relevant": "", "note": ""})
    rv = pd.DataFrame(rv)
    if not rv.empty:
        rv = rv.sort_values(["cancer_type", "n_samples", "n_subjects"], ascending=[True, False, False])
    rv.to_csv(out / "review.tsv", sep="\t", index=False)

    write_summary(out, types, plans, sel, rdf, time.perf_counter() - t_start)
    print(f"[bench] wrote {out}/: cohort_concepts, search_terms, queries, runtime, selected, "
          f"gains_by_query, gains_by_concept, review (.tsv), summary.md")


def write_summary(out, types, plans, sel, rdf, wall):
    lines = ["# Default vs ontology-based CCDI search", "",
             f"Queries - default: {', '.join(plans['default'])}; ontology: {', '.join(plans['ontology'])}", "",
             "| Cancer type | Entity | Default | Ontology | Both | Ontology only | Default only |",
             "|---|---|---|---|---|---|---|"]
    for ct in types:
        for entity, _ in ENTITIES:
            s = sel[(sel.cancer_type == ct) & (sel.entity == entity)] if not sel.empty else sel
            d = int(s.selected_default.sum()) if len(s) else 0
            o = int(s.selected_ontology.sum()) if len(s) else 0
            b = int((s.selected_default & s.selected_ontology).sum()) if len(s) else 0
            lines.append(f"| {ct} | {entity}s | {d} | {o} | {b} | {o - b} | {d - b} |")
    lines += ["", "| Search | Queries | API requests | Seconds (subjects + samples) |", "|---|---|---|---|"]
    for _, r in rdf[rdf.endpoint == "both"].iterrows():
        lines.append(f"| {r.search} | {r.queries} | {r.requests} | {r.seconds} |")
    lines += ["", f"Benchmark wall time: {wall:.0f} s (each distinct query sent once).",
              "Precision: fill in `relevant` in review.tsv, then run with --score."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ─── precision from the filled-in review sheet ───────────────────────────────────

def score(review_path):
    review_path = Path(review_path)
    rv = pd.read_csv(review_path, sep="\t", dtype=str).fillna("")
    lab = rv[rv.relevant.str.strip().str.lower().isin(["yes", "no", "y", "n", "1", "0"])]
    if len(lab) < len(rv):
        print(f"[score] {len(rv) - len(lab)} of {len(rv)} values have no yes/no label; "
              "records whose values are all unlabelled are left out")
    yes = {(r.cancer_type, r.diagnosis) for r in lab.itertuples()
           if r.relevant.strip().lower() in ("yes", "y", "1")}
    labelled = {(r.cancer_type, r.diagnosis) for r in lab.itertuples()}
    sel = pd.read_csv(review_path.parent / "selected.tsv", sep="\t", dtype=str).fillna("")
    rows = []
    for (ct, entity), g in sel.groupby(["cancer_type", "entity"]):
        recs = []
        for r in g.itertuples():
            vals = [(ct, v) for v in r.diagnoses.split(" | ") if v]
            if not any(v in labelled for v in vals):
                continue
            recs.append((r.selected_default == "True", r.selected_ontology == "True",
                         any(v in yes for v in vals)))
        relevant_any = sum(1 for _, _, rel in recs if rel)
        for i, mode in enumerate(("default", "ontology")):
            s = [rel for *flags, rel in recs if flags[i]]
            tp = sum(s)
            rows.append({"cancer_type": ct, "entity": entity, "search": mode, "selected": len(s),
                         "relevant": tp, "precision": round(tp / len(s), 3) if s else None,
                         "recall_vs_union": round(tp / relevant_any, 3) if relevant_any else None})
    p = pd.DataFrame(rows)
    p.to_csv(review_path.parent / "precision.tsv", sep="\t", index=False)
    print(p.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cancer_types", nargs="*", help="default: cancer_type_interest in pipeline_config.yaml")
    ap.add_argument("--out", default="data/metadata/search_benchmark")
    ap.add_argument("--terms-only", action="store_true", help="write cohort_concepts.tsv and search_terms.tsv only")
    ap.add_argument("--score", metavar="REVIEW_TSV", help="compute precision from a filled-in review.tsv")
    args = ap.parse_args()
    if args.score:
        return score(args.score)
    types = args.cancer_types or list(ds.cancer_type_interest)
    out = Path(args.out)
    if args.terms_only:
        out.mkdir(parents=True, exist_ok=True)
        concepts, terms = ontology_tables(types)
        concepts.to_csv(out / "cohort_concepts.tsv", sep="\t", index=False)
        terms.to_csv(out / "search_terms.tsv", sep="\t", index=False)
        print(concepts[["cancer_type", "concept_id", "concept_label", "n_terms", "n_queried"]].to_string(index=False))
        return
    benchmark(types, out)


if __name__ == "__main__":
    main()
