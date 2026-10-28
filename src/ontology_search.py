# src/ontology_search.py
"""
ontology_search.py - CCDI search terms and cohort membership from the ontology layer

Used by data_source.py (data_fetch) when `search.source: ontology` in pipeline_config.yaml.
Nothing here is typed per cancer type: a cohort is a set of NCIt concepts in
config/ontology/analysis_groups.tsv (e.g. "wilms_tumor" = {C3267}; "glioma" = {C3059,
C4822, C3058, ...}), and everything else is read from the SSSOM files.

1) Query terms. The CCDI Data Federation API matches the `search` string as a
   case-insensitive substring of a sample's `diagnosis` (/sample-diagnosis) or of any of a
   subject's `associated_diagnoses` (/subject-diagnosis). The terms for a cohort are:
     - the NCIt label of every cohort concept                       (analysis_groups.tsv)
     - every source label mapped to a cohort concept with an exact, close or broad match
       (broad = the source term is narrower than the concept)       (source_terms.sssom.tsv)
     - ICD-O-3 morphology codes and labels of the cohort concepts   (source_terms.sssom.tsv,
       exact or close match                                          concept_xrefs.sssom.tsv)
   OncoTree codes used as labels (e.g. "LGGNOS") and terms shorter than `min_length` are
   skipped. A term that contains another term is not sent as its own query, because the
   shorter term's substring search already returns those records.

2) Cohort membership. Each returned diagnosis value is resolved to an NCIt concept with the
   ontology layer (label, then ICD-O-3 code, then the label part of "code : label"):
     concept        resolves to a cohort concept                          -> kept
     outside_cohort resolves to a concept that is not in the cohort       -> dropped
     term_only      does not resolve, but a cohort term occurs in it as a whole word
                    -> kept when include_unresolved is true (listed for curation)
     no_match       anything else                                         -> dropped
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from .ontology import OntologyLayer, parse_icdo3_morphology, read_sssom

PathLike = Union[str, Path]

COHORT_PREDICATES = ("skos:exactMatch", "skos:closeMatch", "skos:broadMatch")
XREF_PREDICATES = ("skos:exactMatch", "skos:closeMatch")


def load_groups(ontology_dir: PathLike) -> Dict[str, Dict[str, str]]:
    """analysis_groups.tsv -> {group: {concept_id: concept_label}}."""
    g = pd.read_csv(Path(ontology_dir) / "analysis_groups.tsv", sep="\t", comment="#", dtype=str)
    return {name: dict(zip(df.concept_id, df.concept_label)) for name, df in g.groupby("group")}


def _key(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip()).casefold()


def minimal_queries(terms: Iterable[str]) -> List[str]:
    """Drop every term that contains another term (case-insensitive substring)."""
    uniq = {}
    for t in terms:
        uniq.setdefault(_key(t), t)
    keys = sorted(uniq, key=len)
    keep: List[str] = []
    for k in keys:
        if not any(s in k for s in keep):
            keep.append(k)
    return [uniq[k] for k in keep]


def term_regex(terms: Iterable[str]) -> re.Pattern:
    """Whole-word match of any term; also works for terms that start or end with punctuation."""
    alts = sorted({_key(t) for t in terms if t.strip()}, key=len, reverse=True)
    if not alts:
        return re.compile(r"(?!x)x")
    body = "|".join(re.escape(a).replace(r"\ ", r"\s+") for a in alts)
    return re.compile(r"(?<!\w)(?:" + body + r")(?!\w)", re.I)


@dataclass
class CohortTerms:
    group: str
    concepts: Dict[str, str]      # concept_id -> NCIt label
    table: pd.DataFrame           # term, origin, concept_id, predicate, queried

    @property
    def terms(self) -> List[str]:
        return self.table.term.tolist()


def cohort_terms(group: str, ontology_dir: PathLike = "config/ontology",
                 min_length: int = 4) -> CohortTerms:
    d = Path(ontology_dir)
    groups = load_groups(d)
    if group not in groups:
        raise KeyError(f"Cohort {group!r} is not defined in {d / 'analysis_groups.tsv'} "
                       f"(defined: {sorted(groups)})")
    concepts = groups[group]
    rows = [{"term": label, "origin": "NCIt label (analysis_groups.tsv)", "concept_id": cid,
             "predicate": "self"} for cid, label in concepts.items()]

    src = read_sssom(d / "source_terms.sssom.tsv")
    src = src[src.object_id.isin(concepts) & src.predicate_id.isin(COHORT_PREDICATES)]
    for r in src.itertuples(index=False):
        code = r.subject_id.split(":", 1)[1] if ":" in r.subject_id else ""
        if r.subject_source == "ONCOTREE" and r.subject_label.strip().upper() == code.upper():
            continue                                    # an OncoTree code, not a label
        rows.append({"term": r.subject_label, "origin": f"source_terms.sssom.tsv ({r.subject_source})",
                     "concept_id": r.object_id, "predicate": r.predicate_id})
        if r.subject_source == "ICDO3M":
            rows.append({"term": code, "origin": "source_terms.sssom.tsv (ICD-O-3 code)",
                         "concept_id": r.object_id, "predicate": r.predicate_id})

    xr = read_sssom(d / "concept_xrefs.sssom.tsv")
    xr = xr[xr.subject_id.isin(concepts) & xr.predicate_id.isin(XREF_PREDICATES)
            & xr.object_id.str.startswith("ICDO3M:")]
    for r in xr.itertuples(index=False):
        for term, what in ((r.object_id.split(":", 1)[1], "ICD-O-3 code"), (r.object_label, "ICD-O-3 label")):
            rows.append({"term": term, "origin": f"concept_xrefs.sssom.tsv ({what})",
                         "concept_id": r.subject_id, "predicate": r.predicate_id})

    t = pd.DataFrame(rows)
    t["term"] = t.term.astype(str).str.strip()
    t = t[t.term.str.len() >= min_length]
    t = t.assign(_k=t.term.map(_key)).drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    q = {_key(x) for x in minimal_queries(t.term)}
    t["queried"] = t.term.map(lambda x: _key(x) in q)
    return CohortTerms(group, concepts, t)


class CohortMatcher:
    """Decides cohort membership of one diagnosis value; `search(value)` mirrors re.Pattern.search
    so data_source.classify_* can use it in place of a regex."""

    def __init__(self, onto: OntologyLayer, cohort: CohortTerms, include_unresolved: bool = True):
        self.onto = onto
        self.cohort = cohort
        self.include_unresolved = include_unresolved
        self.regex = term_regex(cohort.terms)
        xr = onto.xrefs[onto.xrefs.object_id.str.startswith("ICDO3M:")
                        & onto.xrefs.predicate_id.isin(XREF_PREDICATES)]
        self._xref_code = {}
        for r in xr.itertuples(index=False):
            self._xref_code.setdefault(r.object_id.split(":", 1)[1], (r.subject_id, r.subject_label))
        self._cache: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {}

    def _concept(self, value: str) -> Tuple[Optional[str], Optional[str]]:
        r = self.onto.resolve("TEXT", value)
        if r.ok:
            return r.concept_id, r.concept_label
        code = parse_icdo3_morphology(value)
        if code and code in self._xref_code:
            return self._xref_code[code]
        if " : " in value:                              # CCDI style "9421/1 : Pilocytic astrocytoma"
            r = self.onto.resolve("TEXT", value.split(" : ", 1)[1])
            if r.ok:
                return r.concept_id, r.concept_label
        return None, None

    def explain(self, value: object) -> Tuple[str, Optional[str], Optional[str]]:
        """(decision, concept_id, concept_label); decision in concept/outside_cohort/term_only/no_match."""
        v = "" if value is None else str(value).strip()
        if v not in self._cache:
            cid, label = self._concept(v) if v else (None, None)
            if cid is not None:
                decision = "concept" if cid in self.cohort.concepts else "outside_cohort"
            else:
                decision = "term_only" if self.regex.search(v) else "no_match"
            self._cache[v] = (decision, cid, label)
        return self._cache[v]

    def search(self, value: object) -> bool:
        decision = self.explain(value)[0]
        return decision == "concept" or (decision == "term_only" and self.include_unresolved)
