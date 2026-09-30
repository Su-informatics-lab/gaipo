# src/ontology.py
"""
ontology.py - Ontology layer for metadata semantic mapping in GAIPO

Replaces string-to-string lookups with concept-based mapping:

    source value --(SSSOM mapping)--> NCIt concept --(GDC termDef / ICD-O-3 rule)--> GDC value
                                            |
                                            +--> ICD-O-3, SNOMED CT, ICD-10-CM, OMOP cross-references

Mapping knowledge lives in data files under config/ontology/, not in code:

    element_map.yaml               element level: source column -> standard data element (priority order)
    source_terms.sssom.tsv         value level:   source code/label -> NCIt concept
    concept_xrefs.sssom.tsv        NCIt concept -> ICD-O-3 / SNOMED CT / ICD-10-CM
    concept_relations.tsv          NCIt disease -> typical primary site (fallback inference)
    icdo3_morphology_blocks.tsv    ICD-O-3 morphology block -> GDC case.disease_type

NCIt concept -> GDC permissible value is NOT typed by hand: it is read from the termDef
(NCIt code) that the bundled GDC dictionary (_terms_enum.yaml) attaches to every enum value.

Resolution is deterministic (exact code match, then normalised-label match). Anything that
does not resolve is returned as unmapped and written to a review file; nothing is guessed.

Usage
-----
    from src.ontology import OntologyLayer
    onto = OntologyLayer("config/ontology", "src/gdcdictionary/schemas")
    ann  = onto.annotate_cases(patients_df, samples_df)   # one row per subject
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import yaml

PathLike = Union[str, Path]


# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------

def read_sssom(path: PathLike) -> pd.DataFrame:
    """Read an SSSOM TSV (YAML metadata block in '#' lines is skipped)."""
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, keep_default_na=False)


def normalize_label(text: object) -> str:
    """Case/punctuation-insensitive key for label matching ("Wilms' Tumor" == "wilms tumor")."""
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    s = s.lower().replace("'s ", " ").replace("'", "")
    s = re.sub(r"[^0-9a-z]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def gdc_term_key(value: object) -> str:
    """Key used by GDC _terms_enum.yaml for an enum value ("Kidney, NOS" -> "kidney_nos", "8960/3" -> "89603")."""
    s = re.sub(r"[^0-9a-z ]+", "", str(value).lower())
    return re.sub(r"\s+", "_", s.strip())


_MORPH_RE = re.compile(r"(\d{4})\s*/\s*(\d)")
_TOPO_RE = re.compile(r"^C?(\d{2})\.?(\d)?$")


def parse_icdo3_morphology(value: object) -> Optional[str]:
    """'8960/3', '8960/3 : Nephroblastoma, NOS' -> '8960/3'."""
    m = _MORPH_RE.search(str(value))
    return f"{m.group(1)}/{m.group(2)}" if m else None


def parse_icdo3_topography(value: object) -> Optional[str]:
    """'C649', 'c64.9', 'C64' -> 'C64.9' (missing digit -> '.9', as in OMOP ICDO3 ETL)."""
    s = re.sub(r"\s+", "", str(value)).upper()
    m = _TOPO_RE.match(s)
    if not m:
        return None
    return f"C{m.group(1)}.{m.group(2) or '9'}"


# ------------------------------------------------------------------------------
# Result record
# ------------------------------------------------------------------------------

@dataclass
class Resolution:
    concept_id: Optional[str]          # e.g. "NCIT:C3267"
    concept_label: Optional[str]
    predicate: Optional[str]           # skos:exactMatch / broadMatch / ...
    justification: Optional[str]       # semapv:ManualMappingCuration / ...
    confidence: Optional[float]
    source_table: Optional[str] = None
    source_column: Optional[str] = None
    source_value: Optional[str] = None
    matched_on: Optional[str] = None   # the SSSOM subject_id that matched
    extras: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.concept_id is not None


_UNRESOLVED = Resolution(None, None, None, None, None)


# ------------------------------------------------------------------------------
# Ontology layer
# ------------------------------------------------------------------------------

class OntologyLayer:
    def __init__(self, ontology_dir: PathLike = "config/ontology",
                 schemas_dir: PathLike = "src/gdcdictionary/schemas"):
        d = Path(ontology_dir)
        self.dir = d
        self.elements: dict = yaml.safe_load(open(d / "element_map.yaml"))["elements"]
        self.terms = read_sssom(d / "source_terms.sssom.tsv")
        self.xrefs = read_sssom(d / "concept_xrefs.sssom.tsv")
        self.relations = pd.read_csv(d / "concept_relations.tsv", sep="\t", comment="#", dtype=str)
        self.blocks = pd.read_csv(d / "icdo3_morphology_blocks.tsv", sep="\t", comment="#",
                                  dtype={"start": int, "end": int, "block": str, "gdc_disease_type": str})

        # Code index: subject_id -> row   (e.g. "ICDO3M:8960/3", "ONCOTREE:WT", "ICDO3T:C64.9")
        self._by_code = {r.subject_id: r for r in self.terms.itertuples(index=False)}
        # Label index: normalised label -> row (TEXT rows only)
        txt = self.terms[self.terms.subject_source == "TEXT"]
        self._by_label = {normalize_label(r.subject_label): r for r in txt.itertuples(index=False)}

        self._gdc_index, self._gdc_enums = self._build_gdc_index(Path(schemas_dir))

    # ---------- GDC dictionary: NCIt code -> permissible value ----------
    @staticmethod
    def _build_gdc_index(schemas_dir: Path):
        """
        {(node, property): {"NCIT:C12415": ["Kidney"], ...}} for every enum value whose
        _terms_enum.yaml entry carries an NCIt termDef. This is what removes hand-typed
        value tables: the dictionary already says which concept each value denotes.
        """
        terms = yaml.safe_load(open(schemas_dir / "_terms_enum.yaml")) or {}

        def ncit_for(value: str, node: str, prop: str) -> Optional[str]:
            entry = terms.get(gdc_term_key(value))
            if not isinstance(entry, dict):
                return None
            td = ((entry.get(node) or {}).get(prop) or {}).get("termDef") \
                or (entry.get("common") or {}).get("termDef") or {}
            if td.get("source") == "NCIt" and td.get("term_id"):
                return "NCIT:" + str(td["term_id"]).strip()
            return None

        index: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        enums: Dict[Tuple[str, str], List[str]] = {}
        for node in ("case", "diagnosis", "demographic", "sample"):
            f = schemas_dir / f"{node}.yaml"
            if not f.exists():
                continue
            props = (yaml.safe_load(open(f)) or {}).get("properties", {}) or {}
            for prop, spec in props.items():
                enum = spec.get("enum") if isinstance(spec, dict) else None
                if not enum:
                    continue
                enums[(node, prop)] = list(enum)
                bucket: Dict[str, List[str]] = {}
                for v in enum:
                    code = ncit_for(v, node, prop)
                    if code:
                        bucket.setdefault(code, []).append(v)
                index[(node, prop)] = bucket
        return index, enums

    def gdc_enum(self, node: str, prop: str) -> List[str]:
        """All permissible values of a GDC property (coded or not)."""
        return self._gdc_enums.get((node, prop), [])

    def gdc_values(self, node: str, prop: str) -> Dict[str, List[str]]:
        return self._gdc_index.get((node, prop), {})

    # ---------- value-level resolution ----------
    def resolve(self, system: str, raw: object) -> Resolution:
        """Resolve one source value (with its coding system) to an NCIt concept."""
        if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() == "":
            return _UNRESOLVED
        row = None
        if system == "ICDO3M":
            code = parse_icdo3_morphology(raw)
            row = self._by_code.get(f"ICDO3M:{code}") if code else None
        elif system == "ICDO3T":
            code = parse_icdo3_topography(raw)
            row = self._by_code.get(f"ICDO3T:{code}") if code else None
        elif system == "ONCOTREE":
            row = self._by_code.get(f"ONCOTREE:{str(raw).strip().upper()}")
        elif system == "TEXT":
            row = self._by_label.get(normalize_label(raw))
            if row is None:                         # "8960/3 : Nephroblastoma, NOS" style labels
                code = parse_icdo3_morphology(raw)
                row = self._by_code.get(f"ICDO3M:{code}") if code else None
        if row is None:
            return Resolution(None, None, None, None, None, source_value=str(raw))
        return Resolution(
            concept_id=row.object_id, concept_label=row.object_label,
            predicate=row.predicate_id, justification=row.mapping_justification,
            confidence=float(row.confidence) if row.confidence else None,
            source_value=str(raw), matched_on=row.subject_id,
        )

    # ---------- concept -> other terminologies ----------
    def xref(self, concept_id: str, prefix: str) -> Optional[Tuple[str, str, str]]:
        """First cross-reference of a concept into a terminology: (code, label, predicate)."""
        hit = self.xrefs[(self.xrefs.subject_id == concept_id)
                         & self.xrefs.object_id.str.startswith(prefix + ":")]
        if hit.empty:
            return None
        r = hit.iloc[0]
        return r.object_id.split(":", 1)[1], r.object_label, r.predicate_id

    def related(self, concept_id: str, relation: str) -> Optional[Tuple[str, str]]:
        hit = self.relations[(self.relations.subject_id == concept_id)
                             & (self.relations.relation == relation)]
        if hit.empty:
            return None
        return hit.iloc[0].object_id, hit.iloc[0].object_label

    def disease_type_for_morphology(self, morph: Optional[str]) -> Optional[str]:
        """ICD-O-3 '8960/3' -> GDC disease_type via the ICD-O-3 block table."""
        if not morph:
            return None
        n = int(morph[:4])
        hit = self.blocks[(self.blocks.start <= n) & (self.blocks.end >= n)]
        return None if hit.empty else hit.iloc[0].gdc_disease_type

    def gdc_value(self, concept_id: Optional[str], node: str, prop: str) -> Optional[str]:
        vals = self.gdc_values(node, prop).get(concept_id or "", [])
        return sorted(vals)[0] if vals else None

    # ---------- element-level: per-subject annotation ----------
    def _resolve_element(self, element: str, tables: Dict[str, pd.DataFrame],
                         subject_ids: pd.Index, earlier: Dict[str, Dict[str, Resolution]]
                         ) -> Dict[str, Resolution]:
        """Walk the element's source columns in priority order; first hit per subject wins."""
        out: Dict[str, Resolution] = {}
        for src in self.elements[element]["sources"]:
            if "derive_from" in src:                          # inference fallback
                for sid in subject_ids:
                    if sid in out:
                        continue
                    base = earlier.get(src["derive_from"], {}).get(sid)
                    if base is not None and base.ok:
                        rel = self.related(base.concept_id, src["relation"])
                        if rel:
                            out[sid] = Resolution(rel[0], rel[1], "inferred", "semapv:LogicalReasoning",
                                                  None, source_table=src["derive_from"],
                                                  source_column=src["relation"],
                                                  source_value=base.concept_id, matched_on=base.concept_id)
                continue
            df = tables.get(src["table"])
            if df is None or src["column"] not in df.columns:
                continue
            per_subject = (df[["__subject_id", src["column"]]].dropna()
                           .groupby("__subject_id")[src["column"]].agg(lambda s: s.mode().iat[0]))
            for sid, raw in per_subject.items():
                if sid in out or sid not in subject_ids:
                    continue
                res = self.resolve(src["system"], raw)
                if res.ok:
                    res.source_table, res.source_column = src["table"], src["column"]
                    out[sid] = res
        return out

    def annotate_cases(self, patients: pd.DataFrame, samples: pd.DataFrame,
                       patient_id_col: str = "patientId") -> pd.DataFrame:
        """
        One row per subject with: resolved disease and site concepts, the GDC values derived
        from them, cross-references, and full provenance. Unresolved subjects keep NaN so the
        caller can fall back to its previous logic and log the gap.
        """
        p = patients.copy()
        p["__subject_id"] = p[patient_id_col].astype(str)
        s = samples.copy()
        if "subject_id" in s.columns:
            s["__subject_id"] = s["subject_id"].astype(str)
        else:
            s["__subject_id"] = s["sampleId"].astype(str).str.replace(r"-\d+$", "", regex=True)
        tables = {"patients": p, "samples": s}
        subjects = pd.Index(p["__subject_id"].unique())

        resolved: Dict[str, Dict[str, Resolution]] = {}
        resolved["disease"] = self._resolve_element("disease", tables, subjects, resolved)
        resolved["primary_site"] = self._resolve_element("primary_site", tables, subjects, resolved)

        rows = []
        for sid in subjects:
            d = resolved["disease"].get(sid, _UNRESOLVED)
            t = resolved["primary_site"].get(sid, _UNRESOLVED)
            # morphology: from the source code if that is what matched, else from the concept's xref
            morph = d.matched_on.split(":", 1)[1] if (d.matched_on or "").startswith("ICDO3M:") else None
            if morph is None and d.ok:
                x = self.xref(d.concept_id, "ICDO3M")
                morph = x[0] if x else None
            topo = t.matched_on.split(":", 1)[1] if (t.matched_on or "").startswith("ICDO3T:") else None
            snomed = self.xref(d.concept_id, "SNOMEDCT") if d.ok else None
            icd10 = self.xref(d.concept_id, "ICD10CM") if d.ok else None
            rows.append({
                "subject_id": sid,
                # disease
                "disease_concept_id": d.concept_id,
                "disease_concept_label": d.concept_label,
                "disease_predicate": d.predicate,
                "disease_justification": d.justification,
                "disease_source": f"{d.source_table}.{d.source_column}" if d.ok else None,
                "disease_source_value": d.source_value if d.ok else None,
                "icdo3_morphology": morph,
                "snomedct_code": snomed[0] if snomed else None,
                "icd10cm_code": icd10[0] if icd10 else None,
                "icd10cm_predicate": icd10[2] if icd10 else None,
                # site
                "site_concept_id": t.concept_id,
                "site_concept_label": t.concept_label,
                "site_predicate": t.predicate,
                "site_source": f"{t.source_table}.{t.source_column}" if t.ok else None,
                "icdo3_topography": topo,
                "omop_icdo3_code": f"{morph}-{topo}" if morph and topo else None,
                # GDC values derived from concepts
                "gdc_disease_type": self.disease_type_for_morphology(morph),
                "gdc_primary_diagnosis": self.gdc_value(d.concept_id, "diagnosis", "primary_diagnosis"),
                "gdc_morphology": morph if morph in self._morph_enum() else None,
                "gdc_primary_site": self.gdc_value(t.concept_id, "case", "primary_site"),
                "gdc_tissue_or_organ_of_origin": self.gdc_value(t.concept_id, "diagnosis",
                                                                "tissue_or_organ_of_origin"),
            })
        return pd.DataFrame(rows)

    def _morph_enum(self) -> set:
        if not hasattr(self, "_morph_cache"):
            vals = self.gdc_values("diagnosis", "morphology")
            self._morph_cache = {v for vs in vals.values() for v in vs}
        return self._morph_cache

    def unmapped_report(self, patients: pd.DataFrame, samples: pd.DataFrame,
                        patient_id_col: str = "patientId") -> pd.DataFrame:
        """Distinct source values that did not resolve, per element/column, for curator review."""
        tables = {"patients": patients, "samples": samples}
        out = []
        for element, spec in self.elements.items():
            if spec.get("status", "active") != "active":
                continue
            for src in spec.get("sources", []):
                df = tables.get(src.get("table"))
                if df is None or src.get("column") not in df.columns:
                    continue
                for raw, n in df[src["column"]].dropna().astype(str).value_counts().items():
                    if not self.resolve(src["system"], raw).ok:
                        out.append({"element": element, "table": src["table"], "column": src["column"],
                                    "system": src["system"], "source_value": raw, "n_rows": int(n)})
        return pd.DataFrame(out, columns=["element", "table", "column", "system", "source_value", "n_rows"])


# ------------------------------------------------------------------------------
# SNOMED CT -> ICD-10-CM (NLM rule-based map)
# ------------------------------------------------------------------------------

_AGE_RULE = re.compile(r"IFA\s+445518008\s*\|[^|]*\|\s*(<=|>=|<|>|=)\s*([\d.]+)\s*(years?|months?|days?)?",
                       re.IGNORECASE)
_CONCEPT_RULE = re.compile(r"IFA\s+(\d+)")


def snomed_to_icd10cm(map_rows: pd.DataFrame, sctid: str, context_sctids: Iterable[str] = (),
                      age_years: Optional[float] = None) -> List[dict]:
    """
    Evaluate the NLM SNOMED CT -> ICD-10-CM map for one concept and one patient context.

    map_rows       : the NLM ExtendedMap file (tab-separated) as a DataFrame, with columns
                     referencedComponentId, mapGroup, mapPriority, mapRule, mapAdvice, mapTarget.
    context_sctids : SNOMED ids present for the patient, e.g. {"248152002"} for Female (finding).
    age_years      : patient age for age rules (IFA 445518008 | Age at onset ... | < 17.0 years).

    Returns one dict per map group: the first row (by mapPriority) whose rule holds.
    """
    ctx = {str(c) for c in context_sctids}
    rows = map_rows[map_rows["referencedComponentId"].astype(str) == str(sctid)]
    ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
           ">=": lambda a, b: a >= b, "=": lambda a, b: a == b}
    results = []
    for group, grp in rows.groupby("mapGroup"):
        for r in grp.sort_values("mapPriority").itertuples(index=False):
            rule = str(r.mapRule).strip()
            if rule in ("TRUE", "OTHERWISE TRUE"):
                hit = True
            elif (m := _AGE_RULE.search(rule)):
                unit = (m.group(3) or "years").lower()
                limit = float(m.group(2)) / (12.0 if unit.startswith("month") else
                                             365.25 if unit.startswith("day") else 1.0)
                hit = age_years is not None and ops[m.group(1)](age_years, limit)
            elif (m := _CONCEPT_RULE.search(rule)):
                hit = m.group(1) in ctx
            else:
                hit = False
            if hit:
                target = r.mapTarget if isinstance(r.mapTarget, str) and r.mapTarget.strip() else None
                results.append({"mapGroup": int(group), "icd10cm": target,
                                "rule": rule, "advice": r.mapAdvice})
                break
    return results
