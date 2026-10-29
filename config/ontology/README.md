# GAIPO ontology layer

Concept-based metadata mapping, used at two stages:

- **`data_fetch` (search), optional:** with `search.use_ontology: true` in
  `pipeline_config.yaml`, the CCDI search terms of a cohort are generated from its NCIt
  concepts (`analysis_groups.tsv`) and every label and ICD-O-3 code mapped to them below, and a
  returned record joins the cohort when its diagnosis resolves to one of those concepts
  (`src/ontology_search.py`). Off by default (fixed term lists), because the broader search
  takes much longer.
- **`data_model` (standardization):** source values resolve to an **NCIt concept** first; GDC
  values and cross-references are derived from the concept.

```
source value ──SSSOM──▶ NCIt concept ──GDC termDef / ICD-O-3 block──▶ GDC permissible value
                             └──▶ ICD-O-3 · SNOMED CT · ICD-10-CM · OMOP ICDO3 code
```

| File | Level | What to edit |
| --- | --- | --- |
| `element_map.yaml` | element | Which source columns feed each standard data element, in priority order, and which GDC properties it fills |
| `source_terms.sssom.tsv` | value | Source codes/labels → NCIt (add a row per new synonym or code) |
| `concept_xrefs.sssom.tsv` | value | NCIt → ICD-O-3 morphology, SNOMED CT, ICD-10-CM |
| `concept_relations.tsv` | concept | Disease → typical primary site (used only when no topography code) |
| `icdo3_morphology_blocks.tsv` | rule | ICD-O-3 morphology block → GDC `case.disease_type` |
| `analysis_groups.tsv` | cohort | Cohort → set of NCIt concepts (search terms for `data_fetch`; counts in `scripts/ontology_report.py`) |

NCIt → GDC value is **not** in these files: it is read from the NCIt `termDef` that the bundled
GDC dictionary (`src/gdcdictionary/schemas/_terms_enum.yaml`) attaches to each enum value.

Mapping files follow [SSSOM](https://mapping-commons.github.io/sssom/): every row states its
predicate (`skos:exactMatch`, `skos:broadMatch`, …), justification and confidence.

## Use

- Switch on/off in `config/gdc_mapping.yaml` → `ontology.enabled`. Off = legacy behaviour.
- Outputs: `data/gdm/indexes/concept_annotations.parquet` (codes + provenance per subject) and
  `data/metadata/unmapped_terms.tsv` (values to curate). Unresolved values fall back to the
  legacy maps, or `Unknown` if the legacy value is not a valid GDC enum.
- Worked example: `python -m examples.ontology.run_example`; tests: `pytest tests/test_ontology.py`.
- Search (with `use_ontology: true`): `data_fetch` writes
  `data/fetch/ccdi_api_request/search_terms.tsv` (each term, its origin row and NCIt concept). `python scripts/search_report.py "Wilms Tumor" "Glioma"` compares
  the ontology search with the fixed term lists on the same CCDI records
  (`data/metadata/search_report/`) without changing the cohort files; tests:
  `pytest tests/test_ontology_search.py`.

## Adding a cancer type

1. Add its source labels/codes to `source_terms.sssom.tsv` (object = NCIt code; look it up in
   `_terms_enum.yaml` or the NCI EVS browser).
2. Add its ICD-O-3 morphology (and SNOMED CT / ICD-10-CM if needed) to `concept_xrefs.sssom.tsv`.
3. Add its ICD-O-3 block to `icdo3_morphology_blocks.tsv` if not present.
4. Add its cohort (one row per NCIt concept) to `analysis_groups.tsv` and map the cancer type
   to it under `search.cohorts` in `pipeline_config.yaml`.
5. Run the pipeline and review `unmapped_terms.tsv` and `search_values.tsv` (`term_only` rows).
