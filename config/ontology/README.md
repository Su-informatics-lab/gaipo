# GAIPO ontology layer

Concept-based metadata mapping for the `data_model` stage. Source values resolve to an
**NCIt concept** first; GDC values and cross-references are derived from the concept.

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

## Adding a cancer type

1. Add its source labels/codes to `source_terms.sssom.tsv` (object = NCIt code; look it up in
   `_terms_enum.yaml` or the NCI EVS browser).
2. Add its ICD-O-3 morphology (and SNOMED CT / ICD-10-CM if needed) to `concept_xrefs.sssom.tsv`.
3. Add its ICD-O-3 block to `icdo3_morphology_blocks.tsv` if not present.
4. Run the pipeline and review `unmapped_terms.tsv`.
