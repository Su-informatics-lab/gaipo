# GAIPO

**Graph Artificial Intelligence for Pediatric Oncology**

The Childhood Cancer Data Initiative (CCDI) ecosystem provides essential clinicogenomic data for deep learning in pediatric cancer research. To enable the efficient use of the CCDI resource for AI model training and implementation, we developed a generic graph AI platform, Graph Artificial Intelligence for Pediatric Oncology (GAIPO), as well as the standards of data, model, and pipeline, to streamline the use of CCDI clinicogenomics data of various data modalities from bulk and single-cell omics data to clinical information in AI development. GAIPO provides a comprehensive workflow: 

(1) Data fetching through CCDI Data Federation Resource API and cBioPortal API, and ScPCA for clinical metadata, multi-omics, and spatial transcriptomics data; 

(2) Data modeling through mapping and harmonizing the fetched clinical and genomics data according to the CCDI data model; 

(3) Graph construction functionalities through GraphML from NetworkX; 

(4) Implementation of previously published graph AI models and development of novel models; 

(5) Post-analysis, such as cancer type classification, patient stratification, clustering, survival analysis, and feature selection. 


![GAIPO overview](image/GAIPO.png)

## Overview

Advances in artificial intelligence (AI) are shifting the paradigm in precision medicine for pediatric cancer, including biomarker identification, drug discovery, and survival analysis. Studies of advanced AI models for Pediatric oncology research often combine heterogeneous clinical and omics data collected across multiple programs and platforms. GAIPO organizes this process into reproducible modules that can be run independently or as an end-to-end pipeline.

The framework is designed to support:

- Cohort and identifier discovery from CCDI Federation sources.
- Extraction of clinical and multi-omics data.
- Harmonization into CCDI/GDC-supported, analysis-ready data models.
- Modality-specific quality control, feature selection, and scaling.
- Patient-similarity graph construction (for clincogenomics data).
- Graph AI models for tumor classification and survival analysis.
- Model interpretation, biomarker prioritization, and post-model survival analysis.
- Reproducible execution through Docker Compose.

> [!IMPORTANT]
> GAIPO is research software. It is not intended for clinical diagnosis, treatment selection, or other direct clinical use.

## Workflow

![GAIPO workflow](image/GAIPO_WORKFLOW.png)

The pipeline contains seven ordered stages:

| Stage | CLI name | Purpose | Representative outputs |
| --- | --- | --- | --- |
| 1 | `data_fetch` | Discover and retrieve cohort-level subject and sample identifiers. | Subject IDs, sample IDs, cohort manifests |
| 2 | `data_extract` | Extract clinical and omics data from configured sources. | Raw or source-aligned TSV, Zarr, and Parquet files |
| 3 | `data_model` | Harmonize extracted records into a GDC-shaped data model. | Case, sample, demographic, ID-map, and file-node Parquets |
| 4 | `data_process` | Perform quality control, feature filtering, normalization, and dimensionality reduction. | Analysis-ready feature matrices and preprocessing metadata |
| 5 | `graph_construct` | Build patient-similarity graphs from processed features. | GraphML files, adjacency data, node features, and graph metadata |
| 6 | `graph_ai_model` | Train and evaluate graph neural networks for prediction and survival tasks. | Model checkpoints, predictions, risk scores, and latent representations |
| 7 | `post_analysis` | Interpret trained models and perform downstream statistical analyses. | Feature rankings, Kaplan–Meier analyses, log-rank results, and figures |

Each stage consumes the outputs of the preceding stage. Intermediate artifacts are retained so that a stage can be rerun without repeating the entire workflow.

## Data and Modeling Components

### Data sources

GAIPO is designed to work with pediatric cancer data available through the CCDI ecosystem and related genomic portals. Depending on the study configuration and access permissions, sources may include:

- PCDC
- Treehouse
- St. Jude Cloud
- Kids First
- CCDI ecDNA
- cBioPortal or PedcBioPortal-compatible studies
- Single-cell Pediatric Cancer Atlas Portal (ScPCA)

Availability varies by cohort. Users are responsible for complying with the data-use agreements, authentication requirements, and access controls of each source.

### Supported data types

The framework can integrate clinical variables with one or more genomics modalities, including:

- mRNA expression
- Copy-number alteration data
- DNA methylation
- miRNA expression

The exact modalities used in a run depend on the cohort configuration and data availability.

### Graph construction

For clincogenomics data, patients are represented as graph nodes, with node features derived from processed clinical or omics measurements. Edges encode patient similarity, such as cosine similarity under a configurable radius or neighborhood rule. To prevent information leakage, graph-construction parameters estimated from the training set should be reused for validation and test data.

### Graph AI and interpretation

GAIPO supports graph neural network architectures such as GCN, GAT, and GIN. The modeling layer can be configured for:

- Tumor or histological subtype classification.
- Time-to-event modeling using a Cox partial-likelihood objective.
- Multi-task learning across classification and survival endpoints.
- Modality-specific graph encoders and cross-attention fusion.

Post-analysis can include feature-attribution summaries, biomarker ranking, latent-space visualization, hierarchical clustering, Kaplan–Meier curves, and log-rank tests.

## Getting Started

### Prerequisites

Choose one of the following execution options:

1. **Docker Compose** — recommended for a reproducible environment.
2. **Local Python environment** — useful for development and debugging.

For either option, clone the repository and enter its root directory:

```bash
git clone https://github.com/Su-informatics-lab/gaipo.git
cd GAIPO
```

Before running the pipeline, set the cancer types, search terms, studies, modalities and labels in `config/pipeline_config.yaml`; [Configuring a run](#configuring-a-run-configpipeline_configyaml) walks through every block with Wilms tumor and glioma as the example.

Do not commit access tokens, passwords, controlled-data identifiers, or other secrets to the repository.

### Local Python environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root so that Python can resolve the `src` package and the pipeline can locate its configuration and output directories.

## Configuring a run (`config/pipeline_config.yaml`)

Every step reads one file, `config/pipeline_config.yaml` (or the file named by the `CONFIG_PATH` environment variable). Edit it in place before a run, and keep a copy of the version used for each set of published results. The blocks are keyed by **cancer type**: the same name (e.g. `"Wilms Tumor"`, `"Glioma"`) must be used in every block, and output folders use its lower-case form with underscores (`wilms_tumor`, `glioma`).

| Block | Read by | What it sets |
| --- | --- | --- |
| `cancer_type_interest` | every step | The cancer types to run. Types not listed are skipped. |
| `default_type_terms`, `custom_type_terms` | `data_fetch` | Search strings sent to the CCDI Data Federation per cancer type (case-insensitive substring match on the diagnosis). Put new cancer types in `custom_type_terms`. |
| `search` | `data_fetch` | `use_ontology: false` (default) searches with the lists above; `true` builds the search from the NCIt concept set named in `search.cohorts` (see `config/ontology/`). |
| `cBio_studyId`, `pedcBio_studyId` | `data_extract` onward | The cBioPortal / PedcBioPortal studies per cancer type (one ID or a list). Only CCDI subjects that are patients of these studies are extracted. |
| `modeling_studyId` | `data_process` onward | Optional subset of those studies to model per cancer type (e.g. Glioma: pbta_all). Unset: all. |
| `expression_profiles` | `data_extract`, `data_process` | The omics per cancer type: `mrna`, `mirna`, `methylation`, `linear_cna`. |
| `processor.labels` | `data_process`, `benchmark` | Per cancer type: the label column, where it is (`samples` or `patients`), the map from its values to 0/1, and the feature-selection settings. |
| `patient_similarity_graph`, `model_mogonet`, `post_analysis`, `benchmark` | later steps | Graph, model, analysis and cross-validation settings. |

### Example: Wilms tumor and glioma

**1. Cancer types.** List both:

```yaml
cancer_type_interest:
  - "Wilms Tumor"
  - "Glioma"
```

**2. Search terms for `data_fetch`.** The fixed lists for both types are already defined; keep the ontology search off to use them:

```yaml
default_type_terms:
  Wilms Tumor: ["wilms", "nephroblastoma", "8960/3 : Nephroblastoma, NOS"]
  Glioma: ["glioma", "high-grade glioma", "low-grade glioma"]
search:
  use_ontology: false
```

To search with the ontology layer instead, set `use_ontology: true` (both types already have a cohort under `search.cohorts`). Also set `compare_with_config: false` unless you want the ontology-vs-fixed-list report: with `true`, both sets of queries are sent. For a cancer type not listed here, add its terms under `custom_type_terms`, for example `Neuroblastoma: ["neuroblastoma"]`.

**3. Studies for `data_extract`.** Wilms tumor comes from cBioPortal; glioma from PedcBioPortal (PBTA, used for modeling) and cBioPortal (CPTAC):

```yaml
cBio_studyId:
  "Wilms Tumor": "wt_target_2018_pub"
  "Glioma": "brain_cptac_2020"
pedcBio_studyId:
  "Glioma": "pbta_all"
```

To fetch and extract both glioma studies but model only one study per cancer type (wt_target_2018_pub
for Wilms tumor, pbta_all for glioma), add `modeling_studyId`. It is read by `data_process`,
`data_model`, `graph_construct`, `graph_ai_model`, `post_analysis` and `benchmark`; `data_fetch` and
`data_extract` still use every study above, so brain_cptac_2020 stays available to
`scripts/ontology_report.py` and `scripts/label_concordance.py --compare`:

```yaml
modeling_studyId:
  "Wilms Tumor": "wt_target_2018_pub"
  "Glioma": "pbta_all"          # a study ID, "portal:study_id", or a list
```

Without `modeling_studyId` (or for a cancer type not listed in it), every study is modeled.

By default `data_extract` keeps the samples that CCDI lists for the matched patients and that the
portal also has. CCDI can list only some of a patient's samples (in TARGET Wilms tumor, 126 of 657),
which drops omics samples. To take every portal sample of the matched patients instead:

```yaml
extract:
  samples:
    "Wilms Tumor": patients   # default for every type: ccdi; a single value applies to all types
```

PedcBioPortal needs a data-access token. Set it in the shell (or in a git-ignored `.env` file next to `docker-compose.yml`) and leave `auth:` in the YAML as `null`:

```bash
export PEDCBIOPORTAL_API_TOKEN="<your token>"
```

**4. Omics.**

```yaml
expression_profiles:
  "Wilms Tumor": [mrna, mirna, methylation]
  "Glioma": [mrna, linear_cna]
```

A study can have several files per modality (wt_target_2018_pub has microarray and RNA-seq mRNA).
`data_process` takes the first match (for mRNA, RNA-seq z-scores first) and prints the file it used.
To choose one, name it (or a glob) under the cancer type's `processor.labels` block:

```yaml
processor:
  labels:
    Wilms Tumor:
      omics_files: {mrna: "wt_target_2018_pub_mrna.tsv"}   # microarray
```

**5. Labels for `data_process`.** The map must cover every value of the label column in the extracted study (the processor stops otherwise), so look the values up after `data_extract`:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/fetch/cbioportal_api_request/pedcbio/glioma/pbta_all/clinical_samples.csv')['CANCER_TYPE'].value_counts())"
```

```yaml
processor:
  labels:
    Wilms Tumor:
      source: samples
      column: HISTOLOGY_CLASSIFICATION_IN_PRIMARY_TUMOR
      map: "FHWT=0,DAWT=1"
    Glioma:
      source: samples
      column: CANCER_TYPE
      map:                     # mapping form: needed when a value contains a comma
        "Low-grade glioma/astrocytoma (WHO grade I/II)": 0
        "High-grade glioma/astrocytoma (WHO grade III/IV)": 1
      drop_unmapped: true      # leave out samples whose value is not in the map
```

`map` is either a string (`"FHWT=0,DAWT=1"`) or a YAML mapping as above; use the mapping when a value
contains a comma (e.g. "Low-Grade Glioma, NOS or NEC"). Several values can map to the same class.
Optional `class_names` (e.g. `class_names: {0: Low-grade, 1: High-grade}`) names the classes in the
cohort table and the KM plots; without it, names come from the map (the single value of a class, or
Low-grade / High-grade when every value of the class says so).

The labels apply to every study of a cancer type. If two studies of one type use different values (as PBTA and CPTAC do), process one study at a time by commenting out the other study ID in step 3. To model only the classes in the map (e.g. low- vs high-grade glioma when the extract also holds other tumor types or blank values), add `drop_unmapped: true` under the cancer type: samples whose value is missing or not in the map are then left out, and the processor prints how many per value.

**6. Run and check.**

```bash
python -m src.main --call data_fetch      # both cancer types; about 2 h for these two with the fixed lists
python -m src.main --call data_extract
python scripts/fetch_counts.py            # counts for the run (see below)
```

`data_fetch` writes `data/fetch/ccdi_api_request/subject_ids/<type>.csv` and `sample_ids/<type>.csv` (one row per CCDI source and ID), `data/fetch/ccdi_api_request/search_skipped.tsv` (sources that returned no records, e.g. PCDC and ccdi-ecDNA), and `data/gdm/indexes/idmap.parquet`. `data_extract` writes `data/fetch/cbioportal_api_request/<portal>/<type>/<study_id>/` with `clinical_patients.csv`, `clinical_samples.csv` and one TSV per omics profile. Both steps overwrite earlier outputs, so copy `data/fetch/` before rerunning if you need the previous files.

`scripts/fetch_counts.py` reports the subject and sample IDs found per cancer type and CCDI source (and unique across sources), and the patients, samples and omics samples extracted per study. With `--compare <folder with an earlier ccdi_api_request/ and cbioportal_api_request/>` it also lists the IDs added or removed since that run.

## Running the Pipeline

### Run the complete pipeline

```bash
python -m src.main --all
```

This executes all stages in dependency order:

```text
data_fetch → data_extract → data_process → data_model → graph_construct → graph_ai_model → post_analysis
```

### Run one stage

```bash
# 1. Fetch subject and sample identifiers
python -m src.main --call data_fetch

# 2. Extract clinical and omics data
python -m src.main --call data_extract

# 3. Build GDC-shaped data-model tables
python -m src.main --call data_model

# 4. Process data and select features
python -m src.main --call data_process

# 5. Construct patient-similarity graphs
python -m src.main --call graph_construct

# 6. Train and evaluate Graph AI models
python -m src.main --call graph_ai_model

# 7. Run model interpretation and downstream analyses
python -m src.main --call post_analysis
```

When running a single downstream stage, confirm that all required upstream outputs already exist.

### Run selected stages

Pass stage names as a comma-separated list, with no spaces:

```bash
python -m src.main --call data_fetch,data_extract
```

For example, to rebuild the graph and rerun the model and post-analysis:

```bash
python -m src.main --call graph_construct,graph_ai_model,post_analysis
```

### Run through a specified stage

Use `--until` to start at the beginning of the pipeline and stop after the named stage:

```bash
python -m src.main --until graph_construct
```

### Cross-validation benchmark

`benchmark` is an on-demand stage (not part of `--all`). It compares MOGONET with Gaussian Naive Bayes and Random Forest by stratified k-fold cross-validation with one seed, and reports the mean and standard deviation of accuracy, macro F1, and AUC across the test folds.

```bash
python -m src.main --call benchmark
```

Settings are in the `benchmark` block of `config/pipeline_config.yaml` (`n_splits`, `seed`, `sd_ddof`, `models`, `feature_selection`); MOGONET uses the `model_mogonet` hyperparameters. With `feature_selection: in_fold` (default), the processor's feature selection and min-max scaling are refitted on each training fold, so test folds are not used before prediction; this needs the `data_extract` outputs. `feature_selection: precomputed` reuses the `data_process` matrices, which were selected on all patients. Results are written to `data/analysis/benchmark/` (`cv_folds.csv` per fold, `cv_splits.csv` with the fold assignment of every patient, and `cv_summary.csv` / `cv_summary.md` with mean ± SD).

### Cohort table (Table 2)

After `data_process`, `scripts/cohort_table.py` tabulates the modeled patients (`labels_train.csv` +
`labels_test.csv` of each modeled study) by class, with their clinical values from the extract
(`clinical_patients.csv`, then `clinical_samples.csv`): age at diagnosis (No. (%) and median (IQR),
overall and in 0-5, 6-11 and 12+ years), sex, ethnicity, race, OS status and follow-up, EFS status
and event-free follow-up, and clinical stage.

```bash
python scripts/cohort_table.py                 # every cancer type in cancer_type_interest
```

It writes `data/analysis/cohort_table/cohort_table.tsv` and `.md` (one column per class, then All),
`cohort_values.tsv` (every source value and the row it was counted in) and `cohort_patients.tsv` (one
row per patient), and prints the column used for each row. The first column found is used (e.g. age:
`AGE`, then `AGE_AT_DIAGNOSIS`, `AGE_IN_DAYS`, `AGE_AT_DIAGNOSIS_DAYS`; EFS: `EFS_STATUS`, then
`EVENT_TYPE`, where a blank TARGET event type is no event). To use another column or unit:

```bash
python scripts/cohort_table.py --column "Glioma:age=AGE_AT_DIAGNOSIS_DAYS:days"
```

## Docker Compose

Docker Compose provides the most reproducible way to build and run GAIPO.

### Build the image

```bash
docker compose build
```

Recent Docker Compose versions can optionally delegate builds to Bake:

```bash
export COMPOSE_BAKE=true
docker compose build
```

### Run the complete pipeline

```bash
docker compose run --rm app python -m src.main --all
```

### Run selected stages

```bash
docker compose run --rm app \
  python -m src.main --call data_fetch,data_extract
```

### Run through graph construction

```bash
docker compose run --rm app \
  python -m src.main --until graph_construct
```

The Compose configuration should mount persistent data, output, and model directories so that artifacts remain available after the temporary `app` container exits.

## Outputs

Output paths depend on the project configuration, but a complete run typically produces:

- Cohort manifests and subject/sample mappings.
- Raw and harmonized clinical tables.
- Processed modality-specific feature matrices.
- Saved preprocessing parameters for validation and test data.
- Patient graphs and graph-construction metadata.
- Trained model checkpoints and evaluation metrics.
- Class predictions, survival-risk scores, and learned embeddings.
- Feature-attribution and biomarker-ranking tables.
- Kaplan–Meier plots, log-rank test results, and other publication-ready figures.

Large datasets, credentials, model checkpoints, and generated results should not be committed to Git unless intentionally managed through an appropriate large-file or artifact-storage system.

## Reproducibility and Leakage Prevention

For valid evaluation, GAIPO should fit all data-dependent transformations using the training data only. The fitted parameters are then applied unchanged to validation and test sets. This includes:

- Missingness and zero-expression filters.
- Variance and statistical feature-selection thresholds.
- Scaling and dimensionality-reduction transformations.
- Graph similarity thresholds or radius parameters.
- Risk-group thresholds used for survival comparisons.

Record the configuration, software versions, random seeds, cohort manifest, and source-data release used for each experiment.

## Troubleshooting

- **A downstream stage cannot find its inputs:** run the required upstream stages or verify the configured artifact paths.
- **The `src` module cannot be imported:** run the command from the repository root and confirm that the environment is activated.
- **A container cannot read or write data:** check the host-directory mounts and permissions in `docker-compose.yml`.
- **Data extraction fails:** verify network access, source endpoints, credentials, and cohort availability.
- **Results differ across runs:** fix random seeds and confirm that the same cohort manifest, configuration, split, and dependency versions were used.
- **Validation performance is unexpectedly high:** check that preprocessing, graph construction, and risk-group thresholds were learned from the training set only.

## Contributing

Issues and pull requests are welcome. For substantial changes, open an issue first to describe the proposed feature, affected pipeline stages, and validation plan. New contributions should preserve modular execution, configuration-driven behavior, and reproducible outputs.

## Citation

If you use GAIPO in a publication, please cite the associated manuscript and software release. Formal citation information will be added when available.

Based on GAIPO, we developed an end-to-end multimodal framework, PCGS (i.e., biomarker and risk group identification for Pediatric Cancers via GNNs with Shapley-value-based explainability) for pediatric cancer by incorporating omics-specific representation learning via GNN models with cross-attention fusion and multi-head task losses in model training for downstream tasks such as classification, clustering, and survival analysis. We applied Shapley value-based feature attribution to identify key biomarkers for patient stratification, with sensitivity to baseline background selection. Please cite:

Shi, Zanyu, Aishwarya Budhkar, Waqas Amin, Karen E. Pollok, Jing Su, and Kun Huang. "PCGS: biomarker and risk group identification for Pediatric Cancers via explainable Graph neural networks with Shapley values." medRxiv (2026): 2026-08.


## License

See the repository's `LICENSE` file for licensing terms.

## Contact

For questions, bug reports, or feature requests, please open a GitHub issue.
