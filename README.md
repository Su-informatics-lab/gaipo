# GAIPO
* GAIPO: Graph Artificial Intelligence for Pediatric Oncology

## Run each step
### Step 1: writes subject_ids / sample_ids
python -m src.main --call data_fetch

### Step 2: Pull clinical + omics → TSV/Zarr/Parquet
python -m src.main --call data_extract

### Step 3: Map to GDC-shaped Parquets (case, sample, demographic, idmap, file nodes)   
python -m src.main --call data_model

### Step 4: Construct graph
python -m src.main --call graph_construct

### Step 5: Use Graph AI
python -m src.main --call graph_ai_model

### Step 6: Feature selection & survival analysis
python -m src.main --call post_analysis


## Run multiple steps
python -m src.main --call {data_fetch,data_extract,process,graph_construct,graph_ai_model,post_analysis}
* Run the default full pipeline:
python -m src.main --all
* Run up to a step
python -m src.main --until graph_construct
* Run multiple steps (e.g., first two setps)
python -m src.main --call data_fetch,data_extract


# Docker Compose
optional: faster builds
export COMPOSE_BAKE=true

## build and run
export COMPOSE_BAKE=true

docker compose build
docker compose run --rm app python -m src.main --all
docker compose run --rm app python -m src.main --call data_fetch,data_extract
