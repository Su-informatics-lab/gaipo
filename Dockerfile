# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# project files
# includes pipeline_config.yaml AND gdc_mapping.yaml
COPY config/ /app/config/          
COPY src/ /app/src/
COPY data/ /app/data/

# ensure src is a package
RUN python - <<'PY'\nimport pathlib; p=pathlib.Path('/app/src/__init__.py'); p.parent.mkdir(parents=True, exist_ok=True); p.touch()\nPY

# Common env
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CONFIG_PATH=/app/config/pipeline_config.yaml \
    GDC_MAPPING_PATH=/app/config/gdc_mapping.yaml

# default help
CMD ["python", "-m", "src.main", "--help"]