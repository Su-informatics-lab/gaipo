# syntax=docker/dockerfile:1.7
# DockerImage/Dockerfile
FROM python:3.12-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# project files
COPY config/ /app/config/
COPY src/ /app/src/
COPY data/ /app/data/

# ensure src is a package (no heredoc; single-line python)
RUN python -c "import pathlib; p=pathlib.Path('/app/src/__init__.py'); p.parent.mkdir(parents=True, exist_ok=True); p.touch()"

ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=/app/config/pipeline_config.yaml

CMD ["python", "-m", "src.main", "--help"]