FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/

RUN pip install --no-cache-dir -e ".[demo]"

# Default data & output mounts
VOLUME ["/app/data", "/app/runs"]

# Default: run training
ENTRYPOINT ["python", "-m", "eeg_classifier.training.train"]
CMD ["--config", "configs/default.yaml"]
