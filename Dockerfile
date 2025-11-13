# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

RUN if [ "$USE_PROXY" = "true" ]; then \
      echo "Setting proxy variables..."; \
      export http_proxy=${HTTP_PROXY}; \
      export https_proxy=${HTTPS_PROXY}; \
    else \
      echo "Skipping proxy setup"; \
    fi && \
    echo "Continuing build steps..."

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      build-essential \
      ffmpeg \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages -r requirements.txt

ENV HF_HOME=/root/.cache/huggingface
RUN --mount=type=cache,target=/root/.cache/huggingface \
    echo "Using persistent Hugging Face cache at $HF_HOME"

COPY src ./src

EXPOSE 3000

ENV http_proxy=""
ENV https_proxy=""

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/src/config/google_credential.json"

ENTRYPOINT ["/bin/bash", "-c", "export LD_LIBRARY_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))') && exec \"$@\"", "--"]

CMD ["python3", "-m", "src.main", "--host", "0.0.0.0", "--port", "3000"]
