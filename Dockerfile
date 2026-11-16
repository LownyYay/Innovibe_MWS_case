ARG RUNTIME_IMAGE=python:3.12-slim-bookworm
FROM ${RUNTIME_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PREBUILT_RUNTIME=0

USER root
WORKDIR /opt/compactfact
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="" \
    HF_HUB_OFFLINE=0 \
    HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.txt /tmp/compactfact-requirements.txt
RUN if [ "$PREBUILT_RUNTIME" = 0 ]; then \
      apt-get update && apt-get install -y --no-install-recommends libgomp1 \
      && rm -rf /var/lib/apt/lists/* \
      && python -m pip install --no-cache-dir "torch==2.7.1" --index-url "$TORCH_INDEX"; \
    else \
      python -c 'import torch; assert torch.__version__.split("+")[0] == "2.7.1"'; \
    fi \
    && python -m pip install --no-cache-dir -r /tmp/compactfact-requirements.txt \
    && (id appuser >/dev/null 2>&1 || useradd --uid 10001 --create-home appuser)

COPY inference/extractor/download_model.py inference/extractor/data_v21.py inference/extractor/inference_contract.json inference/extractor/training_config.json /tmp/download/
RUN if [ ! -s /models/qwen/model.safetensors ] && [ ! -s /models/qwen/model.safetensors.index.json ]; then \
      python /tmp/download/download_model.py --output /models/qwen; \
    fi \
    && rm -rf /models/qwen/.cache /tmp/download /tmp/compactfact-requirements.txt

COPY inference/ ./inference/
COPY weights/ ./weights/
COPY artifact_identity.json ./

ENV FACT_BASE_MODEL=/models/qwen HF_HUB_OFFLINE=1
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=5m \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8080/readyz', timeout=2).close()"]
ENTRYPOINT ["python", "-B", "inference/service.py"]
CMD ["--device", "auto"]
