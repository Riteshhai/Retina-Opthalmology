FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS dependencies
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES="" STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 8 torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 8 -r requirements.txt

FROM dependencies AS aptos-training
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 qsart
USER qsart

FROM dependencies AS runtime
COPY qsart ./qsart
COPY api ./api
COPY dashboard ./dashboard
COPY configs ./configs
RUN useradd --create-home --uid 10001 qsart
USER qsart
EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM runtime AS portal
USER root
COPY services ./services
RUN install -d -m 0700 -o qsart -g qsart /var/lib/qsart/portal-images
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c 'import tomllib; from pathlib import Path; print("\n".join(tomllib.loads(Path("pyproject.toml").read_text())["project"]["optional-dependencies"]["portal"]))' > /tmp/portal-requirements.txt \
    && pip install --timeout 120 --retries 8 -r /tmp/portal-requirements.txt \
    && rm /tmp/portal-requirements.txt
USER qsart

FROM runtime AS default
