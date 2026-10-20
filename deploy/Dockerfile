# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
ARG TORCH_BACKEND=cpu
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY deploy/requirements-cu126.txt ./deploy/requirements-cu126.txt
# 两种依赖各自从空环境安装，运行时不再同步根锁，避免把 GPU 构建覆盖成 CPU。
RUN case "$TORCH_BACKEND" in \
      cpu) uv sync --locked --no-dev --no-editable ;; \
      cu126) uv venv /opt/venv \
        && uv pip sync --python /opt/venv/bin/python --require-hashes --torch-backend cu126 --no-build deploy/requirements-cu126.txt \
        && uv pip install --python /opt/venv/bin/python --no-deps . ;; \
      *) printf '%s\n' 'TORCH_BACKEND 仅支持 cpu/cu126' >&2; exit 2 ;; \
    esac

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    FACE_MODEL_PATH=/models/yunet_face.onnx PLATE_MODEL_PATH=/models/hyperlpr_plate.onnx \
    EGOBLUR_FACE_MODEL_PATH=/models/face.jit EGOBLUR_PLATE_MODEL_PATH=/models/license_plate.jit
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
USER 10001:10001
CMD ["python", "-m", "uvicorn", "vision_shield.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--limit-concurrency", "32"]
