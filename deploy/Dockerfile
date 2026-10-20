# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
ARG TORCH_BACKEND=cpu
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY deploy/requirements-cu126.txt ./deploy/requirements-cu126.txt
# 源码和 README 不进入依赖层；缓存挂载在依赖变更时仍可复用下载。
RUN --mount=type=cache,target=/root/.cache/uv \
    case "$TORCH_BACKEND" in \
      cpu) uv sync --locked --no-dev --no-editable --no-install-project ;; \
      cu126) uv venv /opt/venv \
        && uv pip sync --python /opt/venv/bin/python --require-hashes --torch-backend cu126 --no-build deploy/requirements-cu126.txt ;; \
      *) printf '%s\n' 'TORCH_BACKEND 仅支持 cpu/cu126' >&2; exit 2 ;; \
    esac

FROM dependencies AS builder
COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /dist

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    FACE_MODEL_PATH=/models/yunet_face.onnx PLATE_MODEL_PATH=/models/hyperlpr_plate.onnx \
    EGOBLUR_FACE_MODEL_PATH=/models/face.jit EGOBLUR_PLATE_MODEL_PATH=/models/license_plate.jit
COPY --from=dependencies /opt/venv /opt/venv
# pip 的 NVIDIA 库不在系统默认搜索路径；注册全部组件供 ONNX Runtime 动态加载。
RUN /opt/venv/bin/python -c "from pathlib import Path; paths=sorted(Path('/opt/venv/lib/python3.12/site-packages/nvidia').glob('*/lib')); Path('/etc/ld.so.conf.d/vision-shield-cuda.conf').write_text(''.join(str(p)+'\n' for p in paths))" \
    && ldconfig
# 应用单独安装到新层，源码变化不重打包 PyTorch；不重算 CPU/GPU 依赖。
RUN --mount=from=dependencies,source=/usr/local/bin/uv,target=/usr/local/bin/uv \
    --mount=from=builder,source=/dist,target=/dist \
    uv pip install --no-cache --python /opt/venv/bin/python --no-deps /dist/*.whl
WORKDIR /app
USER 10001:10001
CMD ["python", "-m", "uvicorn", "vision_shield.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--limit-concurrency", "32"]
