# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
ARG TORCH_BACKEND=cpu
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY deploy/requirements-cu126.txt deploy/requirements-cu130.txt ./deploy/
# 源码和 README 不进入依赖层；缓存挂载在依赖变更时仍可复用下载。
# 串行使用共享缓存，避免 CPU/GPU 并行构建等待 uv 文件锁超时。
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    case "$TORCH_BACKEND" in \
      cpu) uv sync --locked --no-dev --no-editable --no-install-project ;; \
      cu126|cu130) uv venv /opt/venv \
        && uv pip sync --python /opt/venv/bin/python --require-hashes --torch-backend "$TORCH_BACKEND" --no-build "deploy/requirements-$TORCH_BACKEND.txt" ;; \
      *) printf '%s\n' 'TORCH_BACKEND 仅支持 cpu/cu126/cu130' >&2; exit 2 ;; \
    esac

FROM dependencies AS builder
COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv build --wheel --out-dir /dist

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS app-runtime
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

FROM ghcr.io/miaque/pyav-runtime:19.0.2-hos.1@sha256:37592ce2dc1b2c2e07bd64eb78251e42e800f973b658845fb357bcf944fe2813 AS pyav-artifact

FROM app-runtime AS rtsp-test
USER root
RUN --mount=from=dependencies,source=/usr/local/bin/uv,target=/usr/local/bin/uv \
    --mount=from=pyav-artifact,source=/opt/wheels,target=/wheels \
    uv pip install --no-cache --no-index --no-deps --python /opt/venv/bin/python /wheels/*.whl \
    && python -c "import av; from importlib.metadata import version; assert av.__version__ == version('av') == '19.0.2+hos.1' and av.ffmpeg_version_info == '9.0.2' and hasattr(av.container.OutputContainer, 'flush_mux')"
USER 10001:10001

# The ordinary build keeps its original runtime; only compose.test selects rtsp-test.
FROM app-runtime AS runtime
