"""GPU 部署锁应沿用共同版本，不改变本机依赖来源。"""

from pathlib import Path
import re
import tomllib

import pytest

from packaging.markers import default_environment
from packaging.requirements import Requirement


@pytest.mark.parametrize("backend", ["cu126", "cu130"])
def test_gpu_lock_keeps_common_versions_and_root_requirements(backend):
    root = Path(__file__).resolve().parents[2]
    lock = tomllib.loads((root / "uv.lock").read_text("utf-8"))
    versions = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    text = (root / f"deploy/requirements-{backend}.txt").read_text("utf-8")
    gpu = dict(re.findall(r"(?m)^([a-z0-9-]+)==([^\s\\]+)", text))
    assert gpu["torch"] == f"2.14.1+{backend}" and gpu["torchvision"] == f"0.29.1+{backend}"
    changed = {"torch", "torchvision"}
    if backend == "cu130":
        assert gpu["onnxruntime-gpu"] == "1.31.1"
        assert gpu["cuda-bindings"].startswith("13.0.")
        assert gpu["nvidia-nvjitlink"].startswith("13.0.")
        changed.update({"onnxruntime-gpu", "cuda-bindings", "cuda-toolkit"})
    for name in gpu.keys() & versions.keys() - changed:
        assert gpu[name] in versions[name], name
    environment = {**default_environment(), "sys_platform": "linux", "platform_machine": "x86_64",
                   "python_version": "3.12", "python_full_version": "3.12.14"}
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
    for raw in project["project"]["dependencies"] + project["project"]["optional-dependencies"]["cuda"]:
        requirement = Requirement(raw)
        if backend == "cu130" and requirement.name == "onnxruntime-gpu":
            requirement = Requirement("onnxruntime-gpu==1.31.1")
        if requirement.marker is None or requirement.marker.evaluate(environment):
            assert gpu[requirement.name] in requirement.specifier, raw
    # 每项 pin 必须保留至少一个哈希，禁止用未锁定的索引结果补齐。
    for block in re.split(r"\n(?=[a-z0-9-]+==)", text):
        if re.search(r"(?m)^[a-z0-9-]+==", block):
            assert "--hash=sha256:" in block


def test_image_dependency_layers_do_not_depend_on_source_or_readme():
    root = Path(__file__).resolve().parents[2]
    text = (root / "deploy/Dockerfile").read_text("utf-8")
    application_copy = min(match.start() for match in re.finditer(r"(?m)^COPY .*", text)
                           if "src/" in match[0] or "README.md" in match[0])
    dependencies = text[:application_copy]
    application = text[application_copy:]
    assert "--no-install-project" in dependencies, "CPU 依赖须在复制源码前独立安装"
    assert "uv pip sync" in dependencies and "--require-hashes" in dependencies
    assert "FROM dependencies AS builder" in dependencies
    assert "uv build --wheel" in application
    assert "COPY --from=dependencies /opt/venv /opt/venv" in application
    assert "COPY --from=builder /opt/venv" not in application
    assert "uv pip install" in application and "--no-deps /dist/*.whl" in application
