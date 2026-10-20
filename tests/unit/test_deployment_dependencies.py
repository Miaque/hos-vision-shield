"""GPU 部署锁应沿用共同版本，不改变本机依赖来源。"""

from pathlib import Path
import re
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement


def test_gpu_lock_keeps_common_versions_and_root_requirements():
    root = Path(__file__).resolve().parents[2]
    lock = tomllib.loads((root / "uv.lock").read_text("utf-8"))
    versions = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    text = (root / "deploy/requirements-cu126.txt").read_text("utf-8")
    gpu = dict(re.findall(r"(?m)^([a-z0-9-]+)==([^\s\\]+)", text))
    assert gpu["torch"] == "2.14.1+cu126" and gpu["torchvision"] == "0.29.1+cu126"
    for name in gpu.keys() & versions.keys() - {"torch", "torchvision"}:
        assert gpu[name] in versions[name], name
    environment = {**default_environment(), "sys_platform": "linux", "platform_machine": "x86_64",
                   "python_version": "3.12", "python_full_version": "3.12.14"}
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
    for raw in project["project"]["dependencies"]:
        requirement = Requirement(raw)
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
