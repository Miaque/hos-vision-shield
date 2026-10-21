"""Run the deployment shell against local Git repos, never real Docker services."""

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def deployment(tmp_path):
    bash = shutil.which("bash")
    if not bash and os.name == "nt":
        bash = str(Path(shutil.which("git")).parents[1] / "bin/bash.exe")
    assert bash and Path(bash).is_file(), "Bash is required to test the server deployment script"
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C.UTF-8"}

    def git(*args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, env=env, check=True,
                              capture_output=True, text=True, encoding="utf-8", timeout=20).stdout.strip()

    def commit(directory):
        git("add", ".", cwd=directory)
        git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "测试部署更新",
            cwd=directory)

    remote = tmp_path / "remote"
    remote.mkdir()
    git("init", "-b", "main", cwd=remote)
    (remote / "scripts").mkdir()
    (remote / "scripts/deploy-server.sh").write_bytes((root / "scripts/deploy-server.sh").read_bytes())
    (remote / ".gitattributes").write_text("*.sh text eol=lf\n", encoding="utf-8")
    (remote / ".gitignore").write_text(".env.compose\ndata/\nmodels/*.jit\n", encoding="utf-8")
    (remote / "app.txt").write_text("old", encoding="utf-8")
    commit(remote)
    server = tmp_path / "server"
    git("clone", str(remote), str(server), cwd=tmp_path)
    (server / ".env.compose").write_text("server settings", encoding="utf-8")
    (server / "data").mkdir()
    (server / "data/result.bin").write_bytes(b"media")
    (server / "models").mkdir()
    (server / "models/face.jit").write_bytes(b"weights")
    (remote / "app.txt").write_text("latest", encoding="utf-8")
    commit(remote)

    binaries = tmp_path / "bin"
    binaries.mkdir()
    compose = binaries / "docker-compose"
    compose.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$DEPLOY_TEST_LOG"\n'
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == "${DEPLOY_TEST_FAIL:-}" ]]; then exit 17; fi\n'
        'done\n', encoding="utf-8", newline="\n",
    )
    compose.chmod(0o755)
    log = tmp_path / "compose.log"
    env.update(DEPLOY_TEST_LOG=log.as_posix(),
               PATH=os.pathsep.join([str(binaries), str(Path(bash).parent),
                                     str(Path(bash).parents[1] / "usr/bin"), env["PATH"]]))

    def run():
        return subprocess.run([bash, (server / "scripts/deploy-server.sh").as_posix()],
                              cwd=tmp_path, env=env, capture_output=True, text=True,
                              encoding="utf-8", timeout=30)

    return SimpleNamespace(server=server, remote=remote, env=env, log=log,
                           git=git, commit=commit, run=run)


def test_server_deployment_pulls_builds_and_waits_without_touching_data(deployment):
    d = deployment
    result = d.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert d.git("rev-parse", "HEAD", cwd=d.server) == d.git("rev-parse", "HEAD", cwd=d.remote)
    assert (d.server / "app.txt").read_text("utf-8") == "latest"
    assert (d.server / ".env.compose").read_text("utf-8") == "server settings"
    assert (d.server / "data/result.bin").read_bytes() == b"media"
    assert (d.server / "models/face.jit").read_bytes() == b"weights"
    commands = d.log.read_text("utf-8").splitlines()
    assert commands[1].endswith("-f compose.gpu.yaml config --quiet")
    assert commands[2].endswith(" build")
    assert commands[3].endswith(" up -d --no-build --wait --wait-timeout 240")
    assert "部署完成" in result.stdout


@pytest.mark.parametrize("failure,forbidden", [("config", " build"), ("build", " up "), ("up", " ps")])
def test_server_deployment_stops_on_compose_failure(deployment, failure, forbidden):
    d = deployment
    d.env["DEPLOY_TEST_FAIL"] = failure
    result = d.run()
    assert result.returncode != 0
    assert forbidden not in d.log.read_text("utf-8")
    assert "部署完成" not in result.stdout


@pytest.mark.parametrize("problem", ["dirty", "ahead", "diverged", "wrong_branch", "fetch"])
def test_server_deployment_rejects_unsynchronised_source(deployment, problem):
    d = deployment
    if problem == "fetch":
        d.git("remote", "set-url", "origin", str(d.remote / "missing"), cwd=d.server)
    elif problem == "wrong_branch":
        d.git("switch", "-c", "feature/local", cwd=d.server)
    else:
        if problem == "ahead":
            d.git("pull", "--ff-only", "origin", "main", cwd=d.server)
        (d.server / "app.txt").write_text("server change", encoding="utf-8")
        if problem != "dirty":
            d.commit(d.server)
    result = d.run()
    assert result.returncode != 0
    assert " build" not in d.log.read_text("utf-8")
    assert " up " not in d.log.read_text("utf-8")


def test_server_deployment_survives_updating_its_own_script(deployment):
    d = deployment
    script = d.remote / "scripts/deploy-server.sh"
    script.write_text("# Updated during pull\n" * 500 + script.read_text("utf-8"), encoding="utf-8", newline="\n")
    d.commit(d.remote)
    result = d.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "部署完成" in result.stdout
