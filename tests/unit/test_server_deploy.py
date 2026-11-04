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
    (remote / ".gitignore").write_text(".env.compose\n.env.compose.test\ndata/\nmodels/*.jit\n", encoding="utf-8")
    (remote / "app.txt").write_text("old", encoding="utf-8")
    commit(remote)
    server = tmp_path / "server"
    git("clone", str(remote), str(server), cwd=tmp_path)
    (server / ".env.compose").write_text("server settings", encoding="utf-8")
    (server / ".env.compose.test").write_text("test settings", encoding="utf-8")
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
        'done\n'
        'if [[ "$*" == *"config --services"* && "${DEPLOY_TEST_MPS:-}" == 1 ]]; then echo cuda-mps; fi\n'
        'if [[ "$*" == *"create --no-build stream-worker"* ]]; then touch "$DEPLOY_TEST_CREATED"; fi\n',
        encoding="utf-8", newline="\n",
    )
    compose.chmod(0o755)
    docker = binaries / "docker"
    docker.write_text('''#!/usr/bin/env bash
printf 'docker %s\\n' "$*" >> "$DEPLOY_TEST_LOG"
case "$1" in
  ps)
    if [[ "${DEPLOY_TEST_COLD:-}" != 1 || -f "$DEPLOY_TEST_CREATED" ]]; then
      printf 'worker-1\\nworker-2\\n'
    fi ;;
  inspect)
    if [[ -f "$DEPLOY_TEST_UPDATED" && "${DEPLOY_TEST_LOST_AFFINITY:-}" == 1 ]]; then echo '|'
    elif [[ -f "$DEPLOY_TEST_UPDATED" ]]; then cat "$DEPLOY_TEST_UPDATED"
    elif [[ "${DEPLOY_TEST_MIXED:-}" == 1 && "${!#}" == worker-2 ]]; then echo '0-3|0'
    else echo '32-47,96-111|2'; fi ;;
  update)
    printf '%s|%s\\n' "$3" "$5" > "$DEPLOY_TEST_UPDATED" ;;
esac
''', encoding="utf-8", newline="\n")
    docker.chmod(0o755)
    log = tmp_path / "compose.log"
    env.update(DEPLOY_TEST_LOG=log.as_posix(), DEPLOY_TEST_CREATED=(tmp_path / 'created').as_posix(),
               DEPLOY_TEST_UPDATED=(tmp_path / 'updated').as_posix(),
               PATH=os.pathsep.join([str(binaries), str(Path(bash).parent),
                                     str(Path(bash).parents[1] / "usr/bin"), env["PATH"]]))

    def run(*args):
        return subprocess.run([bash, (server / "scripts/deploy-server.sh").as_posix(), *args],
                              cwd=tmp_path, env=env, capture_output=True, text=True,
                              encoding="utf-8", timeout=30)

    return SimpleNamespace(server=server, remote=remote, env=env, log=log,
                           git=git, commit=commit, run=run)


@pytest.mark.parametrize("args,environment,env_file,project,extra_file", [
    ((), "test", ".env.compose.test", "vision-shield-test", " -f compose.test.yaml"),
    (("--env", "test"), "test", ".env.compose.test", "vision-shield-test", " -f compose.test.yaml"),
    (("--env", "prod"), "prod", ".env.compose", "vision-shield", ""),
])
def test_server_deployment_pulls_builds_and_waits_without_touching_data(
    deployment, args, environment, env_file, project, extra_file,
):
    d = deployment
    result = d.run(*args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert d.git("rev-parse", "HEAD", cwd=d.server) == d.git("rev-parse", "HEAD", cwd=d.remote)
    assert (d.server / "app.txt").read_text("utf-8") == "latest"
    assert (d.server / ".env.compose").read_text("utf-8") == "server settings"
    assert (d.server / ".env.compose.test").read_text("utf-8") == "test settings"
    assert (d.server / "data/result.bin").read_bytes() == b"media"
    assert (d.server / "models/face.jit").read_bytes() == b"weights"
    commands = d.log.read_text("utf-8").splitlines()
    prefix = f"--env-file {env_file} -p {project} -f compose.yaml -f compose.gpu.yaml{extra_file}"
    assert all(command.startswith(prefix + " ") for command in commands[1:])
    assert commands[1].endswith(" config --quiet")
    assert commands[2].endswith(" build")
    assert any(command.endswith(" up -d --no-build --wait --wait-timeout 240") for command in commands)
    assert "部署完成" in result.stdout
    assert f"环境：{environment}" in result.stdout


@pytest.mark.parametrize("args,env_file", [
    ((), ".env.compose.test"),
    (("--env", "prod"), ".env.compose"),
])
def test_server_deployment_requires_the_selected_environment_file(deployment, args, env_file):
    d = deployment
    (d.server / env_file).unlink()
    original_head = d.git("rev-parse", "HEAD", cwd=d.server)

    result = d.run(*args)

    assert result.returncode != 0
    assert env_file in result.stderr
    assert d.git("rev-parse", "HEAD", cwd=d.server) == original_head
    assert not d.log.exists()


@pytest.mark.parametrize("args,other_file", [
    ((), ".env.compose"),
    (("--env", "prod"), ".env.compose.test"),
])
def test_server_deployment_does_not_require_the_other_environment_file(deployment, args, other_file):
    d = deployment
    (d.server / other_file).unlink()

    result = d.run(*args)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("args", [
    ("--env",), ("--env", ""), ("--env", "staging"),
    ("--env", "test", "--env", "prod"), ("--env", "prod", "main", "extra"),
])
def test_server_deployment_rejects_invalid_environment_arguments(deployment, args):
    d = deployment
    original_head = d.git("rev-parse", "HEAD", cwd=d.server)

    result = d.run(*args)

    assert result.returncode != 0
    assert d.git("rev-parse", "HEAD", cwd=d.server) == original_head
    assert not d.log.exists()


@pytest.mark.parametrize("failure,forbidden", [("config", " build"), ("build", " up "), ("up", " ps")])
def test_server_deployment_stops_on_compose_failure(deployment, failure, forbidden):
    d = deployment
    d.env["DEPLOY_TEST_FAIL"] = failure
    result = d.run()
    assert result.returncode != 0
    assert forbidden not in d.log.read_text("utf-8")
    assert "部署完成" not in result.stdout


@pytest.mark.parametrize("problem", ["dirty", "ahead", "diverged", "fetch"])
def test_server_deployment_rejects_unsynchronised_source(deployment, problem):
    d = deployment
    if problem == "fetch":
        d.git("remote", "set-url", "origin", str(d.remote / "missing"), cwd=d.server)
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


def test_server_deployment_defaults_to_origin_main_from_another_branch(deployment):
    d = deployment
    branch = "feature/local"
    d.git("switch", "-c", branch, cwd=d.server)
    (d.server / "app.txt").write_text("local branch work", encoding="utf-8")
    d.commit(d.server)
    original_branch = d.git("rev-parse", branch, cwd=d.server)

    result = d.run()

    assert result.returncode == 0, result.stdout + result.stderr
    assert d.git("branch", "--show-current", cwd=d.server) == "main"
    assert d.git("rev-parse", "HEAD", cwd=d.server) == d.git("rev-parse", "main", cwd=d.remote)
    assert d.git("rev-parse", branch, cwd=d.server) == original_branch
    assert (d.server / "app.txt").read_text("utf-8") == "latest"


def test_server_deployment_survives_updating_its_own_script(deployment):
    d = deployment
    script = d.remote / "scripts/deploy-server.sh"
    script.write_text("# Updated during pull\n" * 500 + script.read_text("utf-8"), encoding="utf-8", newline="\n")
    d.commit(d.remote)
    result = d.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "部署完成" in result.stdout


@pytest.mark.parametrize("existing_branch", [False, True])
@pytest.mark.parametrize("env_args", [(), ("--env", "test"), ("--env", "prod")])
def test_server_deployment_selects_and_fast_forwards_an_explicit_branch(deployment, existing_branch, env_args):
    d = deployment
    branch = "feature/gpu-throughput-optimization"
    d.git("switch", "-c", branch, cwd=d.remote)
    if existing_branch:
        d.git("fetch", "origin", cwd=d.server)
        d.git("branch", branch, f"origin/{branch}", cwd=d.server)
    (d.remote / "app.txt").write_text("experiment", encoding="utf-8")
    d.commit(d.remote)
    original_main = d.git("rev-parse", "main", cwd=d.server)

    result = d.run(*env_args, branch)

    assert result.returncode == 0, result.stdout + result.stderr
    assert d.git("branch", "--show-current", cwd=d.server) == branch
    assert d.git("rev-parse", "HEAD", cwd=d.server) == d.git("rev-parse", "HEAD", cwd=d.remote)
    assert d.git("rev-parse", "main", cwd=d.server) == original_main
    assert (d.server / "app.txt").read_text("utf-8") == "experiment"
    assert (d.server / ".env.compose").read_text("utf-8") == "server settings"
    assert (d.server / "data/result.bin").read_bytes() == b"media"
    assert (d.server / "models/face.jit").read_bytes() == b"weights"
    assert branch in result.stdout

    experiment_head = d.git("rev-parse", branch, cwd=d.server)
    result = d.run(*env_args, "main")
    assert result.returncode == 0, result.stdout + result.stderr
    assert d.git("branch", "--show-current", cwd=d.server) == "main"
    assert d.git("rev-parse", "HEAD", cwd=d.server) == d.git("rev-parse", "main", cwd=d.remote)
    assert d.git("rev-parse", branch, cwd=d.server) == experiment_head


@pytest.mark.parametrize("problem", ["dirty", "ahead", "diverged", "untracked"])
def test_server_deployment_preserves_conflicting_experiment_work(deployment, problem):
    d = deployment
    branch = "feature/experiment"
    d.git("switch", "-c", branch, cwd=d.remote)
    d.git("fetch", "origin", cwd=d.server)
    if problem in {"ahead", "diverged"}:
        d.git("switch", "-c", branch, f"origin/{branch}", cwd=d.server)
        (d.server / "app.txt").write_text("local commit", encoding="utf-8")
        d.commit(d.server)
        d.git("switch", "main", cwd=d.server)
    elif problem == "dirty":
        (d.server / "app.txt").write_text("uncommitted", encoding="utf-8")
    else:
        (d.server / "experiment.txt").write_text("local untracked", encoding="utf-8")
        (d.remote / "experiment.txt").write_text("remote tracked", encoding="utf-8")
        d.commit(d.remote)
    if problem == "diverged":
        (d.remote / "app.txt").write_text("remote commit", encoding="utf-8")
        d.commit(d.remote)
    original_head = d.git("rev-parse", "HEAD", cwd=d.server)

    result = d.run(branch)

    assert result.returncode != 0
    assert d.git("branch", "--show-current", cwd=d.server) == "main"
    assert d.git("rev-parse", "HEAD", cwd=d.server) == original_head
    assert " build" not in d.log.read_text("utf-8")
    assert " up " not in d.log.read_text("utf-8")
    if problem == "untracked":
        assert (d.server / "experiment.txt").read_text("utf-8") == "local untracked"


@pytest.mark.parametrize("args", [("feature/missing",), ("--force",), ("bad..branch",), ("main", "extra")])
def test_server_deployment_rejects_unknown_or_invalid_branch(deployment, args):
    d = deployment
    original_head = d.git("rev-parse", "HEAD", cwd=d.server)
    result = d.run(*args)
    assert result.returncode != 0
    assert d.git("rev-parse", "HEAD", cwd=d.server) == original_head
    if d.log.exists():
        assert " build" not in d.log.read_text("utf-8")


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("env_args", [(), ("--env", "test"), ("--env", "prod")])
def test_server_deployment_selects_an_exact_tag_without_moving_branches(deployment, annotated, env_args):
    d = deployment
    tag = "exp-test-p0-01"
    if annotated:
        d.git("-c", "user.name=Test", "-c", "user.email=test@example.com",
              "tag", "-a", tag, "-m", "固定实验版本", cwd=d.remote)
    else:
        d.git("tag", tag, cwd=d.remote)
    original_main = d.git("rev-parse", "main", cwd=d.server)

    result = d.run(*env_args, "--tag", tag)

    assert result.returncode == 0, result.stdout + result.stderr
    assert d.git("branch", "--show-current", cwd=d.server) == ""
    assert d.git("rev-parse", "HEAD", cwd=d.server) == d.git("rev-parse", "HEAD", cwd=d.remote)
    assert d.git("rev-parse", "main", cwd=d.server) == original_main
    assert (d.server / ".env.compose").read_text("utf-8") == "server settings"
    assert (d.server / "data/result.bin").read_bytes() == b"media"
    assert (d.server / "models/face.jit").read_bytes() == b"weights"
    assert tag in result.stdout

    # A remote branch can advance independently; repeating the tag keeps the fixed version.
    fixed_commit = d.git("rev-parse", "HEAD", cwd=d.server)
    (d.remote / "app.txt").write_text("next release", encoding="utf-8")
    d.commit(d.remote)
    result = d.run(*env_args, "--tag", tag)
    assert result.returncode == 0, result.stdout + result.stderr
    assert d.git("rev-parse", "HEAD", cwd=d.server) == fixed_commit
    result = d.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert d.git("branch", "--show-current", cwd=d.server) == "main"
    assert (d.server / "app.txt").read_text("utf-8") == "next release"


def test_server_deployment_rejects_replaced_tag(deployment):
    d = deployment
    tag = "exp-immutable"
    d.git("tag", tag, cwd=d.server)
    d.git("tag", tag, cwd=d.remote)
    original = d.git("rev-parse", tag, cwd=d.server)
    result = d.run("--tag", tag)
    assert result.returncode != 0
    assert d.git("rev-parse", tag, cwd=d.server) == original
    assert d.git("rev-parse", "HEAD", cwd=d.server) == original
    assert " build" not in d.log.read_text("utf-8")


@pytest.mark.parametrize("args", [("--tag",), ("--tag", "missing"), ("--tag", "bad..tag"),
                                  ("--tag", "ok", "extra")])
def test_server_deployment_rejects_unknown_or_invalid_tag(deployment, args):
    d = deployment
    original = d.git("rev-parse", "HEAD", cwd=d.server)
    result = d.run(*args)
    assert result.returncode != 0
    assert d.git("rev-parse", "HEAD", cwd=d.server) == original
    if d.log.exists():
        assert " build" not in d.log.read_text("utf-8")


@pytest.mark.parametrize('cold', [False, True])
def test_test_deployment_restores_affinity_before_starting_mps_clients(deployment, cold):
    d = deployment
    d.env.update(DEPLOY_TEST_MPS='1', DEPLOY_TEST_COLD=str(int(cold)))
    profile = d.server / '.tools/test-stream-affinity'
    if cold:
        profile.parent.mkdir()
        profile.write_text('32-47,96-111|2\n', encoding='utf-8')
    result = d.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert profile.read_text('utf-8') == '32-47,96-111|2\n'
    commands = d.log.read_text('utf-8').splitlines()
    positions = [next(i for i, command in enumerate(commands) if needle in command) for needle in (
        ' build', ' stop stream-worker', ' --scale stream-worker=0', ' create --no-build stream-worker',
        'docker update --cpuset-cpus 32-47,96-111 --cpuset-mems 2 worker-1 worker-2',
        ' up -d --no-build --wait --wait-timeout 240')]
    assert positions == sorted(positions)
    assert all('--filter label=com.docker.compose.oneoff=False' in command
               for command in commands if command.startswith('docker ps '))
    assert '--remove-orphans' not in '\n'.join(commands)


def test_test_deployment_rejects_differing_affinity_before_stopping_workers(deployment):
    d = deployment
    d.env.update(DEPLOY_TEST_MPS='1', DEPLOY_TEST_MIXED='1')
    result = d.run()
    assert result.returncode != 0
    assert ' stop ' not in d.log.read_text('utf-8')
    assert ' up ' not in d.log.read_text('utf-8')


def test_failed_build_does_not_stop_mps_clients_or_overwrite_saved_affinity(deployment):
    d = deployment
    d.env.update(DEPLOY_TEST_MPS='1', DEPLOY_TEST_FAIL='build')
    profile = d.server / '.tools/test-stream-affinity'
    profile.parent.mkdir()
    profile.write_text('32-47,96-111|2\n', encoding='utf-8')
    result = d.run()
    assert result.returncode != 0
    assert ' stop ' not in d.log.read_text('utf-8')
    assert profile.read_text('utf-8') == '32-47,96-111|2\n'


def test_test_deployment_refuses_to_report_success_if_affinity_is_lost(deployment):
    d = deployment
    d.env.update(DEPLOY_TEST_MPS='1', DEPLOY_TEST_LOST_AFFINITY='1')
    result = d.run()
    assert result.returncode != 0
    assert '部署完成' not in result.stdout


def test_production_deployment_does_not_use_test_mps_or_affinity(deployment):
    d = deployment
    d.env.update(DEPLOY_TEST_MPS='1')
    result = d.run('--env', 'prod')
    assert result.returncode == 0, result.stdout + result.stderr
    commands = d.log.read_text('utf-8')
    assert 'docker ' not in commands and ' stop ' not in commands and ' create ' not in commands
    assert not (d.server / '.tools/test-stream-affinity').exists()
