#!/usr/bin/env bash
set -Eeuo pipefail

deploy_stage='检查部署环境'
trap 'printf "部署失败，阶段：%s。修复后可重新执行脚本。\n" "$deploy_stage" >&2' ERR

# 先解析完整函数，避免 git pull 改写当前脚本影响本次执行。
main() {
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
    if [[ ! -e .git || ! -f .env.compose ]]; then
        printf '请先完成当前目录的 Git 初始化，并准备 .env.compose。\n' >&2
        return 1
    fi
    docker-compose version >/dev/null
    if [[ "$(git branch --show-current)" != main ]]; then
        printf '请先切换到 main 分支。\n' >&2
        return 1
    fi
    if ! git diff --quiet HEAD --; then
        printf '存在未提交的跟踪文件修改，请先处理后再部署。\n' >&2
        return 1
    fi

    deploy_stage='拉取远程 main'
    printf '%s\n' "$deploy_stage"
    git -c http.version=HTTP/1.1 pull --ff-only origin main
    if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
        printf '本地提交与远程 main 不一致，请先处理本地额外提交。\n' >&2
        return 1
    fi

    local -a dc=(docker-compose --env-file .env.compose -p vision-shield
                 -f compose.yaml -f compose.gpu.yaml)
    deploy_stage='检查 Compose 配置'
    printf '%s\n' "$deploy_stage"
    "${dc[@]}" config --quiet

    deploy_stage='构建 CPU/GPU 镜像'
    printf '%s\n' "$deploy_stage"
    "${dc[@]}" build

    deploy_stage='更新容器并等待就绪（最多 240 秒）'
    printf '%s\n' "$deploy_stage"
    "${dc[@]}" up -d --no-build --wait --wait-timeout 240
    "${dc[@]}" ps
    printf '部署完成，提交：%s\n' "$(git rev-parse --short HEAD)"
}

main "$@"; exit
