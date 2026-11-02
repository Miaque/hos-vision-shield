#!/usr/bin/env bash
set -Eeuo pipefail

deploy_stage='检查部署环境'
trap 'printf "部署失败，阶段：%s。修复后可重新执行脚本。\n" "$deploy_stage" >&2' ERR

# 先解析完整函数，避免 Git 更新改写当前脚本影响本次执行。
main() {
    local environment='test' env_file='.env.compose.test' project='vision-shield-test'
    if [[ "${1:-}" == --env ]]; then
        environment="${2:-}"
        case "$environment" in
            test|prod) shift 2 ;;
            *) printf '部署环境仅支持 test/prod，请使用 --env test 或 --env prod。\n' >&2; return 1 ;;
        esac
    fi
    if [[ "$environment" == prod ]]; then
        env_file='.env.compose'
        project='vision-shield'
    fi

    local branch="${1:-main}" tag='' remote_ref ref_kind='分支' ref_name
    if [[ "$branch" == --tag && $# == 2 ]]; then
        tag="$2"
        git check-ref-format "refs/tags/$tag"
    elif (( $# <= 1 )); then
        git check-ref-format --branch "$branch" >/dev/null
    else
        printf '用法：bash scripts/deploy-server.sh [--env test|prod] [远程分支 | --tag 标签]（默认 test/main）\n' >&2
        return 1
    fi
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
    if [[ ! -e .git || ! -f "$env_file" ]]; then
        printf '请先完成当前目录的 Git 初始化，并准备 %s。\n' "$env_file" >&2
        return 1
    fi
    docker-compose version >/dev/null
    if ! git diff --quiet HEAD --; then
        printf '存在未提交的跟踪文件修改，请先处理后再部署。\n' >&2
        return 1
    fi
    printf '部署环境：%s，项目：%s，配置：%s\n' "$environment" "$project" "$env_file"

    if [[ -n "$tag" ]]; then
        ref_kind='tag'
        ref_name="$tag"
        remote_ref="refs/tags/$tag"
        deploy_stage="拉取远程 tag $tag"
        printf '%s\n' "$deploy_stage"
        git -c http.version=HTTP/1.1 fetch origin "$remote_ref:$remote_ref"
        git switch --detach "$remote_ref"
    else
        ref_name="$branch"
        remote_ref="refs/remotes/origin/$branch"
        deploy_stage="拉取远程分支 $branch"
        printf '%s\n' "$deploy_stage"
        git -c http.version=HTTP/1.1 fetch origin "refs/heads/$branch:$remote_ref"
        if git show-ref --verify --quiet "refs/heads/$branch"; then
            if ! git merge-base --is-ancestor "refs/heads/$branch" "$remote_ref"; then
                printf '目标分支存在本地额外提交或分叉，请先处理。\n' >&2
                return 1
            fi
            git switch "$branch"
        else
            git switch --track -c "$branch" "$remote_ref"
        fi
        git merge --ff-only "$remote_ref"
    fi
    if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "$remote_ref^{commit}")" ]]; then
        printf '本地提交与指定远程版本不一致，请先处理。\n' >&2
        return 1
    fi

    local -a dc=(docker-compose --env-file "$env_file" -p "$project"
                 -f compose.yaml -f compose.gpu.yaml)
    if [[ "$environment" == test ]]; then
        dc+=(-f compose.test.yaml)
    fi
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
    printf '部署完成，环境：%s，%s：%s，提交：%s\n' "$environment" "$ref_kind" "$ref_name" "$(git rev-parse --short HEAD)"
}

main "$@"; exit
