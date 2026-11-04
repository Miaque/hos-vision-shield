# 正式媒体依赖与 MPS/NUMA 发布

2026-11-04：用户确认测试环境验证完成后，将相同媒体制品接入正式构建。本记录不代表正式服务器已更新或验收通过。

## 发布范围

- 默认 `runtime` 继承已有 `rtsp-test`，正式 CPU/cu126 镜像安装同一 PyAV `19.0.2+hos.1` wheel，保留来源镜像 digest、wheel SHA256、离线无依赖安装及 `flush_mux()` 检查。文件名 `requirements-rtsp-test.txt` 保留，避免两份锁漂移。
- 正式 MediaMTX 使用与测试相同的 `1.21.2-gortsplib-5.6.7-hos.1` 镜像 digest，并设置 `MTX_PATHDEFAULTS_USEABSOLUTETIMESTAMP=true`。
- CUDA 12.6、GPU 单卡预留和 worker 副本数保留。将私有 MPS 共用定义从测试覆盖文件移到 GPU Compose，仅流 worker 接入 PID/IPC 和控制目录；测试的附加组 1022、全卡兼容映射及 UUID 覆盖保留，正式不复制这些机器专用配置。回放源不属于本次更新。
- 正式 NUMA 由 `.tools/prod-stream-affinity` 明确配置；部署入口在启动模型前应用并逐容器核对。该文件与测试的 `.tools/test-stream-affinity` 分离，不把测试机 node 2 或 CPU 编号作为默认值。
- 不修改根声明/基础依赖锁、应用逻辑、正式环境文件、模型或 Redis 数据。

## 本地验证

以下为媒体依赖候选 `cand-20261104-prod-media-01` 的结果；后续 MPS/NUMA 候选另行记录，不覆盖旧标签。

- 新增正式 PyAV 继承与 MediaMTX 制品/时间戳回归，修改前均按预期失败。
- 依赖锁、GPU 设备选择及部署入口定向回归：62 passed。全量单元：996 passed、96 skipped、12 warnings（186.43 秒）；未新增跳过或放宽门槛。
- 临时将绝对时间配置改为 false 时新增回归失败，恢复后依赖回归 6 passed。
- `uv build --wheel` 成功，包内应用和模型资产、文档链接核对通过。构建提示现有 uv_build 范围与本机 uv 版本不同，未修改项目构建依赖范围。
- 独立代码审查未发现本次引入的问题。本机没有 Docker，Windows 本地 PyAV 仍为官方 19.0.1；这些本地结果不代替正式 Linux CPU/cu126 镜像构建、实际 GPU、RTSP/HLS 或长期验收。

MPS/NUMA 候选 `cand-20261104-prod-mps-numa-01`：

- 新增及调整的12项回归在旧实现上均失败，初次依赖/GPU/部署定向72 passed。
- Compose 2.29.2 实际解析 CPU、正式 GPU 和测试 GPU 三种组合；测试合并后的完整配置与 `c1951ef` 逐项一致，正式 MPS/两个 worker 仍只预留同一张卡。
- 独立审查发现仅检查格式不能拒绝倒序范围或不存在节点，补充3项先失败的回归，并加入停旧 worker 前的 Docker 实际资源校验。修复后相关定向17 passed；反转校验条件时3项保护均失败，恢复后3 passed。
- 初次全量为1 failed、1005 passed、96 skipped（202.55秒），失败是未修改的 `test_slow_encoder_does_not_squeeze_following_final_packets_together` 的18ms墙钟间隔断言。相关应用源码与测试均和基线一致，单独在基线与候选各复测1 passed，未稳定复现，未改动该门槛；失败日志保留在本地 `.tools-unit-mps.log`。
- 最终完整单元1009 passed、96 skipped、12 warnings（206.96秒）。Bash语法、文档链接、wheel构建通过；独立复审确认阻断问题已解决，未发现新增问题。没有新增跳过或放宽测试。
- 本机只使用已有 Compose CLI 解析配置，没有 Docker 引擎；GPU、NUMA真实应用、MPS客户端及正式媒体链路仍待获授权后在目标机器验证。

## 正式拓扑准备

获得本次服务器授权后，只读核对正式 GPU、CPU/内存节点及已有 driver 工具：

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version --format=csv
nvidia-smi topo -m
for node in /sys/devices/system/node/node[0-9]*; do
  printf '%s: ' "$node"
  cat "$node/cpulist"
done
test -x /usr/bin/nvidia-cuda-mps-control && test -x /usr/bin/nvidia-cuda-mps-server
```

根据所选正式 GPU 的拓扑和当前 CPU 分配填写目标文件；下列两个占位值必须替换为现场核对的实际值。单 NUMA 节点也应填写实际 CPU 列表及内存节点0，不能猜测测试配置适用。

```bash
umask 077
mkdir -p .tools
printf '%s|%s\n' '<正式CPU列表>' '<正式内存节点列表>' > .tools/prod-stream-affinity
```

正式单卡容器内部使用编号0，避免将宿主卡号用于 MPS 重映射后的设备选择；测试全卡覆盖继续使用原所选 UUID。依据：[NVIDIA MPS 设备编号规则](https://docs.nvidia.com/deploy/topics/topic_5_2_1.html)。CPU 与内存节点通过 [docker update](https://docs.docker.com/reference/cli/docker/container/update/) 在停止状态的新 worker 上应用，不在运行中迁移内存。

正式部署在停旧 worker 前，用已构建的 CPU 镜像运行一次 `/bin/true`，无网络、挂载或 GPU 请求，先由 Docker/内核验证目标 CPU 和内存节点约束。失败不停止原流 worker。它能拒绝不可用集合，仍不能代替人工核对这些节点是否与所选 GPU 邻近、是否适合现有 CPU 分配；本机无 Docker 引擎，真实宿主校验与 MPS/GPU 启动尚待正式现场验证。

## 正式更新

服务器操作需要本次明确授权。操作前确认正式仓库、原正式版本及运行镜像，保存恢复所需环境文件和旧镜像；不要用测试候选标签猜测正式回退版本。保持 `.env.compose` 中的正式 GPU UUID、端口、Redis 数据库、目录与通知目标，设置 `DEBUG=false`。等待文件任务完成或安排维护窗口。

首次从旧入口升级时，先切换到新标签，让本次调用使用支持正式 MPS/NUMA 的脚本；旧脚本在 Git 切换前已解析函数，直接调用旧脚本不能保证本轮采用新流程。以下在已准备好正式亲和性文件、确认跟踪文件无修改的正式仓库根目录执行：

```bash
git diff --quiet HEAD -- &&
git -c http.version=HTTP/1.1 fetch origin refs/tags/cand-20261104-prod-mps-numa-01:refs/tags/cand-20261104-prod-mps-numa-01 &&
git switch --detach refs/tags/cand-20261104-prod-mps-numa-01 &&
bash scripts/deploy-server.sh --env prod --tag cand-20261104-prod-mps-numa-01
dc=(docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.gpu.yaml)
"${dc[@]}" ps
"${dc[@]}" logs --tail 100 api file-worker stream-worker mediamtx
```

镜像更新后先核对真实运行库和模型预检，再检查正式 API 的 `/health/ready`。以下命令检查运行中的正式 API 和每个现有 worker，不创建媒体任务：

```bash
for id in $("${dc[@]}" ps -q api file-worker stream-worker); do
  docker exec "$id" python -c 'import av; from importlib.metadata import version; assert av.__version__ == version("av") == "19.0.2+hos.1"; assert av.ffmpeg_version_info == "9.0.2"; assert hasattr(av.container.OutputContainer, "flush_mux"); print(av.__version__, av.ffmpeg_version_info)'
done
```

确认运行中的 MediaMTX 镜像 digest 及绝对时间设置；核对私有 MPS 健康、所选 GPU、每个流 worker 的实际 CPU/内存节点，且文件 worker 未加入 MPS。使用正式环境允许的样例验证图片/视频任务、实际 RTSP/HLS 播放、人脸/车牌遮挡、AAC/音画同步、双流吞吐及断线恢复；就绪成功不能代替这些验收。测试环境结果也不直接证明正式 CUDA 12.6 和目标 GPU 的运行结果。

构建期间旧容器继续服务，替换容器会短暂中断；未完成文件不续跑，运行意图保留的流按原机制尝试恢复。日常更新不先 `down`，不删除数据或使用 `--no-cache`，失败不会自动回滚。

## 回退

用更新前确认的正式标签执行 `bash scripts/deploy-server.sh --env prod --tag <原正式标签>`，必要时还原保存的正式环境与亲和性文件；核对镜像、模型预检和实际媒体输出。回退到不含 MPS 的版本时先停止本项目普通流 worker，切换回原 worker 后确认旧私有 MPS 无客户端才停止其控制容器；不使用 `--remove-orphans` 或操作其他项目。清除 NUMA 约束应通过原 Compose 重新创建 worker，不能依赖 `docker update` 的空值。旧镜像应保留到正式验收结束，不能仅凭 Git 提交号判断容器已经恢复。
