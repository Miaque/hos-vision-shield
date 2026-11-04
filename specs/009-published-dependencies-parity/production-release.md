# 正式媒体依赖发布

2026-11-04：用户确认测试环境验证完成后，将相同媒体制品接入正式构建。本记录不代表正式服务器已更新或验收通过。

## 发布范围

- 默认 `runtime` 继承已有 `rtsp-test`，正式 CPU/cu126 镜像安装同一 PyAV `19.0.2+hos.1` wheel，保留来源镜像 digest、wheel SHA256、离线无依赖安装及 `flush_mux()` 检查。文件名 `requirements-rtsp-test.txt` 保留，避免两份锁漂移。
- 正式 MediaMTX 使用与测试相同的 `1.21.2-gortsplib-5.6.7-hos.1` 镜像 digest，并设置 `MTX_PATHDEFAULTS_USEABSOLUTETIMESTAMP=true`。
- CUDA 12.6、GPU 单卡预留、worker 副本数及 `--env prod` 入口保留。测试机的 MPS、附加组 1022、全卡可见及 NUMA 配置不迁入正式环境；回放源不属于本次更新。
- 不修改根声明/基础依赖锁、应用逻辑、正式环境文件、模型或 Redis 数据。

## 本地验证

- 新增正式 PyAV 继承与 MediaMTX 制品/时间戳回归，修改前均按预期失败。
- 依赖锁、GPU 设备选择及部署入口定向回归：62 passed。全量单元：996 passed、96 skipped、12 warnings（186.43 秒）；未新增跳过或放宽门槛。
- 临时将绝对时间配置改为 false 时新增回归失败，恢复后依赖回归 6 passed。
- `uv build --wheel` 成功，包内应用和模型资产、文档链接核对通过。构建提示现有 uv_build 范围与本机 uv 版本不同，未修改项目构建依赖范围。
- 独立代码审查未发现本次引入的问题。本机没有 Docker，Windows 本地 PyAV 仍为官方 19.0.1；这些本地结果不代替正式 Linux CPU/cu126 镜像构建、实际 GPU、RTSP/HLS 或长期验收。

## 正式更新

服务器操作需要本次明确授权。操作前确认正式仓库、原正式版本及运行镜像，保存恢复所需环境文件和旧镜像；不要用测试候选标签猜测正式回退版本。保持 `.env.compose` 中的正式 GPU UUID、端口、Redis 数据库、目录与通知目标，设置 `DEBUG=false`。等待文件任务完成或安排维护窗口。

从已推送的正式候选标签更新，以下在正式仓库根目录执行：

```bash
bash scripts/deploy-server.sh --env prod --tag cand-20261104-prod-media-01
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

确认运行中的 MediaMTX 镜像 digest 及绝对时间设置。使用正式环境允许的样例验证图片/视频任务、实际 RTSP/HLS 播放、人脸/车牌遮挡、AAC/音画同步、双流吞吐及断线恢复；就绪成功不能代替这些验收。测试环境结果也不直接证明正式 CUDA 12.6 和目标 GPU 的运行结果。

构建期间旧容器继续服务，替换容器会短暂中断；未完成文件不续跑，运行意图保留的流按原机制尝试恢复。日常更新不先 `down`，不删除数据或使用 `--no-cache`，失败不会自动回滚。

## 回退

用更新前确认的正式标签执行 `bash scripts/deploy-server.sh --env prod --tag <原正式标签>`，必要时还原保存的正式环境文件；核对镜像、模型预检和实际媒体输出。旧镜像应保留到正式验收结束，不能仅凭 Git 提交号判断容器已经恢复。
