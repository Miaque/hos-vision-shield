# 部署契约：CPU 与指定单卡 GPU

本文件定义已实现的部署交付接口；初次验证边界见[实施记录](../validation/gates.md)，后续服务器分阶段实测见[时序优化记录](../temporal-optimization.md)，完整部署验收仍须按[验收指南](../quickstart.md)执行。目标为 Ubuntu 24.04.4 x86_64、RTX 6000 Ada，沿用既有独立入口 `docker-compose v2.29.2`，不在 Windows 本机安装 Docker。

## 交付与调用方式

| 文件 | 职责 |
| --- | --- |
| `compose.yaml` | 五服务 CPU 部署，复用现有带密码 Redis，不预留 GPU |
| `compose.gpu.yaml` | 覆盖两类 worker 的镜像、构建参数、检测后端、设备设置、每池两个副本及同一张 GPU 预留 |
| `deploy/Dockerfile` | 一个构建定义，生成 CPU 与 cu126 两种应用镜像 |
| `deploy/requirements-cu126.txt` | 从根声明与原锁共同版本约束生成的带哈希 GPU 锁，不改变本机默认依赖 |
| `deploy/compose.env.example` | 服务器配置示例，复制为根目录 `.env.compose` |
| `scripts/deploy-server.sh [远程分支]` | GPU 服务器拉取指定分支（默认 main）、安全切换及快进、构建镜像、更新容器并等待就绪，复用现有配置及项目名 |
| `deploy/mediamtx.compose.yml` | 内部控制接口、RTSP TCP、指向 API 服务的发布鉴权回调 |
| `.dockerignore` | 排除权重、压缩包、媒体、样本、虚拟环境、Git 和本机环境文件 |

所有命令在仓库根目录运行，固定项目名 `vision-shield`：

```bash
# CPU
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml up -d --build
# GPU
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.gpu.yaml up -d --build
```

切换模式前使用当前模式的同一文件组合执行 `down`，保留数据；不得以删卷作为切换步骤。镜像构建和拉取可以联网，但模型由本地目录提供；运行容器直接调用已安装的 Python，启动时不执行依赖同步或模型下载。

应用镜像名固定为 `vision-shield:cpu` 与 `vision-shield:cu126`，Dockerfile 构建参数 `TORCH_BACKEND` 仅接受 cpu/cu126；API/reconciler 始终使用前者。基础镜像和依赖选择见 [研究记录](../research.md)，CPU 按根 uv.lock 安装，GPU 按生成锁一次安装，再无依赖安装同一应用包。

`compose.yaml` 为四个应用服务显式设置 `build.args.TORCH_BACKEND: cpu`。`compose.gpu.yaml` 必须同时将 `file-worker`、`stream-worker` 的 `image` 覆盖为 `vision-shield:cu126`，并将各自的 `build.args.TORCH_BACKEND` 覆盖为 `cu126`；仅修改镜像名不会切换 Dockerfile 构建分支。合并后 API/reconciler 的镜像和构建参数仍为 CPU。

## 配置输入

| 名称 | 默认/要求 | 使用位置 |
| --- | --- | --- |
| `MODEL_DIR` | 必填宿主绝对目录；必须已经存在 | worker 只读挂载 `/models`，禁止缺目录时隐式创建空目录 |
| `MEDIA_DIR` | 必填宿主绝对目录；运行用户须可写 | API、两个 worker 统一挂载 `/data` |
| `REDIS_URL` | 必填带凭据的现有 Redis 地址；部署示例使用 `192.168.1.104:6378`、数据库 `6` | 四个应用服务统一使用，密码须做 URL 编码 |
| `BIND_ADDRESS` | 部署示例为 `192.168.1.104`；未配置时回落 `127.0.0.1` | 宿主 API 与 RTSP 端口绑定 |
| `API_PORT` / `RTSP_PORT` | 默认 `8000` / `8554` | 宿主发布端口 |
| `RTSP_PUBLIC_BASE_URL` | 必填 `rtsp://<客户端可达地址>:<RTSP_PORT>` | 仅用于既有响应中的 `playback_url` |
| `SOURCE_ALLOWLIST` | 默认 `[]`，远程媒体须填授权主机/IP/CIDR 的 JSON 数组 | 保留现有来源校验 |
| `GPU_DEVICE_ID` | GPU 模式必填一个卡号或 UUID；不允许空值、`all` 或列表 | 两类 worker 预留同一设备 |

两种应用镜像固定 `WORKDIR=/app`，并在镜像 ENV 设置 `FACE_MODEL_PATH=/models/yunet_face.onnx`、`PLATE_MODEL_PATH=/models/hyperlpr_plate.onnx`、`EGOBLUR_FACE_MODEL_PATH=/models/face.jit`、`EGOBLUR_PLATE_MODEL_PATH=/models/license_plate.jit`，使独立预检与 Compose 使用同一绝对路径。Compose 从 `.env.compose` 读取 `REDIS_URL`，示例为 `redis://:URL编码后的密码@192.168.1.104:6378/6`。`MEDIA_ROOT=/data`、`MEDIAMTX_CONTROL_URL=http://mediamtx:9997` 保持固定。默认 `DETECTION_BACKEND=quality`、`DETECTION_MAX_SIDE=1280`、`DETECTION_BOX_PADDING=0.15`，继承既有限流和缓冲参数。

新增通用应用配置仅 `RTSP_PUBLISH_BASE_URL`：可选 RTSP 基地址，复用现有 URL 校验约束；未设置时回落 `RTSP_PUBLIC_BASE_URL`，保持本机行为。Compose 显式设置为 `rtsp://mediamtx:8554`，只改变 worker 推流，不改变业务响应结构。

CPU 基础配置对应用角色显式设置 `INFERENCE_DEVICE=cpu`；GPU 覆盖只把两个 worker 改为 `cuda`。API/reconciler 使用 CPU 镜像，不分配 GPU 或挂载模型。`.env.compose` 不覆盖 Windows 原生 `.env`，镜像也不包含它。

## GPU 选择

使用 `deploy.resources.reservations.devices`，设置 `driver: nvidia`、`capabilities: [gpu]`、单元素 `device_ids`；不同时设置 `count`，不使用需要 Compose 2.30.0 的 `services.gpus`。卡号/UUID 由部署人员通过 `nvidia-smi` 确认；目标卡在容器内作为唯一可见设备供既有 `cuda` 使用。

GPU 覆盖将相同 `GPU_DEVICE_ID` 同时注入两个 worker 的环境。probe 在该部署变量存在时校验它是单个非负卡号或完整 GPU UUID，并确认仅一张 CUDA 设备可见；拒绝空值、all 与列表，不增加新的 Settings 设备选项。原生 Windows 未设置该部署变量时保留既有设备选择规则。

GPU Compose 的文件池和流池各启动两个 worker 副本，每个副本同时处理一个任务，共用所选卡。后端为 quality_yolo：YOLO 人脸与 YuNet/HyperLPR 使用 ONNX Runtime CUDA，EgoBlur 车牌使用 PyTorch CUDA；CUDA 离线视频使用 NVENC，解码、图像预处理、后处理、跟踪和遮挡仍使用 CPU，实时流仍用 libx264。选择不提供独占保证，不做多卡调度，不据单任务推理成功承诺并发显存容量。无效设备、CUDA 不可用或算子失败均不得改用 CPU 或其他卡。

## 角色、存储和网络

| 服务 | 入口 | 依赖与数据 |
| --- | --- | --- |
| `mediamtx` | 官方 MediaMTX，载入容器专用配置 | RTSP TCP 对外；控制 9997 仅容器网络；无录像 |
| `api` | `python -m uvicorn vision_shield.main:app --host 0.0.0.0 --port 8000 --no-access-log --limit-concurrency 32` | MediaMTX started；媒体目录可读写 |
| `reconciler` | `python -m vision_shield.jobs.reconcile` | MediaMTX started；无需模型或媒体挂载 |
| `file-worker` | `python -m vision_shield.jobs.worker files` | 模型只读、媒体可读写 |
| `stream-worker` | `python -m vision_shield.jobs.worker streams` | API/MediaMTX started；模型只读、媒体可读写 |

API healthcheck 使用既有 `/health/ready`；任何 worker 均不得等待 API healthy，避免 ready 依赖 worker 的循环。MediaMTX 不等待 API ready；其发布回调改为 `http://api:8000/internal/media/publish-check`，保留 owner/generation 鉴权、禁止发布者覆盖及禁录像。冷启动时的暂时连接失败沿用已有恢复流程。

应用使用非 root 用户 `10001:10001`，部署人员准备媒体目录权限。Redis 复用已部署实例，不定义 Redis 服务或数据卷，也不操作其生命周期；部署方核对端口可达性、认证、持久化和淘汰策略。仅 API 和 RTSP 端口按指定接口发布，控制接口不发布。沿用既有受信网络接入边界，不新增用户鉴权、TLS 网关或管理界面。

## 就绪、失败与停止

1. worker 通过既有 `media.runner → media.child` 执行内部 `kind=probe`，不增加对外任务种类。子进程先核验模型清单/读取权限/哈希，再复用检测工厂实际推理固定 320×320 合成帧，并执行非空框 NMS；CUDA 模式必须验证实际模型和 NMS 设备。
2. 预检总超时 120 秒，失败返回非零并给出文件/设备/阶段诊断；不输出完整配置。成功事件和退出码均正常、子进程已退出后，才启动 Receiver 并发布心跳。
3. 模型目录持续只读挂载；启动和处理不下载、不换模型。干净首次启动预检失败时 ready 不成功；运行中崩溃后，旧心跳有默认 20 秒租约检测窗口，过期后 ready 必须失败。正常退出主动撤心跳，不清空整个池。ready 不保证每个未来任务成功、检测准确率或显存容量。
4. worker 处理 Linux SIGTERM/SIGINT，通知 Receiver 停止接单，并通过同进程停止事件触发现有任务控制和媒体回收；完成后撤心跳、关 broker。不能仅设置 Receiver finish_event，因为常驻流可能占满其并发信号量。
5. 停机不写入用户的 `desired_state=cancelled/stopped`。execute 须区分部署停止：媒体回收后停止续租，未完成文件跳过正常 failed/processing_failed 的 finish 分支，交给既有失租约 worker_lost 流程。已提交成功结果保留；流在 run 意图下允许重启后恢复，用户显式停止的流不复活。
6. 应用容器 `init: true`，停止宽限 60 秒；保留既有媒体停止超时 10 秒，随后强杀并等待直接子进程。Linux 在模型加载前监视父管道 EOF，Windows 保留避免数值 DLL 初始化死锁的既有顺序。reconciler 取消主协程后走既有 finally。

预检不在 API 事件循环执行；应用处理仍由现有有界子进程完成，不新增常驻模型服务。完整验收见 [quickstart.md](../quickstart.md)。

另提供内部运维入口 `python -m vision_shield.detection.preflight`，读取既有 Settings 并复用 probe 的模型核验与实际推理函数，用于无网络独立容器验证；不连接 Redis/MediaMTX，不创建业务任务或写媒体目录。worker 启动仍由 runner 监督 probe 并限制总超时，不以启动脚本绕过失败门槛。
