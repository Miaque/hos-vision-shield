# Vision Shield

Python/FastAPI + Taskiq 媒体脱敏项目，支持图片、视频与 RTSP/RTMP 实时流中的人脸和车牌深灰色遮挡或马赛克，不做识别或音频脱敏。
支持直接在 Windows 本机运行；服务器另提供 CPU／指定单卡 GPU 的 Docker Compose 部署。
项目以 `main` 分支作为主干。

当前进度见 [任务清单](specs/001-media-redaction/tasks.md) 与 [验证记录](specs/001-media-redaction/validation/gates.md)。
已接通上传/URL文件任务、结果下载、取消、默认 RTSP 输出、可选 RTMP 输出及断流/worker重启恢复。API与任务控制使用异步I/O；模型推理、编解码在受控子进程中执行。

## 安装 Python 依赖

```powershell
# 在仓库根目录执行
uv sync --frozen
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\scripts\download-cpu-models.ps1
```

解释器使用Python3.12，依赖由uv.lock锁定。Windows x64 安装 PyTorch 2.14.1 / torchvision 0.29.1 的 CUDA 12.6 构建，支持本机 GTX 1080；其他平台继续使用 CPU 构建。CUDA 构建也可执行 CPU 推理，不需要单独安装 CUDA Toolkit，但 GPU 运行需要兼容的 NVIDIA 驱动。
默认`DETECTION_BACKEND=quality`：EgoBlur全图双类检测加YuNet/HyperLPR补充，兼顾遮挡效果和速度。最长边不超过1920的输入只做全图检测，更大输入使用1920像素重叠分块；允许极小目标漏遮。四个模型都必须可用。模型可以从已准备机器离线复制整个 `models/`（含两份 manifest 和许可证），启动及推理不下载权重；上面的下载脚本仅供联网准备模型时使用。已有.env需同步后端及四条模型路径。

GPU Compose 选用 `DETECTION_BACKEND=quality_yolo`，用轻量 YOLO 人脸候选替换 EgoBlur 人脸，保留 EgoBlur 车牌、YuNet/HyperLPR、现有确认和跟踪规则。本机可在 CUDA 环境显式选择该后端。新权重随版本包交付，来源、哈希、GPL-3.0 许可与导出方法见[模型说明](src/vision_shield/detection/assets/README.md)。原 `quality` 和 CPU 默认值保留；该后端的目标服务器吞吐须单独实测，不能由本地片段速度推定双路达标。

`quality_yolo`在计算能力主版本至少8的GPU上使用FP16车牌模型参数和自动混合精度，沿用原Gen1完整执行图，保留FP32归一化、FP32输出坐标/分数及最多1000个ROI候选，撤回服务器实测没有收益的512候选裁减；较旧GPU及其他后端保持原路径。ONNX精度设置不变。精度变化可能改变临界分数及框，效果和吞吐均须在目标服务器验证，见[时序优化记录](specs/002-compose-cpu-gpu/temporal-optimization.md)。

CUDA补充车牌检测恢复原有视图逐张推理，撤回服务器实测没有收益的视图合批。保留全部检测区域、letterbox、阈值、跨模型确认与逐帧跟踪；NVDEC、GPU预处理/推理/遮罩和NVENC路径保持。完整4K双路两次基线验收含初始化约13.0–13.6 FPS，尚未达到每路15 FPS。

GPU离线任务查询的 `Server-Timing` 额外提供主/补预处理、人脸/车牌前向调用、后处理、局部/跟踪复核、拷贝及显式同步的累计主机耗时。计时不增加GPU同步；嵌套、并行项目不能相加，ORT的CPU输出前向调用包含内部等待与下载，不能当作纯GPU内核耗时。详见[分项测量说明](specs/002-compose-cpu-gpu/temporal-optimization.md)。

环境标志 `DEBUG` 默认为 `false`（生产）。优化流程开始前，在本机 `.env` 或容器 `.env.compose` 设置 `DEBUG=true`（调试），重启 API 和全部 worker；容器需重新创建以应用环境变量。生产模式不启动资源采样，不收集详细模型/媒体耗时和逐帧输出证据或计算画面哈希，不返回诊断响应头；诊断接口返回 404，profiling 请求返回 422。任务进度、推流健康判断、模型预检和必要错误日志保留。切回 `DEBUG=false` 后同样需要重启，不会删除已有诊断文件。

详细连接历史、异常栈、FFmpeg 协议日志捕获及连接结束诊断日志同样仅在 `DEBUG=true` 时收集或输出；生产仍保留安全错误码、重连及输出健康控制。`scripts/profile-local-clip.py` 不再强制打开调试，需在其进程环境显式设置 `DEBUG=true`；未开启时在读取原片或创建诊断目录前拒绝执行。P0 人脸长尾修复、P1 发布节奏和必要的编码时钟/背压保护属于正常功能，不依赖 DEBUG。

实时 DEBUG 诊断通过现有 `GET /v1/tasks/{id}/diagnostics` 的 `live` 字段提供当前 worker 的源码/权重 SHA256、设备/精度及生效参数。最多保留128条处理/渲染操作和64条超过100ms的主机耗时范围，每秒至多更新一次，使用独立且小于192 KiB的进程事件；超过500ms的范围附结束后的调用位置，不是阻塞当时的原生调用栈。不新增 GPU 同步或设备 profiler。设置 `STREAM_DIAGNOSTICS=false` 可关闭这部分观测，`DEBUG=false` 始终关闭；环境变量变更仍按现有方式重启生效。`source_pts` 和 `frame_sequence` 关联本次连接的源帧，前视操作的 `trigger_source_pts` 仅表示触发帧；嵌套或并行模型耗时不能相加。`oldest_input_age_seconds` 从读包开始计龄，包含入队前背压，不能当作纯队列内等待。mux完成时间只证明复用器接受数据，实际网络发包需要独立接收端或发包探针验证。

GPU 的 `3/flow` 文件及实时路径采用最多 0.5 秒媒体时间、16 帧的车牌前视补遮。实时输入暂停时额外等待队列最多保留约 0.75 秒，再按不完整窗口检查并输出；模型、编码及传输时间另计。DEBUG 任务响应的 `stats.stream.lookahead_frames` 表示前视队列，`queue_frames` 仍表示输入队列，接收至输出时延包含前视等待；旧 attempt 可不含该字段。

实时前视队列完成复核和遮罩后，在编码后、视频包写出前按媒体间隔控制发布节奏，单帧额外等待最多 0.5 秒；慢处理后从当前发布时钟继续，不突发追赶历史帧。`STREAM_OUTPUT_PACING=false` 可关闭节奏控制（Compose 修改后需重新创建应用容器）。文件处理不等待墙钟，不更改帧或音频时间戳；节奏控制仍与处理共用循环，模型再次长时间阻塞时不能保证连续输出，也不把 0.5 秒前视窗口称作总延迟。DEBUG 的 `task-output-pacing` 单独报告发布等待，`task-encode` 扣除这段等待。

调试环境下，主车牌设备诊断仍须显式启用。上传视频时可加 `X-Detection-Profile: 1`，在第64、424、768个合格全图调用采样；`GET /v1/tasks/{id}/diagnostics`读取摘要，添加`?sample=64`等固定索引可下载原始trace。每份最多32 MiB、每任务最多三份，短视频或未进入采样位置返回partial。该模式要求CUDA模型，采样会扰动耗时，其FPS不作为优化验收。

局部人脸长尾诊断可用 `X-Detection-Profile: local-face`：只在独立视频诊断进程追踪实际 NumPy 局部人脸 ORT 调用，保留最先遇到的至多三次超过 500ms 的调用 trace，下载槽仍为 64/424/768；未捕获满三次会明确返回 partial。`host` 仅收集墙钟计时，不打开设备 profiler。两者的调用记录包含帧、复核请求关联号、目标框、ROI/缩放/模型输入尺寸、会话/provider、首次局部 shape、搬运/NumPy/resize/准备/ORT/后处理阶段，摘要报告 P50/P95/P99/max 及 >500ms/>1000ms 次数。请求关联号不是持久轨迹 ID；本项目当前没有该 ID。墙钟搬运计时包含同步，不能代替 trace 中的 GPU 计算、复制、分配与等待。诊断不更改模型、预处理、复核或遮罩决策。

### 本机 GTX 1080

先执行 `uv sync --frozen --extra cuda`，在 `.env` 设置 `INFERENCE_DEVICE=cuda`，然后运行原有 `scripts/start-application.ps1`。EgoBlur 使用 PyTorch CUDA，YuNet/HyperLPR 使用 ONNX Runtime CUDA，模型输入预处理也在GPU完成。兼容的离线H.264视频采用NVDEC解码、GPU颜色转换/遮罩及NVENC编码；跟踪仍下载灰度图，局部确认按需下载裁剪。解码格式边界及实际编解码器响应头见[诊断说明](specs/002-compose-cpu-gpu/diagnostics.md#任务处理阶段计时)。CUDA 推理不可用时启动失败，不静默降级。要使用 CPU，只需改为 `INFERENCE_DEVICE=cpu` 后重启应用。未设置该字段仍默认 CPU；轻量 `DETECTION_BACKEND=cpu` 不接受 cuda 设备。

首次安装或驱动变更后，先执行真实模型验证：

```powershell
$env:RUN_GPU_TESTS='1'
uv run --frozen pytest tests/integration/test_gpu_model_probe.py -q
Remove-Item Env:RUN_GPU_TESTS
```

验证使用本地四个模型和 `samples/robot-quality` 中已标注样例，检查模型校验和、真实 GPU 推理和遮挡区域。GPU 不可用或推理失败会报错，不回退 CPU。更新依赖后须重启已有进程；本机 GPU 验证不代表服务器 x86_64 Compose 部署已通过。

本机已完成真实模型、图片/视频子进程及同图 CPU/GPU 对比，边界与结果见 [GTX 1080 验证记录](specs/002-compose-cpu-gpu/local-gpu.md)。
GPU 模式下，两路独立的全图检测同时执行，随后按原顺序进行局部复核。此前 CPU 补充检测的调度基线见 [GPU 同效果优化记录](specs/002-compose-cpu-gpu/gpu-performance.md)，四模型 CUDA 迁移与部署见 [CUDA 补充检测](specs/002-compose-cpu-gpu/cuda-detail.md)。

## 服务器 Docker Compose

目标为 Ubuntu 24.04 x86_64，沿用服务器现有 Docker 和 `docker-compose v2.29.2`；不在 Windows 本机安装 Docker。默认 CPU，GPU 模式指定一张卡供文件与流 worker 共用。GPU 失败不换卡或回退 CPU；GPU Compose 使用上述 `quality_yolo`，CPU 保留 `quality`。GPU Compose 的文件池和流池各启动2个worker副本，每个副本同时处理1个任务，支持同池双路同时执行；副本共用所选显卡，并发吞吐须另行实测。

按 [服务器部署验收指南](specs/002-compose-cpu-gpu/quickstart.md)准备离线模型目录、媒体目录权限和 `.env.compose`，然后在服务器仓库根目录选择一种方式：

```bash
# CPU
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml up -d --build
# GPU：先在 .env.compose 填写 GPU_DEVICE_ID，推荐完整 GPU UUID
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.gpu.yaml up -d --build
```

日常更新可直接使用 Git 工作目录；从源码包切换可克隆新目录，或在原目录接入 Git，步骤见[Git 更新与重新构建](specs/002-compose-cpu-gpu/quickstart.md#7-git-更新与重新构建)。GPU 服务器在仓库根目录执行：

```bash
bash scripts/deploy-server.sh
# 实验分支：必须先同步支持分支参数的新版脚本
bash scripts/deploy-server.sh feature/gpu-throughput-optimization
# 固定版本部署及回退：推荐使用已推送的不可变 tag
bash scripts/deploy-server.sh --tag exp-20261023-b-deploy-02
```

代码须先从开发机提交并推送到指定远程分支或 tag。推荐 `--tag 标签`：拉取准确 tag 并以 detached HEAD 部署，不移动本地分支；已有同名 tag 指向不同版本时拒绝覆盖。不传参数会从当前分支或 detached HEAD 安全切换到 `main`，并部署最新的 `origin/main`；显式传分支时安全切换并快进到该分支，保留其他分支。首次使用须先同步新版脚本，旧版可先显式传 `main` 完成更新。脚本固定使用 GPU 文件组合、项目名 `vision-shield` 和现有 `.env.compose`，依次拉取代码、检查配置、构建、更新容器并等待就绪（最多 240 秒），成功后输出版本标识和提交号。未提交的跟踪文件修改、目标分支分叉或本地额外提交均会停止部署；未跟踪文件与目标版本冲突时也会停止，不覆盖本地文件。脚本不备份、不强制重置、不自动回滚，网络或构建失败时不会进入容器更新步骤。

日常更新不先执行 `down`，构建成功后才替换容器。重建期间旧容器继续服务，替换时会短暂中断；正在执行的文件任务不会续跑，流任务会尝试恢复。当前 Dockerfile 已分离依赖层与应用层，构建缓存可用且依赖未变时，源码修改可复用已安装的 PyTorch/CUDA 依赖；不要在日常更新中使用 `--no-cache`。

`compose.yaml` 包含 MediaMTX、API、reconciler、notifier 和两个 worker，可显式追加 `compose.replay.yaml` 接入独立视频回放容器。应用服务直接复用现有带密码 Redis。在 `.env.compose` 设置 `REDIS_URL='redis://:URL编码后的密码@192.168.1.104:6378/6'`，并将文件权限设为 `600`；应用服务统一读取该地址。宿主发布端口须能从 Docker 网络访问。现有 Redis 的持久化及淘汰策略由部署方核对，不修改共享实例配置。

仅 API/RTSP 端口按配置发布；部署地址和端口由 `.env.compose` 的 `BIND_ADDRESS`、`API_PORT`、`RTSP_PORT`、`RTSP_PUBLIC_BASE_URL` 指定，模型及媒体目录由 `MODEL_DIR`、`MEDIA_DIR` 指定。容器内部推流与控制沿用服务名，远程输入还须配置来源允许列表。媒体目录保留在宿主机，切换前按原文件组合执行 `down`，不使用 `down -v`。

镜像构建可以联网安装依赖，`models/` 中的模型必须事先离线提供并只读挂载；quality_yolo 的 YOLO 权重随应用包交付，启动及推理不下载模型。worker 先在有 120 秒总超时的子进程中核验清单/哈希、实际推理及 NMS，成功后才接单并发布心跳；故障后的旧心跳最多有默认 20 秒租约窗口。`RTSP_PUBLISH_BASE_URL` 仅控制容器内部推流，对外仍返回公开播放地址。

GPU 镜像按 [研究记录中的生成命令](specs/002-compose-cpu-gpu/research.md)使用 `deploy/requirements-cu126.txt`，共同依赖沿用根锁版本；根声明变更时重新生成，禁止手工改 pin。CPU 使用根 uv.lock，Windows 安装方式保持不变。初次交付证据与未运行项见 [实施记录](specs/002-compose-cpu-gpu/validation/gates.md)，后续服务器分阶段实测见 [时序优化记录](specs/002-compose-cpu-gpu/temporal-optimization.md)；这些记录不代表完整部署验收已通过。

### 三路录像实时回放

`video-replay` 必须显式启动，上述普通 Compose 命令及 `deploy-server.sh` 不启动或更新该服务，也不要求准备回放录像。单个容器内运行 MediaMTX 和三个 FFmpeg 回放进程，不使用 GPU。三段录像按原时间节奏无限循环，视频及音频直接复制码流；回放源允许匿名读取，只开放 `replay/rfm-01`、`replay/rfm-02` 和 `replay/rfm-03` 三条路径。

首次启动回放前，将三段未脱敏的录像放到服务器同一目录，分别命名为 `rfm-01.mp4`、`rfm-02.mp4`、`rfm-03.mp4`，并在 `.env.compose` 设置该绝对目录：

```dotenv
REPLAY_MEDIA_DIR=/home/meiya/vision-shield-replay/media
```

前两段对应本地 `data/recordings/rfm-live-20261027-150312/rfm-original-preview.mp4` 和 `data/recordings/rfm-live-20261027-150915/rfm-original-preview.mp4`；第三段使用同一挂载目录中的 `rfm-03.mp4`。录像不提交 Git、不打进镜像，只读挂载；未指定目录时使用仓库下的 `data/replay/`。目录缺失会拒绝启动，视频缺失或任一路不可读取会令回放容器的健康检查失败。

实时任务使用以下输入地址，按服务器的 `BIND_ADDRESS` 替换 IP：

```text
rtsp://192.168.1.104:8555/replay/rfm-01
rtsp://192.168.1.104:8555/replay/rfm-02
rtsp://192.168.1.104:8555/replay/rfm-03
```

在 `SOURCE_ALLOWLIST` 原列表中保留已有条目并追加服务器 IP `192.168.1.104`，随后按原部署方式重新创建应用容器。测试任务显式使用 `X-Detection-Interval: 3`、`X-Tracking-Mode: flow`，以保持文件复测的调度条件。循环拼接点会发生画面跳变，分析跟踪失败与卡顿时需单独标记。

在仓库根目录显式启动或更新回放，仅操作 `video-replay`，无需添加 GPU 配置：

```bash
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.replay.yaml up -d --build --wait video-replay
```

从两路升级为三路时，同步 `deploy/mediamtx.replay.yml` 和 `compose.replay.yaml` 后执行上述命令，健康检查变更会触发回放容器重建，已有回放连接会短暂中断。新增回放源不会自动替换脱敏任务：先确认第三路可读取，再通过 `PATCH /v1/tasks/{id}` 提交 `{"desired_state":"stopped"}`，等待旧任务进入 `stopped`，最后通过 `POST /v1/stream-tasks` 使用第三路地址创建新任务。默认脱敏播放地址随新任务 ID 改变。

检查和临时停止回放使用同一文件组合：

```bash
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.replay.yaml logs --tail 50 video-replay
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.replay.yaml stop video-replay
```

不再需要回放时，执行上述停止命令及同参数的 `rm -f video-replay` 即可，不必删除仓库文件。已显式启动的回放会持续运行，普通部署不会自动停止它；仅使用主配置时可能提示该容器为 orphan，不要使用 `--remove-orphans` 清理仍需运行的回放。现有 `8554` 媒体服务与检测 worker 不依赖回放容器。

## 本地 Redis / MediaMTX

如果.env已指向运行中的Redis和MediaMTX，直接启动应用即可，不必再运行基础服务脚本。持久化检查与服务端配置步骤见[环境核对](specs/001-media-redaction/validation/environment.md)。

准备 Redis7.4.3 和 MediaMTX1.21.2。以下示例要求 `redis-server.exe` 位于 PATH，MediaMTX 位于被 Git 忽略的 `.tools/mediamtx/`；其他位置可直接传给脚本参数。在项目根目录运行，保持这个终端打开：

```powershell
.\scripts\start-infrastructure.ps1 `
  -RedisExecutable (Get-Command redis-server.exe).Source `
  -MediaMtxExecutable '.tools/mediamtx/mediamtx.exe'
```

退出时Ctrl+C只关闭该脚本创建的两个进程，不操作其他已存在的服务。
Redis数据保存在 `data/redis/`，日志与媒体服务运行文件位于 `data/`。
仅做启动和持久化配置自检，检查后自动回收进程：

```powershell
.\scripts\start-infrastructure.ps1 `
  -RedisExecutable (Get-Command redis-server.exe).Source `
  -MediaMtxExecutable '.tools/mediamtx/mediamtx.exe' -Check
```

Redis端口16379、RTSP端口8554、媒体控制端口9997，当前仅监听127.0.0.1。
Redis启用AOF always与noeviction；MediaMTX禁用录像和发布者覆盖。
媒体发布回调只接受当前任务owner/generation，接管前撤销旧RTSP会话；播放和业务接口没有用户鉴权。

## 启动应用

基础服务保持运行后，在第二个PowerShell终端启动：

```powershell
.\scripts\start-application.ps1
```

打开 http://127.0.0.1:8000/docs 。脚本启动API、协调器和两个独立任务池，各1个worker；每个worker同时处理1个任务。使用 `-StreamWorkers 2` 增加流处理槽位，文件池不受影响。保留终端，Ctrl+C停止本次创建的应用进程。附加 `-Check` 会在健康检查通过后关闭这些进程；基础服务须先启动。

## 调用接口

上传在文件完整接收后返回202及Location，之后由Taskiq处理：

```powershell
curl.exe -F 'kind=image' -F 'file=@samples/official-demo.jpg' http://127.0.0.1:8000/v1/file-tasks/uploads
# 视频同一端点，kind=video；仅MP4/H.264，可含AAC或无音轨
```

远程输入需先在 `.env` 配置授权来源，如 `SOURCE_ALLOWLIST=["192.168.1.50"]`，再重启应用进程；授权内网主机名也必须授权其解析的IP/CIDR。空数组拒绝所有远程输入，上传仍可用。

```powershell
$file = Invoke-RestMethod http://127.0.0.1:8000/v1/file-tasks -Method Post -ContentType application/json -Body '{"kind":"image","url":"http://192.168.1.50/image.jpg"}'
Invoke-RestMethod "http://127.0.0.1:8000/v1/tasks/$($file.id)"
# state=succeeded 后下载；结果未完成时返回409
Invoke-WebRequest "http://127.0.0.1:8000/v1/tasks/$($file.id)/result" -OutFile result.jpg
Invoke-RestMethod "http://127.0.0.1:8000/v1/tasks/$($file.id)" -Method Patch -ContentType application/json -Body '{"desired_state":"cancelled"}'

$stream = Invoke-RestMethod http://127.0.0.1:8000/v1/stream-tasks -Method Post -ContentType application/json -Body '{"url":"rtsp://192.168.1.50/live"}'
Invoke-RestMethod "http://127.0.0.1:8000/v1/tasks/$($stream.id)"
# ready=true 后使用响应中的playback_url播放（例如ffplay -rtsp_transport tcp <地址>）
Invoke-RestMethod "http://127.0.0.1:8000/v1/tasks/$($stream.id)" -Method Patch -ContentType application/json -Body '{"desired_state":"stopped"}'
```

`POST /v1/stream-tasks` 可选 `publish_url` 指定完整推流地址，协议由地址决定；可选 `playback_url` 指定接收平台的完整播放地址（HTTP/HTTPS/RTSP/RTMP）：

输入 `url` 支持 RTSP 和 RTMP，均须授权来源域名及其实际 IP。RTMP 输入保留原厂 URL、签名和 `tcUrl` 域名，通过同一逐帧检测、遮挡、编码和任务恢复链路处理，不另加转流服务。推流目标也须在允许列表中。`ready` 表示已有实际脱敏输出；外部固定播放地址仍须在接收平台验证，不能据此承诺浏览器播放。接入码力时应通过厂商接口的 `livetype=1` 获取 RTMP，不能将 FLV 地址替换协议当作 RTMP。

```json
{
  "url": "rtsp://camera.example/live?token=INPUT_TOKEN",
  "publish_url": "rtmp://media.example/app/camera?sign=PUBLISH_SIGN"
}
```

可选 `max_bitrate` 设置实时视频编码码率上限，单位为 bit/s，例如 `20000000` 表示 20 Mbps。
省略或传 `null` 时不设置码率上限；该参数只影响实时 NVENC，不影响离线文件编码。

- 不传或传 `null`：保持默认 MediaMTX RTSP 输出和 `playback_url`；完整 RTSP 地址也可用于直接推流。
- 指定地址：保留路径、查询参数和鉴权信息，推流到外部服务，不拼接 `/redacted/{task_id}` 或注入内部 owner 凭据。输入与推流目标都必须在 `SOURCE_ALLOWLIST` 中授权；worker 每次重连重新校验并固定到授权 IP，RTMP `tcUrl` 保留原域名。

实时流创建请求可指定 `max_bitrate`（正整数，单位 bit/s）和 `audio_enabled`（布尔值，默认 `true`）。
例如 `{"url":"rtmp://camera.example/live","max_bitrate":2000000,"audio_enabled":false}` 会限制视频编码码率并只输出脱敏视频。
码率限制沿用编码器的一秒 VBV 窗口，瞬时码率仍可波动；降低码率后须复核画质。
静音任务在服务端跳过音轨，沿用视频时间戳修复。有声任务在读包阶段恢复重复或回退的音视频时钟，再按恢复后的时间配对；AAC 保留压缩内容，G711 推 RTMP 时仍转为 AAC。
修复使用真实采样数、视频帧周期或已观测的有效间隔，并保留毫秒时基的小数精度，避免重复修复逐渐累积音画偏移；无法确定可靠间隔时仍失败关闭。调试输出继续分别记录原始视频 PTS 与修复后的时间线。
两项参数随任务冻结并保留到重连；相同幂等键不能改变码率或声音，省略 `audio_enabled` 与显式 `true` 的行为一致。文件任务不接受该声音开关。
- 流输入支持 H.264，音频可为 AAC、G711A、G711U 或无音轨。RTSP 保留音频编码；RTMP 使用 FLV，AAC 直接转封装、G711 转 AAC。文件视频仍只接受 AAC 或无音轨。
- `playback_url` 优先返回请求指定的播放地址；未指定时，外部推流沿用完整 `publish_url`（含其凭据和鉴权参数），默认输出仍使用公开 RTSP 地址。接收平台播放地址与推流地址不同时，应显式传入 `playback_url`。通知不携带这些地址。`ready`/health 证明本 worker 已写出数据，不证明接收平台可播放。同一幂等键不能更换推流或播放地址。
- 外部推流在本次媒体处理确认发布连接、编解码器和读取线程全部关闭，或子进程退出/超时强杀完成后确认停止；空 MediaMTX 会话列表不能证明外部推流已停止。租约过期后，在原发布者关闭确认前保持 `recovering`，不派发替代任务；相同发布者和代次的确认到达后，按停止意图结束任务，或允许新 worker 接管。旧版本任务缺少确认记录时同样保守处理，未开始执行的旧任务仍可正常调度。
- 外部平台没有远程踢流接口；worker 崩溃或退出确认写入失败时，任务可能持续处于 `recovering`，需在接收平台核实和处理旧发布连接。接收平台需限制同一路径重复发布，不应让多个任务同时使用同一地址。MediaMTX 固定一起部署，`/health/ready` 始终检查 Redis、worker 和 MediaMTX。

每次流连接结束后，worker 在 `INFO` 日志输出 `vision_shield_stream_end` 结构化摘要，携带 `task_id`、`run_id`、`attempt_id`（已有时）、代次、起止时间、持续秒数、`reason`、`phase` 和异常类型。Platform 的 FLV 结束摘要通过 `shield_task_id` 与此处 `task_id` 对应，可判断页面断流前是否出现输入 EOF、打开来源失败、解码失败或推流失败。

FFmpeg 异常额外保留 `ffmpeg_error` 和可用时的 `ffmpeg_log`，每字段最多 1024 字符，移除完整 URL、查询凭据、授权头和换行后才写入日志及诊断记录。源打开期间临时启用 PyAV 的错误回调，完成后恢复原日志级别，不打开持续的逐帧日志。PyAV 默认关闭日志会减少异常细节，依据见[官方日志说明](https://pyav.basswood.io/docs/stable/api/utils.html#logging)。没有底层协议详情时，通用 `InvalidDataError` 仍不能证明是鉴权、网络或媒体格式中的哪一项。

调试环境的 `GET /v1/tasks/{id}/diagnostics` 沿用最近 32 次连接记录，也包含这些安全错误字段；重建 worker 后的新故障才会产生新增证据，既有记录不会补全。该变更不增加回调、MQ 字段或公开任务响应字段，也不修改重连与停止规则。

PATCH返回202表示停止意图已保存；查询到cancelled/stopped才表示资源清理完成。已成功文件不能再取消（409）。流自动重连并保留停止意图，文件worker崩溃则标记worker_lost，不自动续跑。

`GET /v1/tasks` 查询所有文件和流的未结束任务，返回 `{"current": [...], "queued": [...]}`，没有任务时对应空数组：

- `current`：处理中、连接中、运行中、仍由 worker 持有或等待外部发布者退出确认的恢复中任务，以及等待取消/停止完成的任务。
- `queued`：等待执行的 `queued` 任务，以及没有 owner、不等待外部发布者退出确认、可重新调度的 `recovering` 任务；只包含仍希望运行的任务。

每项字段与 `GET /v1/tasks/{id}` 相同，不返回源地址、凭据及内部 owner 信息。两组分别按创建时间升序、ID 次序排列，数组长度即该组任务数；展示顺序不代表调度顺序，也不是 Redis 消息队列中的消息数量。查询读取持久状态，非原子快照，状态和 owner 的失效由协调器更新。已结束任务不列出，仍可按 ID 查询。

```powershell
$tasks = Invoke-RestMethod http://127.0.0.1:8000/v1/tasks
$tasks.current
$tasks.queued
```

调试环境下，`GET /v1/diagnostics/resources` 查看有效 worker 的 CPU、内存和绑定显卡资源快照。CPU 为 worker 的 Linux cgroup v2 口径，包含媒体子进程；GPU 为所选显卡整卡口径，缺失和陈旧值会明确标记。字段单位、部署步骤及后续优化验收见[资源观测说明](specs/002-compose-cpu-gpu/diagnostics.md)。

默认只监听本机。其他系统接入时，部署方需同时修改启动脚本的API监听、MediaMTX的RTSP监听与 `.env` 的 `RTSP_PUBLIC_BASE_URL` 为可达内网地址，并限制防火墙可达范围。MediaMTX控制API和Redis继续仅监听本机。

## 模块与运行边界

`api/`负责HTTP契约，`jobs/`负责Taskiq、持久状态和租约，`media/`负责输入、进程生命周期及音视频流水线，`detection/`输出遮挡坐标，`redaction.py`执行可选遮挡及有界的相邻帧补偿。

每个 worker 独占一个常驻媒体子进程，启动预检后保留同一模型，连续文件任务和实时流重连 attempt 复用模型及已初始化的 CUDA 上下文。每个任务或 attempt 重新建立连接、编解码器和跟踪历史，计时独立；切换源 URL 不重载模型。停止超时、读取线程未退出、模型运行时故障或模型配置变化会回收进程；同路径模型权重更新须重启 worker。性能诊断任务先回收常驻进程，再使用独立冷启动进程，保持原有测量口径。

实时流运行中的模型执行故障、CUDA 设备不可用及原生外部库故障会丢弃旧进程，保留原任务的运行意图，沿用 1–30 秒退避在新进程中预检、加载并重试；重新预热时的运行时故障同样可重试。普通源断线继续保留健康模型；无效输入、配置和模型校验失败仍终止任务，文件任务不自动重试。停止请求和租约控制仍优先于恢复。该改动减少重新加载的机会，不改变稳定运行时的逐帧推理算法。

文件和流仍使用各自队列，每个 worker 同时处理一个任务；领取后在当前任务结束前不再预取下一条消息，避免忙碌 worker 占住其他空闲 worker 可以处理的任务。GPU Compose 的两个流 worker 与两个文件 worker 保留四个独立执行槽位。常驻模型在空闲时仍占用内存/显存；实际四路吞吐取决于共享 GPU 及媒体负载。

默认每池最多100个未结束任务、每队列100条消息。实时流只缓存2个待处理画面，过载取最新帧；文件逐帧处理。AAC缓存超过16秒或4MiB会重建流会话，不无限堆积；该上限保留自EgoBlur CPU校准，并不是延迟保证。

## 巡逻机器人素材

面向运动模糊和超宽视角，默认组合在最长边 `DETECTION_MAX_SIDE=1280`（320–1920）的全图运行EgoBlur，再以YuNet/HyperLPR做全图补充。2026-10-21按用户接受极小目标漏遮的要求，补充分块由640改为1920源像素：最长边不超过1920时只运行一个全图视图，3840×1080使用全图加3个重叠分块，每个视图仍按`DETECTION_MAX_SIDE`缩放。第四轮停止对短边不足20源像素的低分人脸（EgoBlur低于0.3、YuNet低于0.6）放大复核；较高分人脸和车牌仍按原规则处理。不保证20源像素以下目标的覆盖，也不承诺所有更大目标都检出。人脸候选门槛分别为0.3/0.6，车牌为0.1/0.25；EgoBlur得分至少0.7的结果保留，其余候选须得到另一模型同类重叠框支持，或通过局部复核；弱车牌须有另一模型得分至少0.7的全图结果佐证，否则转局部复核。复核取候选周边3倍最长边、至少96像素区域，调整到最长边320后交给另一模型；同类框IoU至少0.2。全图人脸佐证至少0.3，否则转局部复核。短边至少128源像素的大人脸候选需至少一路高分结果：弱EgoBlur须有YuNet至少0.9佐证，低于0.9的YuNet须有EgoBlur至少0.7佐证；全图与局部采用相同分数要求。YuNet自身至少0.9时允许EgoBlur全图0.3或局部0.1佐证。短边20至128源像素的人脸保留原有低阈值局部复核，以保护侧脸；不会单独从低分人脸新增遮挡。该策略减少单模型误遮，也可能丢失无法确认的弱目标。车牌网络按其320输入约定做letterbox。`DETECTION_BOX_PADDING=0.15` 表示EgoBlur每侧15%、YuNet人脸每侧15%，HyperLPR车牌水平每侧30%、垂直15%。不做OCR、锐化或生成式修复。

`DETECTION_BACKEND=cpu`保留上一轮轻量方案（YuNet阈值0.3），无需加载EgoBlur；它在固定样本中有远处车牌漏遮，不作为当前效果验收结论。当前组合在CPU上更慢，接口仍由媒体子进程异步执行，不承诺实时帧率。新效果对照见[效果优先验证](specs/001-media-redaction/validation/quality-optimization.md)。 局部复核只运行待确认类别，Torch每个媒体子进程采用4线程，OpenCV仍为2线程；不改变检测范围、阈值或视频逐帧处理。分项计时与效果一致性验证见[同效果CPU推理优化](specs/001-media-redaction/validation/inference-performance.md)。

视频和流对已确认目标进行短时运动匹配，最多保留128个历史目标；匹配成功可补偿500毫秒内的漏检，匹配失败会尝试当前帧局部复核，仍未确认则只保留200毫秒旧框。只有真实检测能刷新有效期。尺寸变化、时钟倒退、较长跳帧或缩略图明显变化会清空历史，每次流重连重新建立状态。没有检测历史的目标不能凭空补出，快速转向或大幅变形仍可能漏遮。

实时任务默认使用路线 A：每三帧全量检测，间隔帧按光流、当前帧人脸检测和车牌候选触发必要复核；解码、遮挡和输出仍逐帧执行，不丢帧。无需额外请求头。显式 `X-Detection-Interval: 1` 使用逐帧检测，默认配合 `template`；`flow` 只能配合间隔 3。缺少调度字段的旧实时任务在下次 attempt 使用路线 A 默认值；新任务会保存完整调度参数。文件任务的默认调度保持原有行为。

CPU 文件视频使用 libx264 slow / CRF 23，CUDA 文件视频使用 h264_nvenc p6/hq/VBR CQ19；源视频码率可用时，以其115%设置两秒VBV缓冲，尽量控制体积。这不是文件大小硬上限，也不保证无画质损失；直播编码设置保持不变。修复范围和验证边界见[编码说明](specs/002-compose-cpu-gpu/diagnostics.md#任务处理阶段计时)。

检测仍可能漏检和误遮，需要按实际机器人素材复核，不承诺模糊素材零漏检。降低DETECTION_MAX_SIDE会影响人脸细节，车牌网络输入及分块数也会影响耗时。与原EgoBlur 1280基线的对照入口：

```powershell
uv run python scripts/evaluate-robot-media.py samples/demo --output data/validation/cpu-evaluated
```

将授权素材放在 `samples/demo/`，或将命令中的该参数替换为实际素材目录。结果包括每张图片、每段视频的前三个连续帧的对照图、脱敏图、耗时和源文件哈希。比较工具还需要本机保留的两个EgoBlur模型。它不是完整视频或准确率验收，原始素材不覆盖、不向外部服务上传。CPU优化与限制见[验证记录](specs/001-media-redaction/validation/cpu-optimization.md)。

调整.env后重启API、协调器和worker；现有任务接口保持不变。

原始文件、完成结果和失败半成品均保留于 `data/<任务ID>/`，不会自动删除；只有成功结果能下载。视频流不录像。日志在 `data/application/`，按业务task ID定位，不输出源URL和密码。

## 检测模型


模型来源：OpenCV Zoo及HyperLPR作者仓库固定版本记录在 `models/cpu-manifest.json`，EgoBlur固定版本记录在 `models/manifest.json`。下载、许可和切换方法见[模型说明](models/README.md)。效果优先组合中任一模型文件缺失或推理失败都会使处理失败，不返回部分模型结果或透传原始画面。

## 运行验证

```powershell
$env:TEST_REDIS_SERVER=(Get-Command redis-server.exe).Source
$env:TEST_MEDIAMTX_BIN=(Resolve-Path .tools/mediamtx/mediamtx.exe).Path
uv run pytest -q
```

测试启动独立临时端口和数据目录，不清空现有Redis。Redis、FFmpeg与ffprobe需在PATH中，或按测试配置指定可执行文件。
真实媒体样本和权重不提交Git；自动生成的图片和视频仅用于媒体机制验证。效果回归另需本机`samples/robot-quality/`中清单对应的授权样本，其哈希和已核对的小区域记录在`tests/fixtures/quality-cases.json`；缺失时明确失败，不跳过。原始demo中的image-2、7、13、17、19按该清单命名保留，rtsp-green.png来自已保存的source-first截图；另三张PNG是demo视频9第0帧、视频10第42帧和视频11第17帧。video-04-frame-24.png来自`tests/test_assets/22-25-32违规占道.mp4`开头一秒连续片段的第24帧（从0计数），另加入同片段的第09、10、16、19帧，覆盖五张夜间轮胎误检帧与同帧车牌保护回归。
实际摄像头接入时在 `.env` 中将 `SOURCE_ALLOWLIST` 设置为授权主机/IP/CIDR的JSON数组；默认空数组拒绝远程来源。

当前没有用户鉴权或自动文件删除，原音频不脱敏；检测效果、完整故障恢复与容量以各阶段实测为准。

### 可靠任务交接

创建三个入口均支持可选 `Idempotency-Key`（1—128 个可打印 ASCII 字符，不能全为空白）。同键同有效参数与上传内容返回同一任务的当前状态，终态也可恢复；同键不同输入返回 409。上传指纹使用文件 SHA256，不包含随机目录。旧调用不传新请求头时保持原接纳方式。

通知只通过 RabbitMQ 交付，不再支持 HTTP 回调。创建任务时传 `X-Notification-Target` 和 `Idempotency-Key`；前者必须在 Shield 的 `NOTIFICATION_TARGETS` 中登记。目标为最多 64 字符的小写字母、数字、连字符及点分段标识，不写死环境名称。普通手工任务可以不订阅通知。

#### 默认 vhost 与环境配置

所有服务使用默认 vhost `/`（AMQP URI 中写为 `/%2F`），通知连接独立于 Platform/Hub 原有业务 MQ 配置。以下均为占位示例，密码须 URL 编码。通知只使用 RabbitMQ 账号和队列权限控制发布与消费，不再配置应用层通知密钥。

Shield 的 `.env.compose`：

```dotenv
VISION_SHIELD_NOTIFICATION_AMQP_URL='amqps://shield-publisher:ENCODED_PASSWORD@MQ_HOST:5671/%2F'
NOTIFICATION_TARGETS='["dev.platform","dev.event-hub","prod.platform","prod.event-hub"]'
```

Platform 和 Event Hub 各部署的环境变量：

```dotenv
VISION_SHIELD_BASE_URL=http://SHIELD_HOST:28000
VISION_SHIELD_NOTIFICATION_AMQP_URL='amqps://environment-consumer:ENCODED_PASSWORD@MQ_HOST:5671/%2F'
VISION_SHIELD_NOTIFICATION_TARGET=dev.platform
```

| 部署 | 通知目标 |
| --- | --- |
| 研发 Platform | `dev.platform` |
| 研发 Event Hub | `dev.event-hub` |
| 正式 Platform | `prod.platform` |
| 正式 Event Hub | `prod.event-hub` |

Platform 播放通过登录、设备权限和共享观看会话鉴权，不再要求 `CAMERA_REDACTION_PLAY_SECRET`，也不再生成额外播放 token。固定外部 FLV 不需要 Platform 内部 HLS 配置。通知消费者分别在 Platform `camera-redaction-worker` 与 Hub `media_redaction_worker` 中启动，不在 API 中启动。更新环境变量后重新创建这些 worker 和 API 容器，单纯 restart 不会更新 Compose 注入的环境变量。

队列固定为 `hos.vision-shield.task-events.v1.<目标>`，例如 `hos.vision-shield.task-events.v1.dev.platform`。使用默认 exchange 按完整队列名直接投递，不修改已有业务 exchange、绑定、策略或队列。首次发布或消费时声明 durable quorum 队列，参数为 `x-max-length=10000`、`x-overflow=reject-publish`、`x-delivery-limit=-1`。最后一项避免暂时数据库故障因 quorum 重投上限丢失消息；消费者延迟 1 秒再 NACK，非法通知直接拒绝，不进行无限重投。

通知依赖使用 `aio-pika>=10.0.1,<11` 及 `pamqp 4`；旧 `pamqp 3.3.0` 会在编码 `x-delivery-limit=-1` 时抛出 `struct.error`，队列声明尚未发到 RabbitMQ。更新依赖锁后需重新构建 Shield 和 Hub 镜像并重建通知进程；Platform 当前锁已使用 `pamqp 4.0.1`。CPU 和 GPU 部署锁保持同步，不通过移除 `-1` 改变重投语义。

部署前在 RabbitMQ 管理界面只读核查四个队列名是否已存在；若同名但属性不符，停止部署并处理命名冲突，不删除或重建已有队列。通知账号仅授权其所需的上述队列；Shield 账号可发布所有已登记目标，各环境消费者只读取自己的队列。代码不会修改 broker 账号权限。RabbitMQ 必须支持 quorum 队列。

#### 持久交接与补偿

独立运行 `python -m vision_shield.jobs.notify`（Compose 已提供 notifier）。任务状态与通知责任原子登记；持久消息、mandatory 路由及 publisher confirm 全部成功后才释放 Shield 的本次通知责任。发布全过程超时 5 秒，失败在 Redis 中指数退避 1—300 秒，仅重试通知，不重新推理。多个 notifier 共享全局最多 4 个、每目标最多 2 个在途通知，目标轮转领取，过期租约可接管。

MQ header `X-Vision-Shield-Target` 标识通知目标，不再发送或校验应用层签名。正文沿用 `schema_version=1`、`event_id=task_id:sequence`、`sequence`、`task_id`、`submission_key`、`kind`、`state`、`desired_state`、`ready`、`occurred_at`、`health`、`error_code`，不含输入、输出 URL 或来源凭据。正文最多 16 KiB。允许消费者离线积压；实时流健康仍以原始 `last_output_at` 和 `valid_until` 为准，收信不能延长旧画面期限。

消费者校验目标、冻结提交身份与序号，数据库提交成功后才 ACK；数据库故障重投，重复和旧序号不重复推进。Platform 与 Hub 都保留 `POST /v1/tasks/query` 批量校准，每批最多 100 项。查询响应不能让旧状态覆盖新通知。

新目标的幂等作用域为 `(notification_target, Idempotency-Key)`，业务 `submission_key` 原样保存。Platform/Hub 首次提交时把目标与业务意图一起持久化，重试不随当前环境变量改路由。历史未登记通知目标的作业沿用原 `platform`、`event-hub` 目标。切换目标前须先收敛旧在途任务及通知，不直接更换仍有在途任务的目标。

通知目标是路由和幂等作用域，不是 Shield API 的身份认证。API 继续依赖受控网络访问；需要阻止任意调用方跨目标操作时，须另配访问控制。研发和正式还必须使用不同的推流路径，避免共享 GPU 输出覆盖；共享 GPU 容量不会因队列隔离而分开。

`NOTIFICATION_LIMIT` 默认 1000，创建时预留责任额度；文件终态被 broker 确认后释放，直播保持至最终 stopped/failed 发布确认。满额返回 429/Retry-After，既有幂等键仍可恢复。Redis 必须持久化且 noeviction。通知发布确认不表示业务已经归档，Hub 仍需下载、校验、上传 MinIO 并更新告警；本期不自动删除原件或成品。

### Platform 内部 HLS 对接

复用现有 MediaMTX 1.21.2，将 `HLS_CDN_SECRET` 配成至少 32 字符随机密钥，并与 Platform `VISION_SHIELD_HLS_CDN_SECRET` 一致；内部端口默认仅本机可达。任务固定 RTSP 路径继续为 `redacted/{task_id}`，Platform 按同一路径代理 HLS。API 转发地址与内部 HLS 地址分别配置，不向浏览器公开 CDN 密钥。

CDN 密钥用于识别内部代理，不能单独禁止匿名 HLS 会话。配置已将 read 纳入原有媒体鉴权：内部 RTSP read 保持可用，普通 HLS read 拒绝，持 CDN 密钥的 Platform 请求绕过浏览器 cookieCheck 跳转；推流仍校验 worker 代次和租约。2026-10-23 用户指定转封装与媒体播放可用性由其他团队确保，本轮不再运行本地 MediaMTX 或浏览器播放验收；这些配置属于对接合同，不代表已部署。

### 遮挡样式参数

创建任务时传入 `redaction_mode`：`solid`（默认）为不透明深灰色矩形 RGB `(64, 64, 64)`，`mosaic` 为马赛克。此参数由接口请求控制，不使用环境变量。每个任务单独保存，查询响应会返回所选样式，流重连继续使用原值；未指定或历史任务缺少该字段时使用 `solid`。

- `POST /v1/file-tasks`：JSON 增加 `"redaction_mode": "mosaic"`，与 `kind`、`url` 一起提交。
- `POST /v1/stream-tasks`：JSON 增加 `"redaction_mode": "mosaic"`，与 `url` 一起提交。
- `POST /v1/file-tasks/uploads`：multipart 增加文本字段 `redaction_mode=mosaic`，与 `kind`、`file` 一起提交。

仅接受 `solid`、`mosaic`，非法值返回422。样式在创建时确定，不支持修改正在处理的任务。马赛克每个区域最多8×8色块；小于2像素的窄区域使用深灰色，避免原像素直接透出。

### 优化轮次与回退

从2026-10-21开始，每轮优化完成验证后必须提交，并创建不可覆盖的 Git 附注标签 `opt-YYYYMMDD-NN`（当日序号从01递增）。标签说明记录改动、验证和未完成的验收；这些是优化检查点，不代表生产性能认证或正式语义版本发布。不得移动已有标签。服务器部署需先将对应提交和标签推送到 `origin`；代码推送不代表服务器已完成部署或性能验收。

各轮检查点与验证边界见 [变更记录](CHANGELOG.md)。效果不理想时，可用 `git worktree add --detach ../hos-vision-shield-baseline <目标标签>` 检查指定基线，保留当前工作区；正式回退使用经确认的 revert 或按标签重新部署，不默认使用 `reset --hard`。源码标签不包含模型、环境配置、Redis任务和忽略的样例文件，重新部署需保留对应运行资料。
