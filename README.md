# Vision Shield

Python/FastAPI + Taskiq 媒体脱敏项目，支持图片、视频与 RTSP 流中的人脸和车牌深灰色遮挡或马赛克，不做识别或音频脱敏。
支持直接在 Windows 本机运行；服务器另提供 CPU／指定单卡 GPU 的 Docker Compose 部署。
项目以 `main` 分支作为主干。

当前进度见 [任务清单](specs/001-media-redaction/tasks.md) 与 [验证记录](specs/001-media-redaction/validation/gates.md)。
已接通上传/URL文件任务、结果下载、取消、RTSP输出及断流/worker重启恢复。API与任务控制使用异步I/O；模型推理、编解码在受控子进程中执行。

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

主车牌设备诊断默认关闭。上传视频时可加 `X-Detection-Profile: 1`，在第64、424、768个合格全图调用采样；`GET /v1/tasks/{id}/diagnostics`读取摘要，添加`?sample=64`等固定索引可下载原始trace。每份最多32 MiB、每任务最多三份，短视频或未进入采样位置返回partial。该模式要求CUDA模型，采样会扰动耗时，其FPS不作为优化验收。

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
```

代码须先从开发机提交并推送到远程 `main`；首次使用前先执行 `git -c http.version=HTTP/1.1 pull --ff-only origin main` 获取脚本。脚本固定使用 GPU 文件组合、项目名 `vision-shield` 和现有 `.env.compose`，依次拉取代码、检查配置、构建、更新容器并等待就绪（最多 240 秒），成功后输出提交号。未提交的跟踪文件修改、分支不符、分叉或本地额外提交均会停止部署。脚本不备份、不强制重置、不自动回滚，网络或构建失败时不会进入容器更新步骤。

日常更新不先执行 `down`，构建成功后才替换容器。重建期间旧容器继续服务，替换时会短暂中断；正在执行的文件任务不会续跑，流任务会尝试恢复。当前 Dockerfile 已分离依赖层与应用层，构建缓存可用且依赖未变时，源码修改可复用已安装的 PyTorch/CUDA 依赖；不要在日常更新中使用 `--no-cache`。

`compose.yaml` 包含 MediaMTX、API、reconciler 和两个 worker，共五个服务，直接复用现有带密码 Redis。在 `.env.compose` 设置 `REDIS_URL='redis://:URL编码后的密码@192.168.1.104:6378/6'`，并将文件权限设为 `600`；四个应用服务统一读取该地址。宿主发布端口须能从 Docker 网络访问。现有 Redis 的持久化及淘汰策略由部署方核对，不修改共享实例配置。

仅 API/RTSP 端口按配置发布；部署地址和端口由 `.env.compose` 的 `BIND_ADDRESS`、`API_PORT`、`RTSP_PORT`、`RTSP_PUBLIC_BASE_URL` 指定，模型及媒体目录由 `MODEL_DIR`、`MEDIA_DIR` 指定。容器内部推流与控制沿用服务名，远程输入还须配置来源允许列表。媒体目录保留在宿主机，切换前按原文件组合执行 `down`，不使用 `down -v`。

镜像构建可以联网安装依赖，`models/` 中的模型必须事先离线提供并只读挂载；quality_yolo 的 YOLO 权重随应用包交付，启动及推理不下载模型。worker 先在有 120 秒总超时的子进程中核验清单/哈希、实际推理及 NMS，成功后才接单并发布心跳；故障后的旧心跳最多有默认 20 秒租约窗口。`RTSP_PUBLISH_BASE_URL` 仅控制容器内部推流，对外仍返回公开播放地址。

GPU 镜像按 [研究记录中的生成命令](specs/002-compose-cpu-gpu/research.md)使用 `deploy/requirements-cu126.txt`，共同依赖沿用根锁版本；根声明变更时重新生成，禁止手工改 pin。CPU 使用根 uv.lock，Windows 安装方式保持不变。初次交付证据与未运行项见 [实施记录](specs/002-compose-cpu-gpu/validation/gates.md)，后续服务器分阶段实测见 [时序优化记录](specs/002-compose-cpu-gpu/temporal-optimization.md)；这些记录不代表完整部署验收已通过。

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

PATCH返回202表示停止意图已保存；查询到cancelled/stopped才表示资源清理完成。已成功文件不能再取消（409）。流自动重连并保留停止意图，文件worker崩溃则标记worker_lost，不自动续跑。

`GET /v1/tasks` 查询所有文件和流的未结束任务，返回 `{"current": [...], "queued": [...]}`，没有任务时对应空数组：

- `current`：处理中、连接中、运行中、仍由 worker 持有的恢复中任务，以及等待取消/停止完成的任务。
- `queued`：等待执行的 `queued` 任务，以及没有 owner、等待重新调度的 `recovering` 任务；只包含仍希望运行的任务。

每项字段与 `GET /v1/tasks/{id}` 相同，不返回源地址、凭据及内部 owner 信息。两组分别按创建时间升序、ID 次序排列，数组长度即该组任务数；展示顺序不代表调度顺序，也不是 Redis 消息队列中的消息数量。查询读取持久状态，非原子快照，状态和 owner 的失效由协调器更新。已结束任务不列出，仍可按 ID 查询。

```powershell
$tasks = Invoke-RestMethod http://127.0.0.1:8000/v1/tasks
$tasks.current
$tasks.queued
```

`GET /v1/diagnostics/resources` 查看有效 worker 的 CPU、内存和绑定显卡资源快照。CPU 为 worker 的 Linux cgroup v2 口径，包含媒体子进程；GPU 为所选显卡整卡口径，缺失和陈旧值会明确标记。字段单位、部署步骤及后续优化验收见[资源观测说明](specs/002-compose-cpu-gpu/diagnostics.md)。

默认只监听本机。其他系统接入时，部署方需同时修改启动脚本的API监听、MediaMTX的RTSP监听与 `.env` 的 `RTSP_PUBLIC_BASE_URL` 为可达内网地址，并限制防火墙可达范围。MediaMTX控制API和Redis继续仅监听本机。

## 模块与运行边界

`api/`负责HTTP契约，`jobs/`负责Taskiq、持久状态和租约，`media/`负责输入、进程生命周期及音视频流水线，`detection/`输出遮挡坐标，`redaction.py`执行可选遮挡及有界的相邻帧补偿。

默认每池最多100个未结束任务、每队列100条消息。实时流只缓存2个待处理画面，过载取最新帧；文件逐帧处理。AAC缓存超过16秒或4MiB会重建流会话，不无限堆积；该上限保留自EgoBlur CPU校准，并不是延迟保证。

## 巡逻机器人素材

面向运动模糊和超宽视角，默认组合在最长边 `DETECTION_MAX_SIDE=1280`（320–1920）的全图运行EgoBlur，再以YuNet/HyperLPR做全图补充。2026-10-21按用户接受极小目标漏遮的要求，补充分块由640改为1920源像素：最长边不超过1920时只运行一个全图视图，3840×1080使用全图加3个重叠分块，每个视图仍按`DETECTION_MAX_SIDE`缩放。第四轮停止对短边不足20源像素的低分人脸（EgoBlur低于0.3、YuNet低于0.6）放大复核；较高分人脸和车牌仍按原规则处理。不保证20源像素以下目标的覆盖，也不承诺所有更大目标都检出。人脸候选门槛分别为0.3/0.6，车牌为0.1/0.25；EgoBlur得分至少0.7的结果保留，其余候选须得到另一模型同类重叠框支持，或通过局部复核；弱车牌须有另一模型得分至少0.7的全图结果佐证，否则转局部复核。复核取候选周边3倍最长边、至少96像素区域，调整到最长边320后交给另一模型；同类框IoU至少0.2。全图人脸佐证至少0.3，否则转局部复核。短边至少128源像素的大人脸候选需至少一路高分结果：弱EgoBlur须有YuNet至少0.9佐证，低于0.9的YuNet须有EgoBlur至少0.7佐证；全图与局部采用相同分数要求。YuNet自身至少0.9时允许EgoBlur全图0.3或局部0.1佐证。短边20至128源像素的人脸保留原有低阈值局部复核，以保护侧脸；不会单独从低分人脸新增遮挡。该策略减少单模型误遮，也可能丢失无法确认的弱目标。车牌网络按其320输入约定做letterbox。`DETECTION_BOX_PADDING=0.15` 表示EgoBlur每侧15%、YuNet人脸每侧15%，HyperLPR车牌水平每侧30%、垂直15%。不做OCR、锐化或生成式修复。

`DETECTION_BACKEND=cpu`保留上一轮轻量方案（YuNet阈值0.3），无需加载EgoBlur；它在固定样本中有远处车牌漏遮，不作为当前效果验收结论。当前组合在CPU上更慢，接口仍由媒体子进程异步执行，不承诺实时帧率。新效果对照见[效果优先验证](specs/001-media-redaction/validation/quality-optimization.md)。 局部复核只运行待确认类别，Torch每个媒体子进程采用4线程，OpenCV仍为2线程；不改变检测范围、阈值或视频逐帧处理。分项计时与效果一致性验证见[同效果CPU推理优化](specs/001-media-redaction/validation/inference-performance.md)。

视频和流对已确认目标进行短时运动匹配，最多保留128个历史目标；匹配成功可补偿500毫秒内的漏检，匹配失败会尝试当前帧局部复核，仍未确认则只保留200毫秒旧框。只有真实检测能刷新有效期。尺寸变化、时钟倒退、较长跳帧或缩略图明显变化会清空历史，每次流重连重新建立状态。没有检测历史的目标不能凭空补出，快速转向或大幅变形仍可能漏遮。

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

需要通知时再传 `X-Notification-Target: event-hub` 或 `platform`，且必须同时传幂等键。部署通过 `NOTIFICATION_TARGETS` JSON 注册固定内部地址和独立密钥，例如下面的占位配置；实际地址必须能从服务器访问，不能使用浏览器的本地转发地址作为服务器回调目标。

```json
{"event-hub":{"url":"http://hub.internal/api/v1/internal/media-redaction/task-events","secret":"REPLACE_WITH_AT_LEAST_32_RANDOM_CHARS"},"platform":{"url":"http://platform.internal/api/v1/device/external/media-redaction/task-events","secret":"REPLACE_WITH_ANOTHER_RANDOM_SECRET"}}
```

独立运行 `python -m vision_shield.jobs.notify`（Compose 已提供 notifier）。任务状态与通知责任原子登记，HTTP 仅 2xx 才确认；重试重新签名，超时 5 秒，指数退避 1—300 秒。多个 notifier 共享全局最多 4 个在途通知、每目标最多 2 个和可接管租约。通知头为 `X-Vision-Shield-Timestamp` Unix 秒及 `X-Vision-Shield-Signature`，签名为 HMAC-SHA256(secret, timestamp + "." + 原始正文)。正文只含 schema_version、event_id、sequence、task_id、submission_key、kind、state、desired_state、ready、occurred_at、health 和安全 error_code，不含媒体或来源凭据。

新增 `POST /v1/tasks/query`，正文 `{"task_ids":["任务UUID"]}`，每批 1—100 项，返回 `items` 与 `missing_ids`，包含已退出活跃集合的终态。创建、查询及通知均带单调 `sequence`；接收方必须按序合并，不能让旧查询覆盖新通知。直播 `health` 由当前 Worker 租约和实际脱敏输出计算，`valid_until` 不晚于最后输出后 45 秒；查询成功不能延长健康。

`NOTIFICATION_LIMIT` 默认 1000，任务创建时预留通知责任额度；文件终态确认后释放，直播保持至最终 stopped/failed 确认。容量满返回 429/Retry-After，已存在幂等键仍可恢复。`MEDIA_DISK_HIGH_WATERMARK` 默认 0.95，磁盘达到高水位拒绝新任务。Redis 必须持久化并配置 noeviction；部署前核验 Redis 与媒体盘容量。通知确认不会删除输入或成品，本期没有自动清理。

Hub 收到通知只表示状态交接；完整结果仍由 Hub 下载、校验、上传 MinIO，随后更新 Platform 告警文件信息。该集成的业务闭环和浏览器播放仍须联调验收，单独通知成功不代表已交付。

### 遮挡样式

创建任务时传入 `redaction_mode`：`solid`（默认）为不透明深灰色矩形 RGB `(64, 64, 64)`，`mosaic` 为马赛克。此参数由接口请求控制，不使用环境变量。每个任务单独保存，查询响应会返回所选样式，流重连继续使用原值；未指定或历史任务缺少该字段时使用 `solid`。

- `POST /v1/file-tasks`：JSON 增加 `"redaction_mode": "mosaic"`，与 `kind`、`url` 一起提交。
- `POST /v1/stream-tasks`：JSON 增加 `"redaction_mode": "mosaic"`，与 `url` 一起提交。
- `POST /v1/file-tasks/uploads`：multipart 增加文本字段 `redaction_mode=mosaic`，与 `kind`、`file` 一起提交。

仅接受 `solid`、`mosaic`，非法值返回422。样式在创建时确定，不支持修改正在处理的任务。马赛克每个区域最多8×8色块；小于2像素的窄区域使用深灰色，避免原像素直接透出。

### 优化轮次与回退

从2026-10-21开始，每轮优化完成验证后必须提交，并创建不可覆盖的 Git 附注标签 `opt-YYYYMMDD-NN`（当日序号从01递增）。标签说明记录改动、验证和未完成的验收；这些是优化检查点，不代表生产性能认证或正式语义版本发布。不得移动已有标签。服务器部署需先将对应提交和标签推送到 `origin`；代码推送不代表服务器已完成部署或性能验收。

各轮检查点与验证边界见 [变更记录](CHANGELOG.md)。效果不理想时，可用 `git worktree add --detach ../hos-vision-shield-baseline <目标标签>` 检查指定基线，保留当前工作区；正式回退使用经确认的 revert 或按标签重新部署，不默认使用 `reset --hard`。源码标签不包含模型、环境配置、Redis任务和忽略的样例文件，重新部署需保留对应运行资料。
