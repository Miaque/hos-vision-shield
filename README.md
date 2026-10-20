# Vision Shield

Python/FastAPI + Taskiq 媒体脱敏项目，支持图片、视频与 RTSP 流中的人脸和车牌实心遮挡，不做识别或音频脱敏。
支持直接在 Windows 本机运行；服务器另提供 CPU／指定单卡 GPU 的 Docker Compose 部署。
项目以 `main` 分支作为主干。

当前进度见 [任务清单](specs/001-media-redaction/tasks.md) 与 [验证记录](specs/001-media-redaction/validation/gates.md)。
已接通上传/URL文件任务、结果下载、取消、RTSP输出及断流/worker重启恢复。API与任务控制使用异步I/O；模型推理、编解码在受控子进程中执行。

## 安装 Python 依赖

```powershell
Set-Location H:\workspace\python\vision-shield
uv sync --frozen
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\scripts\download-cpu-models.ps1
```

解释器使用Python3.12，依赖由uv.lock锁定。Windows x64 安装 PyTorch 2.14.1 / torchvision 0.29.1 的 CUDA 12.6 构建，支持本机 GTX 1080；其他平台继续使用 CPU 构建。CUDA 构建也可执行 CPU 推理，不需要单独安装 CUDA Toolkit，但 GPU 运行需要兼容的 NVIDIA 驱动。
默认`DETECTION_BACKEND=quality`：EgoBlur全图双类检测加YuNet/HyperLPR分块补充，优先减少漏遮。四个模型都必须可用。模型可以从已准备机器离线复制整个 `models/`（含两份 manifest 和许可证），启动及推理不下载权重；上面的下载脚本仅供联网准备模型时使用。已有.env需同步后端及四条模型路径。

### 本机 GTX 1080

在 `.env` 设置 `INFERENCE_DEVICE=cuda`，然后运行原有 `scripts/start-application.ps1`。EgoBlur 的两个模型和输入使用 GPU，YuNet/HyperLPR、编解码和遮挡继续使用 CPU，检测阈值与策略不变。要使用 CPU，只需改为 `INFERENCE_DEVICE=cpu` 后重启应用。未设置该字段仍默认 CPU；轻量 `DETECTION_BACKEND=cpu` 不接受 cuda 设备。

首次安装或驱动变更后，先执行真实模型验证：

```powershell
$env:RUN_GPU_TESTS='1'
uv run --frozen pytest tests/integration/test_gpu_model_probe.py -q
Remove-Item Env:RUN_GPU_TESTS
```

验证使用本地四个模型和 `samples/robot-quality` 中已标注样例，检查模型校验和、真实 GPU 推理和遮挡区域。GPU 不可用或推理失败会报错，不回退 CPU。更新依赖后须重启已有进程；本机 GPU 验证不代表服务器 x86_64 Compose 部署已通过。

本机已完成真实模型、图片/视频子进程及同图 CPU/GPU 对比，边界与结果见 [GTX 1080 验证记录](specs/002-compose-cpu-gpu/local-gpu.md)。
GPU 模式下，两路独立的全图检测同时执行，随后仍按原顺序进行局部复核；固定全集输出保持一致，最新耗时见 [GPU 同效果优化记录](specs/002-compose-cpu-gpu/gpu-performance.md)。

## 服务器 Docker Compose

目标为 Ubuntu 24.04 x86_64，沿用服务器现有 Docker 和 `docker-compose v2.29.2`；不在 Windows 本机安装 Docker。默认 CPU，GPU 模式指定一张卡供文件与流 worker 共用。GPU 失败不换卡或回退 CPU，检测策略及四份权重保持相同。

按 [服务器部署验收指南](specs/002-compose-cpu-gpu/quickstart.md)准备离线模型目录、媒体目录权限和 `.env.compose`，然后在服务器仓库根目录选择一种方式：

```bash
# CPU
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml up -d --build
# GPU：先在 .env.compose 填写 GPU_DEVICE_ID，推荐完整 GPU UUID
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.gpu.yaml up -d --build
```

`compose.yaml` 包含 MediaMTX、API、reconciler 和两个 worker，共五个服务，直接复用现有带密码 Redis。在 `.env.compose` 设置 `REDIS_URL='redis://:URL编码后的密码@192.168.1.104:6378/6'`，并将文件权限设为 `600`；四个应用服务统一读取该地址。宿主发布端口须能从 Docker 网络访问。现有 Redis 的持久化及淘汰策略由部署方核对，不修改共享实例配置。

仅 API/RTSP 端口按配置发布；部署示例设置 `BIND_ADDRESS=192.168.1.104`、`RTSP_PUBLIC_BASE_URL=rtsp://192.168.1.104:8554`，API 文档地址为 `http://192.168.1.104:8000/docs`。模型及媒体目录分别为 `/data/hos-vision-shield/models`、`/data/hos-vision-shield/data`。容器内部推流与控制沿用服务名，远程输入还须配置来源允许列表。媒体目录保留在宿主机，切换前按原文件组合执行 `down`，不使用 `down -v`。

镜像构建可以联网安装依赖，模型必须事先离线提供并只读挂载；不随镜像下载或打包。worker 先在有 120 秒总超时的子进程中核验清单/哈希、实际推理及 NMS，成功后才接单并发布心跳；故障后的旧心跳最多有默认 20 秒租约窗口。`RTSP_PUBLISH_BASE_URL` 仅控制容器内部推流，对外仍返回公开播放地址。

GPU 镜像按 [研究记录中的生成命令](specs/002-compose-cpu-gpu/research.md)使用 `deploy/requirements-cu126.txt`，共同依赖沿用根锁版本；根声明变更时重新生成，禁止手工改 pin。CPU 使用根 uv.lock，Windows 安装方式保持不变。镜像和目标服务器尚未验收，证据与未运行项见 [实施记录](specs/002-compose-cpu-gpu/validation/gates.md)。

## 本地 Redis / MediaMTX

如果.env已指向运行中的Redis和MediaMTX，直接启动应用即可，不必再运行基础服务脚本。当前用户环境的持久化检查与服务端配置步骤见[环境核对](specs/001-media-redaction/validation/environment.md)。

本机已有 Redis7.4.3；MediaMTX1.21.2官方可执行程序位于被Git忽略的 `.tools/mediamtx/`。
在项目根目录运行，保持这个终端打开：

```powershell
.\scripts\start-infrastructure.ps1 `
  -RedisExecutable 'H:\software\Redis-7.4.3\redis-server.exe' `
  -MediaMtxExecutable 'H:\workspace\python\vision-shield\.tools\mediamtx\mediamtx.exe'
```

退出时Ctrl+C只关闭该脚本创建的两个进程，不操作其他已存在的服务。
Redis数据保存在 `data/redis/`，日志与媒体服务运行文件位于 `data/`。
仅做启动和持久化配置自检，检查后自动回收进程：

```powershell
.\scripts\start-infrastructure.ps1 `
  -RedisExecutable 'H:\software\Redis-7.4.3\redis-server.exe' `
  -MediaMtxExecutable 'H:\workspace\python\vision-shield\.tools\mediamtx\mediamtx.exe' -Check
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

默认只监听本机。其他系统接入时，部署方需同时修改启动脚本的API监听、MediaMTX的RTSP监听与 `.env` 的 `RTSP_PUBLIC_BASE_URL` 为可达内网地址，并限制防火墙可达范围。MediaMTX控制API和Redis继续仅监听本机。

## 模块与运行边界

`api/`负责HTTP契约，`jobs/`负责Taskiq、持久状态和租约，`media/`负责输入、进程生命周期及音视频流水线，`detection/`输出遮挡坐标，`redaction.py`执行实心遮挡及有界的相邻帧补偿。

默认每池最多100个未结束任务、每队列100条消息。实时流只缓存2个待处理画面，过载取最新帧；文件逐帧处理。AAC缓存超过16秒或4MiB会重建流会话，不无限堆积；该上限保留自EgoBlur CPU校准，并不是延迟保证。

## 巡逻机器人素材

面向运动模糊、超宽视角和远处小目标，默认效果优先组合在最长边 `DETECTION_MAX_SIDE=1280`（320–1920）的全图运行EgoBlur，再以YuNet/HyperLPR做全图及有界重叠分块补充。人脸候选门槛分别为0.3/0.6，车牌为0.1/0.25；EgoBlur得分至少0.7的结果保留，其余候选须得到另一模型同类重叠框支持，或通过局部复核。复核取候选周边3倍最长边、至少96像素区域，调整到最长边320后交给另一模型；同类框IoU至少0.2。全图人脸佐证至少0.3，否则转局部复核。短边至少128源像素的大人脸候选需至少一路高分结果：弱EgoBlur须有YuNet至少0.9佐证，低于0.9的YuNet须有EgoBlur至少0.7佐证；全图与局部采用相同分数要求。YuNet自身至少0.9时允许EgoBlur全图0.3或局部0.1佐证。较小人脸保留原有低阈值局部复核，以保护远处侧脸；不会单独从低分人脸新增遮挡。该策略减少单模型误遮，也可能丢失无法确认的弱目标。小图不额外分块，车牌网络按其320输入约定做letterbox。`DETECTION_BOX_PADDING=0.15` 表示EgoBlur每侧15%、YuNet人脸每侧15%，HyperLPR车牌水平每侧30%、垂直15%。不做OCR、锐化或生成式修复。

`DETECTION_BACKEND=cpu`保留上一轮轻量方案（YuNet阈值0.3），无需加载EgoBlur；它在固定样本中有远处车牌漏遮，不作为当前效果验收结论。当前组合在CPU上更慢，接口仍由媒体子进程异步执行，不承诺实时帧率。新效果对照见[效果优先验证](specs/001-media-redaction/validation/quality-optimization.md)。 局部复核只运行待确认类别，Torch每个媒体子进程采用4线程，OpenCV仍为2线程；不改变检测范围、阈值或视频逐帧处理。分项计时与效果一致性验证见[同效果CPU推理优化](specs/001-media-redaction/validation/inference-performance.md)。

视频和流保存至多两组真实检测框，在200毫秒源时间内补遮短暂漏检；尺寸变化、时钟倒退、较长跳帧或缩略图明显变化会清空历史，每次流重连重新建立状态。补偿不累计刷新过期时间；没有检测历史的目标不能凭空补出。这不是运动跟踪，快速转向和目标快速移动仍可能漏遮；小于阈值的画面变化也可能保留旧位置造成多遮。

检测仍可能漏检和误遮，需要按实际机器人素材复核，不承诺模糊素材零漏检。降低DETECTION_MAX_SIDE会影响人脸细节，车牌网络输入及分块数也会影响耗时。与原EgoBlur 1280基线的对照入口：

```powershell
uv run python scripts/evaluate-robot-media.py C:\Users\zhanjh\Downloads\demo --output data/validation/cpu-evaluated
```

结果包括每张图片、每段视频的前三个连续帧的对照图、脱敏图、耗时和源文件哈希。比较工具还需要本机保留的两个EgoBlur模型。它不是完整视频或准确率验收，原始素材不覆盖、不向外部服务上传。CPU优化与限制见[验证记录](specs/001-media-redaction/validation/cpu-optimization.md)。

调整.env后重启API、协调器和worker；现有任务接口保持不变。

原始文件、完成结果和失败半成品均保留于 `data/<任务ID>/`，不会自动删除；只有成功结果能下载。视频流不录像。日志在 `data/application/`，按业务task ID定位，不输出源URL和密码。

## 检测模型

模型来源：OpenCV Zoo及HyperLPR作者仓库固定版本记录在 `models/cpu-manifest.json`，EgoBlur固定版本记录在 `models/manifest.json`。下载、许可和切换方法见[模型说明](models/README.md)。效果优先组合中任一模型文件缺失或推理失败都会使处理失败，不返回部分模型结果或透传原始画面。

## 运行验证

```powershell
$env:TEST_REDIS_SERVER='H:\software\Redis-7.4.3\redis-server.exe'
$env:TEST_MEDIAMTX_BIN=(Resolve-Path .tools/mediamtx/mediamtx.exe).Path
uv run pytest -q
```

测试启动独立临时端口和数据目录，不清空现有Redis。FFmpeg与ffprobe需在PATH中，本机已经具备。
真实媒体样本和权重不提交Git；自动生成的图片和视频仅用于媒体机制验证。效果回归另需本机`samples/robot-quality/`十四张授权样本，其哈希和已核对的小区域记录在`tests/fixtures/quality-cases.json`；缺失时明确失败，不跳过。原始demo中的image-2、7、13、17、19按该清单命名保留，rtsp-green.png来自已保存的source-first截图；另三张PNG是demo视频9第0帧、视频10第42帧和视频11第17帧。video-04-frame-24.png来自`tests/test_assets/22-25-32违规占道.mp4`开头一秒连续片段的第24帧（从0计数），另加入同片段的第09、10、16、19帧，覆盖五张夜间轮胎误检帧与同帧车牌保护回归。
实际摄像头接入时在 `.env` 中将 `SOURCE_ALLOWLIST` 设置为授权主机/IP/CIDR的JSON数组；默认空数组拒绝远程来源。

当前没有用户鉴权或自动文件删除，原音频不脱敏；检测效果、完整故障恢复与容量以各阶段实测为准。
