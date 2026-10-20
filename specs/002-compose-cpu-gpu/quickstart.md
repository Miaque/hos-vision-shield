# 服务器部署验收指南

Compose、Dockerfile 和预检实现已经提供；以下构建和运行命令由部署人员在服务器执行，尚未取得镜像或服务器验收证据。不在 Windows 本机安装 Docker，沿用目标服务器现有版本。交付文件与配置含义见 [部署契约](contracts/deployment.md)，记录字段见 [数据与生命周期](data-model.md)，本地验证边界见 [实施记录](validation/gates.md)。

## 1. 准备服务器与离线模型

目标：Ubuntu 24.04.4 x86_64、RTX 6000 Ada、驱动 595.84；使用现有 `docker-compose v2.29.2`。CPU 模式不需要 GPU 或 NVIDIA 容器运行时。

使用本次交付的源码包与离线模型包时，先将两个 tar.gz 和 `SHA256SUMS` 上传至服务器同一目录，再执行：

```bash
sha256sum -c SHA256SUMS
mkdir -p vision-shield
tar -xzf vision-shield-source.tar.gz -C vision-shield
cd vision-shield
```

已有模型包可直接执行下文的解包步骤，跳过重新打包。后续命令均在源码目录运行；单独传输仓库时，将模型包路径改成实际位置。

在服务器记录 Docker Engine 与 Compose 版本；GPU 模式另记录卡号、UUID 和容器运行时：

```bash
docker version
docker-compose version
# 仅 GPU 模式
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
docker info --format '{{json .Runtimes}}'
```

宿主 nvidia-smi 成功不能代替 GPU 容器验证。容器无法请求所选卡时，先按 [NVIDIA 官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)核对运行时；不要在应用启动中自动修改 Docker daemon 或宿主驱动。

在已有权重的准备机器、仓库根目录打包，无需重新下载：

```bash
# Windows 可用 tar.exe 执行相同参数
tar -czf vision-shield-models.tar.gz -C models face.jit license_plate.jit yunet_face.onnx hyperlpr_plate.onnx manifest.json cpu-manifest.json README.md NOTICE.md licenses
```

将仓库和模型包传至服务器；解包到独立目录，例如 `/srv/vision-shield/models`，确认模型与清单位于该目录第一层。准备媒体目录：

```bash
sudo install -d -m 755 /srv/vision-shield/models
sudo install -d -o 10001 -g 10001 -m 750 /srv/vision-shield/data
sudo tar -xzf ../vision-shield-models.tar.gz -C /srv/vision-shield/models
```

模型目录和文件须允许运行用户读取；运行时只读挂载，不让应用修改模型或补下载。镜像可联网构建或获取，权重包不传入构建上下文。

## 2. 填写配置并选择模式

在服务器仓库根目录：

```bash
cp deploy/compose.env.example .env.compose
mkdir -p acceptance/cpu acceptance/gpu
```

填写 `MODEL_DIR`、`MEDIA_DIR`、允许监听的 `BIND_ADDRESS` 和客户端可达的 `RTSP_PUBLIC_BASE_URL`。远程来源须填写 `SOURCE_ALLOWLIST`；授权主机名时也授权实际解析 IP/CIDR，空列表仅允许本地上传。GPU 模式填写所选一张卡的 `GPU_DEVICE_ID`，优先使用 UUID。

以下 Bash 数组统一保存当前模式的命令，避免后续启动与停止使用不同文件组合：

```bash
MODE=cpu  # GPU 验收改成 gpu
dc=(docker-compose --env-file .env.compose -p vision-shield -f compose.yaml)
if [ "$MODE" = gpu ]; then dc+=(-f compose.gpu.yaml); fi
"${dc[@]}" config --quiet
"${dc[@]}" build
```

两种组合都必须通过配置解析，并在本地用 `"${dc[@]}" config` 检查合并配置的镜像与构建参数：CPU 组合四个应用服务均为 `image: vision-shield:cpu`、`build.args.TORCH_BACKEND: cpu`，没有设备预留；GPU 组合两个 worker 均为 `image: vision-shield:cu126`、`build.args.TORCH_BACKEND: cu126`，且必须只有同一个 device_ids 元素，API/reconciler 保持 CPU 镜像及构建参数、不分配 GPU。不要将含凭据的完整展开配置提交或贴入日志。

## 3. 验证镜像与模型离线加载

CPU 模式确认镜像导入成功且为 CPU 构建：

```bash
"${dc[@]}" run --rm --no-deps file-worker python -c 'import av,cv2,torch,torchvision; print(torch.__version__,torchvision.__version__); assert torch.version.cuda is None'
```

GPU 镜像先检查构建依赖，该命令不请求 GPU，不要求实际显卡：

```bash
docker run --rm vision-shield:cu126 python -c 'import torch,torchvision; assert torch.version.cuda == "12.6"; print(torch.__version__,torchvision.__version__,torch.version.cuda)'
```

GPU 模式随后使用当前 GPU 文件组合验证唯一设备和非空 NMS；打印的设备应对应所选卡：

```bash
"${dc[@]}" run --rm --no-deps file-worker python -c 'import torch,torchvision; assert torch.cuda.is_available() and torch.cuda.device_count()==1; b=torch.tensor([[0.,0.,2.,2.],[0.,0.,1.,1.]],device="cuda"); s=torch.tensor([.9,.8],device="cuda"); k=torchvision.ops.nms(b,s,.3); assert k.is_cuda; print(torch.__version__,torchvision.__version__,torch.version.cuda,torch.cuda.get_device_name(),torch.cuda.get_device_capability(),torch.cuda.get_arch_list())'
```

另外用无网络的独立容器执行模型预检。将下面目录与卡标识设为实际值；预检入口与 worker 内部 probe 复用同一验证函数：

```bash
MODEL_DIR=/srv/vision-shield/models
docker run --rm --network none --mount type=bind,source="$MODEL_DIR",target=/models,readonly -e INFERENCE_DEVICE=cpu vision-shield:cpu python -m vision_shield.detection.preflight
# 仅 GPU 模式：填写实际所选 UUID
GPU_DEVICE_ID=GPU-替换为所选卡UUID
docker run --rm --network none --gpus "device=$GPU_DEVICE_ID" --mount type=bind,source="$MODEL_DIR",target=/models,readonly -e INFERENCE_DEVICE=cuda vision-shield:cu126 python -m vision_shield.detection.preflight
```

预期四份模型哈希、加载、实际推理和目标 NMS 均通过，退出码 0；该步骤仅证明检测可在无网络条件下运行。整套媒体验收另须限制外部模型来源，保留容器内部网络与授权媒体源，确认启动及处理均无模型下载尝试。

## 4. 启动与就绪

```bash
"${dc[@]}" up -d
"${dc[@]}" ps
"${dc[@]}" logs --tail 100 file-worker stream-worker
curl --fail http://127.0.0.1:8000/health/ready
```

示例 curl 地址须与实际 API 绑定和端口一致；只绑定指定 LAN 接口时使用该接口地址。预期两类 worker 均完成预检且无残留 probe，ready 返回 200。干净首次启动中某类 worker 预检失败时不得 ready；运行中崩溃后等待原心跳租约过期再验证 ready 失败，默认窗口最多约 20 秒，正常退出主动撤心跳。仅看到容器 running 或 GPU 列表不能算通过。

## 5. 同素材图片、视频与流验收

准备包含可人工核对人脸及车牌的授权图片和 MP4/H.264 视频。每种模式分别保存输入哈希、请求与查询结果、下载结果及人工核对记录。

```bash
API=http://127.0.0.1:8000  # 改成实际 API 地址
curl --fail -F kind=image -F file=@samples/validation.jpg "$API/v1/file-tasks/uploads"
curl --fail -F kind=video -F file=@samples/validation.mp4 "$API/v1/file-tasks/uploads"
# 将 TASK_ID 替换为各次响应 id，重复查询直至终态
TASK_ID=替换为任务ID
curl --fail "$API/v1/tasks/$TASK_ID"
# 仅在 state=succeeded 后下载；视频后缀用 mp4
curl --fail "$API/v1/tasks/$TASK_ID/result" -o "acceptance/$MODE/$TASK_ID.jpg"
```

每种模式的图片与视频分别测一次首次任务和至少三次后续任务，记录端到端耗时与后续中位数。每次业务任务都会新建子进程和加载模型，不能将后续任务称作常驻模型热推理。独立记录模型加载和帧处理时间，不以现有 stats.elapsed_seconds 冒充端到端时间。

每次业务子进程以 `model_load_seconds` 内部事件报告模型加载时间；worker 日志以 `task=<ID> generation=<代次> 模型加载耗时=<秒>` 关联任务。用 `"${dc[@]}" logs --no-color file-worker stream-worker` 提取日志，既有 HTTP stats 字段不增加。效果与样本哈希填入 [效果记录](validation/effects.md)。

流对比使用同一固定视频循环播放的授权 RTSP 源；不要将原始验证视频推到本部署受发布鉴权保护的脱敏路径。将源主机/IP加入允许列表后，提交实际源地址：

```bash
curl --fail -H 'Content-Type: application/json' -d '{"url":"rtsp://替换为授权来源/固定素材流"}' "$API/v1/stream-tasks"
STREAM_TASK_ID=替换为流任务ID
curl --fail "$API/v1/tasks/$STREAM_TASK_ID"
# ready=true 后，在允许网络中的客户端使用实际 playback_url
ffplay -rtsp_transport tcp rtsp://替换为服务器地址:8554/redacted/替换为流任务ID
curl --fail -X PATCH -H 'Content-Type: application/json' -d '{"desired_state":"stopped"}' "$API/v1/tasks/$STREAM_TASK_ID"
```

每种模式观察至少 60 秒，记录处理帧率、丢帧及输出；查询到 `stopped` 并确认会话和处理进程结束，才能算停止完成。CPU/GPU 共六组效果记录，均核对漏遮、误遮及模式差异，不以加载成功宣称零漏检。

GPU 验收期间记录 `docker stats --no-stream` 和 `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv`。至少一项文件任务和一项流任务须实际使用同一指定 GPU；再验证两池同时执行时的峰值占用，不能按单模型显存推算容量。

## 6. 故障、重启与设备切换

- 在独立验收项目和模型副本中验证缺文件、单字节损坏、文件不可读、不可加载、无效 GPU 标识和 CUDA 不可用。预期具体诊断、相关 worker 不登记心跳、ready 失败，且不换模型、不回退 CPU 或另一张卡。不要修改在用模型包。
- 正常停止/启动一次，确认已完成记录及结果仍可查询下载；CPU/GPU 各执行一次。另在活动流期间停止 stream-worker，确认媒体进程与发布会话回收，再启动 worker，验证 run 意图恢复；显式 stopped 的流不得复活。
- 验证 Linux 模型加载期间父进程退出、worker SIGTERM 和超过停止宽限的强制退出。未完成文件按既有失租约流程失败，半成品不得下载；重建容器后没有遗留媒体进程或显存持有者。

```bash
"${dc[@]}" stop -t 60 stream-worker
"${dc[@]}" start stream-worker
# 切换模式或日常移除容器，保留同一项目的数据
"${dc[@]}" down
```

日常命令不使用 `down -v`。切换模式后重新定义当前模式的 dc 数组，再执行配置检查、构建和启动。保留实际执行命令、退出码及证据；未执行的故障或效果场景必须写明，不能以配置解析或模拟测试替代服务器验收。
