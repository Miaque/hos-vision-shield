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

将仓库和模型包传至服务器；解包到独立目录 `/data/hos-vision-shield/models`，确认模型与清单位于该目录第一层。准备媒体目录：

```bash
sudo install -d -m 755 /data/hos-vision-shield/models
sudo install -d -o 10001 -g 10001 -m 750 /data/hos-vision-shield/data
sudo tar -xzf ../vision-shield-models.tar.gz -C /data/hos-vision-shield/models
```

模型目录和文件须允许运行用户读取；运行时只读挂载，不让应用修改模型或补下载。镜像可联网构建或获取，权重包不传入构建上下文。

## 2. 填写配置并选择模式

在服务器仓库根目录：

```bash
cp deploy/compose.env.example .env.compose
mkdir -p acceptance/cpu acceptance/gpu
```

部署示例已设置 `MODEL_DIR=/data/hos-vision-shield/models`、`MEDIA_DIR=/data/hos-vision-shield/data`、`BIND_ADDRESS=192.168.1.104` 和 `RTSP_PUBLIC_BASE_URL=rtsp://192.168.1.104:8554`。`REDIS_URL` 必填现有带密码 Redis 的地址，例如 `REDIS_URL='redis://:URL编码后的密码@192.168.1.104:6378/6'`；使用 6 号数据库，文件权限设为 `600`。宿主发布端口须允许 Docker 网络访问，现有 Redis 的持久化及淘汰策略另行核对，不由本项目修改。远程来源须填写 `SOURCE_ALLOWLIST`；授权主机名时也授权实际解析 IP/CIDR，空列表仅允许本地上传。GPU 模式使用已核对的第 6 号卡 UUID，换卡时修改 `GPU_DEVICE_ID`。

以下 Bash 数组统一保存当前模式的命令，避免后续启动与停止使用不同文件组合：

```bash
MODE=cpu  # GPU 验收改成 gpu
dc=(docker-compose --env-file .env.compose -p vision-shield -f compose.yaml)
if [ "$MODE" = gpu ]; then dc+=(-f compose.gpu.yaml); fi
"${dc[@]}" config --quiet
"${dc[@]}" build
```

两种组合都必须通过配置解析，并在本地用 `"${dc[@]}" config` 检查合并配置的镜像与构建参数：CPU 组合四个应用服务均为 `image: vision-shield:cpu`、`build.args.TORCH_BACKEND: cpu`，没有设备预留；GPU 组合两个 worker 均为 `image: vision-shield:cu126`、`build.args.TORCH_BACKEND: cu126`，且必须只有同一个 device_ids 元素，API/reconciler 保持 CPU 镜像及构建参数、不分配 GPU。不要将含凭据的完整展开配置提交或贴入日志。GPU组合为file-worker和stream-worker各设2个副本，每个worker保留单任务限制；API/reconciler及CPU组合副本数沿用原值。检查资源接口的两个池各有2个有效worker，并确认两个同池任务实际同时进入处理状态，不能仅提交两项任务就视作双路。

构建完成后，从应用容器验证 Redis 认证及数据库选择：

```bash
"${dc[@]}" run --rm --no-deps api python -c 'import os; from redis import Redis; r=Redis.from_url(os.environ["REDIS_URL"],socket_connect_timeout=3,socket_timeout=3); assert r.connection_pool.connection_kwargs["db"]==6; assert r.ping(); print("Redis DB6 PING OK")'
```

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
MODEL_DIR=/data/hos-vision-shield/models
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
curl --fail http://192.168.1.104:8000/health/ready
```

示例 curl 地址须与实际 API 绑定和端口一致；只绑定指定 LAN 接口时使用该接口地址。预期两类 worker 均完成预检且无残留 probe，ready 返回 200。干净首次启动中某类 worker 预检失败时不得 ready；运行中崩溃后等待原心跳租约过期再验证 ready 失败，默认窗口最多约 20 秒，正常退出主动撤心跳。仅看到容器 running 或 GPU 列表不能算通过。

## 5. 同素材图片、视频与流验收

准备包含可人工核对人脸及车牌的授权图片和 MP4/H.264 视频。每种模式分别保存输入哈希、请求与查询结果、下载结果及人工核对记录。

```bash
API=http://192.168.1.104:8000
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

## 7. Git 更新与重新构建

当前服务器已经从源码包部署 GPU 版本时，只需将源码获取方式改成 Git；现有 Dockerfile 和 Compose 已支持从本地源码构建。下面以现有项目名 `vision-shield` 为例；若首次部署用了其他 `-p` 名称，后续必须使用原名称。可用 `docker-compose ls` 核对，避免创建第二套服务占用相同端口。

### 首次从源码包目录切换

先在开发机提交并推送需要部署的变更到远程 `main`。服务器需要 Git 及仓库读取权限；私有仓库使用 SSH key 或 Git 凭据管理，不将令牌写进仓库文件。

在服务器替换下面两项占位值，克隆到旧目录旁边的新目录，保留旧目录用于核对：

```bash
OLD_SOURCE='/实际路径/旧源码目录'
REPO_URL='仓库的SSH或HTTPS地址'
cd "$(dirname "$OLD_SOURCE")" &&
git clone --branch main --single-branch "$REPO_URL" hos-vision-shield-git &&
cd hos-vision-shield-git &&
cp -p "$OLD_SOURCE/.env.compose" .env.compose &&
chmod 600 .env.compose
```

复制服务器已有配置，不用示例覆盖它。确认其中 `MODEL_DIR`、`MEDIA_DIR` 仍指向原来的宿主绝对目录，`REDIS_URL`（包括数据库编号）、GPU UUID、端口及允许列表保持服务器实际配置。模型、结果文件和 Redis 数据无需搬移；若改过旧目录中的 Compose 或 MediaMTX 配置，先核对差异并保留必要设置，再进行更新。`.env.compose` 已被 Git 和 Docker 构建上下文忽略，后续拉取代码不覆盖此文件。

然后执行下方更新命令。新旧目录均操作同一 Compose 项目，切换后统一从新目录管理服务，不再从旧目录执行 `up` 或 `down`。第一次切换目录时，MediaMTX 配置挂载的宿主路径也会变化，Compose 可能重建该容器，播放会短暂中断。

### 在现有源码包目录原地接入 Git

如果需要保留当前源码目录及所有挂载路径，可以使用此方式代替上面的克隆步骤。以下命令仅用于尚无 `.git` 的源码包目录，且已确认服务器没有独立代码修改；按当前部署约定不执行备份。保持在当前目录，将仓库地址替换为实际值，初始化并关联远程 `main`；此阶段只同步 Git 索引，不覆盖工作目录文件：

```bash
git init -b main &&
git remote add origin '仓库的SSH或HTTPS地址' &&
git fetch origin main &&
git reset --mixed origin/main &&
git branch --set-upstream-to=origin/main main
git diff --stat
git diff -- compose.yaml compose.gpu.yaml deploy/
```

核对差异，特别是服务器单独修改过的 Compose、Dockerfile 和 MediaMTX 配置。接下来这条命令会用刚获取的版本覆盖仓库跟踪文件，并补齐原源码包缺少的文件：

```bash
git restore --worktree -- .
git status --short
```

当前仓库未跟踪 `.env.compose`、`data/`、模型 `.jit`/`.onnx` 权重及 `vision-shield-models.tar.gz`，这些文件不会被上述同步覆盖；模型清单和说明属于跟踪文件，会随源码更新。不要执行 `git clean`，也不要把服务器凭据或权重加入暂存区。后续 `pull` 前保持跟踪文件无未提交修改。

然后按下方命令构建及更新 GPU 容器。原地接入 Git 本身不重启容器；实际中断发生在更新容器时。此 Git 迁移流程已在临时目录核对：原有配置、媒体、权重和压缩包内容保持不变，跟踪源码更新；不代表服务器构建或运行已验收。

### 后续日常更新

GPU 服务器使用 `scripts/deploy-server.sh`。前提是当前目录已经接入 Git、分支为 `main`，保留原 `.env.compose`，且原部署项目名为 `vision-shield`。脚本可从任意目录以实际脚本路径调用，会自动进入仓库根目录。

开发机先提交并推送变更；第一次使用时，在服务器仓库根目录获取脚本：

```bash
git -c http.version=HTTP/1.1 pull --ff-only origin main
```

成功后，以后每次更新只需执行：

```bash
bash scripts/deploy-server.sh
```

脚本依次验证 Git/配置前提、使用 HTTP/1.1 拉取 `origin main`、静默校验 Compose 配置、构建 CPU/GPU 镜像，再更新容器并等待服务运行及 API 健康（最多 240 秒）。沿用原 API `/health/ready` 和 worker 预检机制，成功后显示服务状态及部署提交号；HTTP/1.1 不保证网络可用，拉取失败时直接停止。

脚本拒绝未提交的跟踪文件修改、非 `main` 分支、分叉或本地额外提交，不自动 stash、强制重置、备份或回滚，不输出包含凭据的展开配置。脚本更新自身时，本次执行使用启动时已解析的脚本内容。验证使用真实 Bash 和临时 Git 仓库、替代 Compose 命令；没有执行真实 Docker 构建或服务器部署。

CPU 部署仍可使用手动命令；GPU 模式的手动命令与脚本主体对应如下：

```bash
dc=(docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.gpu.yaml)
git pull --ff-only origin main &&
"${dc[@]}" config --quiet &&
"${dc[@]}" build &&
"${dc[@]}" up -d --no-build --wait --wait-timeout 240
```

`--ff-only` 在服务器分支产生分叉时停止更新，不自动合并。服务器的环境差异保存在 `.env.compose`，代码变更在开发机提交；遇到本地修改冲突时先核对，不通过强制重置覆盖。开发机仅提交还不够，必须先推送到远程，服务器才能拉到该提交。

构建时旧容器继续运行，构建失败不会进入 `up`；修复后重新执行即可。`up` 会依据镜像及配置变化重建所需容器；仅执行 `restart` 不会切换到新镜像。日常更新不先 `down`、不删除卷或数据、不使用 `--no-cache`。依赖及构建指令未变化、构建缓存仍可用时，只改源码可复用 PyTorch/CUDA 依赖层。

更新前等文件任务完成，或安排维护窗口：容器替换会中断未完成的文件处理并最终标记 `worker_lost`，文件不自动续跑；未显式停止的流会按原恢复机制重新调度，重连期间播放中断。此流程不提供不停机发布；新镜像启动失败也不会自动回滚。

启动命令退出后核对容器状态和 worker 预检日志，并等就绪接口返回 200：

```bash
git log -1 --oneline
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.gpu.yaml ps
docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.gpu.yaml logs --tail 100 api file-worker stream-worker
# 替换为服务器实际绑定的 API 地址；worker 预检完成后应返回 200。
curl --fail http://192.168.1.104:8000/health/ready
```

Git 提交号仅标识当前源码，不能单独证明容器已经切换或业务验收通过。以上流程在本地仅核对文档和现有构建配置，服务器实际更新后仍需执行上述检查。

命令行为参考 [Git pull](https://git-scm.com/docs/git-pull)、[Compose up](https://docs.docker.com/reference/cli/docker/compose/up/) 和 [Docker 构建缓存](https://docs.docker.com/build/cache/invalidation/)。
