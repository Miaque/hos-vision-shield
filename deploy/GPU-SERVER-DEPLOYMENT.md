# 新 GPU 服务器部署指南

本文面向从零准备的 Linux GPU 服务器，按“宿主机 → 容器 GPU → 离线模型 → 应用配置 → MPS/NUMA → 启动 → 媒体验收”的顺序部署 Vision Shield。

核对日期：2026-11-06。仓库基线：`main`，提交 `684bb76`。系统和 GPU 型号尚未确定，安装示例采用 **Ubuntu Server 22.04/24.04 x86_64、NVIDIA GPU、Docker Engine 和 Compose v2**；其他系统应替换宿主机安装步骤。本文是部署操作指南，未在新服务器上执行，不代表该机器已经通过兼容性或性能验收。

除注明在准备机器、播放客户端执行的命令外，以下命令均在目标服务器的 **Bash** 中执行。按段执行，任一命令失败时先排障再继续，不要整篇批量粘贴。`REPLACE_...` 都必须替换；示例 IP、UUID、CPU 编号不能原样用于新机器。

## 1. 先确定部署路径

### 1.1 当前代码的实际部署组合

| 项目 | 正式环境 | 独立测试环境 |
| --- | --- | --- |
| 部署命令 | `bash scripts/deploy-server.sh --env prod ...` | `bash scripts/deploy-server.sh --env test ...` |
| 环境文件 | `.env.compose` | `.env.compose.test` |
| Compose 项目名 | `vision-shield` | `vision-shield-test` |
| 配置组合 | `compose.yaml` + `compose.gpu.yaml` | 正式组合 + `compose.test.yaml` |
| GPU 镜像 | `vision-shield:cu126` | `vision-shield-test:cu130` |
| CUDA 构建 | 12.6 | 13.0 |
| worker 检测后端 | `quality_yolo` | `quality_yolo` |
| 文件 / 实时流槽位 | 2 / 3 | 2 / 3 |
| 流 worker 的 MPS | 项目私有 MPS | 项目私有 MPS |
| CPU/内存节点约束 | 必须准备 `.tools/prod-stream-affinity` | 沿用旧 worker；冷启动可读 `.tools/test-stream-affinity` |
| GPU 设备映射 | 单卡预留；容器内编号 0 | 流 worker/MPS 保留旧服务器全卡兼容映射 |

API、协调器和通知进程使用 CPU 镜像。文件 worker 与流 worker 共用所选显卡，每个 worker 同时执行一个任务；**3 个流槽位不等于这张卡一定能实时处理 3 路视频**。

本文第 2～12 节按正式环境说明；测试环境差异见第 13 节。脚本无参数时默认部署 **test/main**，正式部署始终明确写出 `--env prod`。

### 1.2 GPU 型号未定时，先过这三关

1. **架构及驱动**：GPU 必须支持所选 PyTorch、torchvision、ONNX Runtime CUDA 组合。新架构不能仅凭 `nvidia-smi` 成功就视为支持；需要执行第 9 节的非空 NMS 和真实模型预检。
2. **视频硬件能力**：核对该型号对目标输入格式的 NVDEC 和目标输出格式的 NVENC 支持。不要把推理算力、显存容量当作视频编解码支持。参考 [NVIDIA Video Codec SDK 与 GPU 支持矩阵入口](https://developer.nvidia.com/video-codec-sdk)。
3. **真实业务容量**：用实际分辨率、帧率、码率、音轨及并发数验收。部署成功不证明每路 15/25/30 FPS，也不证明没有持续积压。

如果选择 RTX 5090 等新架构，应先验证 CUDA 13.0 路径。当前正式入口仍固定 cu126，**在 `.env.compose` 中填写 `TORCH_BACKEND=cu130` 不会改变它**。若正式 cu126 预检不通过，需要先交付并验证适配该 GPU 的正式 Compose/依赖版本，不能把测试覆盖文件直接当作通用正式配置。

### 1.3 准备信息

部署前记录：系统版本、GPU 型号和完整 UUID、驱动版本、CPU/NUMA 拓扑、可用磁盘、部署标签、服务 IP、端口、Redis 地址/数据库、媒体源允许范围、模型包及其 SHA256。

本指南使用联网拉取源码和镜像、构建依赖，模型权重离线交付。完全断网环境还需要准备全部镜像/依赖缓存及离线部署流程；当前脚本包含 `git fetch` 和 `build`，不能直接视为离线安装器。

## 2. 准备操作系统和 NVIDIA 驱动

### 2.1 基础检查

```bash
cat /etc/os-release
uname -m
lscpu
free -h
df -h / /data
timedatectl status
```

架构应为 `x86_64`。`/data` 尚不存在时改为检查计划使用的挂载点。媒体目录应放在已挂载的数据盘上，并为输入、输出、诊断资料及 Docker 镜像预留空间；不要将临时系统盘默认当作持久存储。

Ubuntu 新机器安装基础工具：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg git python3 openssl \
  pciutils numactl ubuntu-drivers-common
```

宿主 Python 仅用于配置检查和验收命令；应用 Python 3.12 及运行依赖位于镜像内，宿主无需另行安装项目的 Python 环境。

### 2.2 安装与 GPU 匹配的驱动

已有可用驱动时先检查，不直接覆盖：

```bash
ubuntu-drivers list --gpgpu
nvidia-smi
```

尚未安装时，依据 GPU、Ubuntu 版本和目标 CUDA 构建，选择官方支持的驱动包；下面的包名必须替换为现场确认的实际包名：

```bash
DRIVER_PACKAGE='REPLACE_WITH_SUPPORTED_NVIDIA_DRIVER_PACKAGE'
sudo apt-get install -y "$DRIVER_PACKAGE"
sudo reboot
```

重启并重新登录后：

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version --format=csv
```

驱动安装及 Secure Boot 的模块签名处理以 [Ubuntu 官方 NVIDIA 驱动指南](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/index.html)为准；不要混用发行版包和 `.run` 安装器。

CUDA 12.6 GA 对应 Linux 驱动为 560.28.03 或更高；CUDA 13.x 的驱动兼容基线为 R580。它们是版本参考，不是对任意新卡、cuDNN 或具体 wheel 的完整兼容保证。新卡还需要包含该硬件支持的驱动分支，最终以实际预检为准。依据：[CUDA 12.6 发布说明](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/)、[CUDA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

视频库的 NVENC/NVDEC 接口也可能要求更高版本驱动，不能只按 CUDA 最低兼容值选驱动；完成模型预检后仍要执行实际视频和流任务。

`nvidia-smi` 显示的 CUDA Version 不是容器实际安装的 CUDA 版本。本项目容器带有 CUDA 用户态依赖，宿主不需要为了应用运行再装完整 CUDA Toolkit。

## 3. 安装 Docker、Compose 和 NVIDIA Container Toolkit

### 3.1 Docker Engine 与 Compose 插件

已有符合要求的 Docker 时保留，直接验证。全新 Ubuntu 可按以下方式配置 Docker 官方软件源；若发现系统已经装有其他来源的 Docker/containerd，应先按官方文档处理包冲突，不在已有业务机器上直接卸载。

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

安装方法见 [Docker 官方 Ubuntu 指南](https://docs.docker.com/engine/install/ubuntu/)。生产交付时记录实际安装版本，后续不要把宿主机自动升级与应用发布混为一次变更。

后续部署脚本直接调用 `docker`。可由具有 Docker 权限的部署账号执行；若采用 docker 用户组方式：

```bash
sudo usermod -aG docker "$(id -un)"
# 退出当前登录会话，重新登录后继续。
```

Docker 组具有宿主机高权限，只授予部署账号。重新登录后确认：

```bash
docker version
docker compose version
docker info
```

### 3.2 提供脚本要求的 `docker-compose` 命令

仓库脚本目前调用带连字符的 `docker-compose`。如果系统只有 `docker compose` 插件，安装一个转发入口；已有 `docker-compose` 时不覆盖，先确认它是支持所需功能的版本：

```bash
if ! command -v docker-compose >/dev/null 2>&1; then
  sudo tee /usr/local/bin/docker-compose >/dev/null <<'EOF'
#!/bin/sh
exec /usr/bin/docker compose "$@"
EOF
  sudo chmod 0755 /usr/local/bin/docker-compose
fi
docker-compose version
```

测试配置使用 `!override`，要求 Compose 2.24.4 或更高；仓库已有 Compose 2.29.2 的解析验证记录。Compose v1 不适用，其他版本仍须执行配置解析。依据：[Compose 合并规则](https://docs.docker.com/reference/compose-file/merge/)。

### 3.3 NVIDIA Container Toolkit

以下使用 NVIDIA 稳定软件源，不启用实验源：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info --format '{{json .Runtimes}}'
```

`nvidia-ctk` 会修改 Docker daemon 配置，重启 Docker 会影响宿主机其他容器；这一步应在新机初始化阶段完成。已有配置的机器先保留并检查原配置。安装依据：[NVIDIA Container Toolkit 官方指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

实际容器 GPU/算子验证在第 9 节使用应用自己的镜像完成，避免只通过另一个 CUDA 镜像误判应用环境。

## 4. 获取源码并准备持久目录

推荐将代码和数据分开：

```text
/opt/hos-vision-shield/              Git 工作目录
  .env.compose                     正式环境配置，权限 0600
  .tools/prod-stream-affinity       当前机器的正式 CPU/NUMA 目标
/data/hos-vision-shield/models/      离线权重、清单及许可证，只读挂载
/data/hos-vision-shield/data/        输入、输出、运行目录，UID/GID 10001 可写
```

以下以当前部署账号维护代码目录：

```bash
sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" /opt/hos-vision-shield
git clone https://github.com/Miaque/hos-vision-shield.git /opt/hos-vision-shield
cd /opt/hos-vision-shield
git status --short --branch

RELEASE_TAG='REPLACE_WITH_PUBLISHED_AND_VALIDATED_TAG'
git fetch origin "refs/tags/$RELEASE_TAG:refs/tags/$RELEASE_TAG"
git switch --detach "refs/tags/$RELEASE_TAG"
git rev-parse HEAD

sudo install -d -m 0755 /data/hos-vision-shield/models
sudo install -d -m 0750 -o 10001 -g 10001 /data/hos-vision-shield/data
```

使用已推送并包含当前部署能力的交付标签。候选标签名不自动代表正式验收通过；不要直接选历史实验标签。私有仓库先配置只读 Git 凭据，避免把令牌写入远程地址或文档。

容器应用运行用户是 `10001:10001`。新建媒体目录可按上述方式授权；已有数据盘需单独核对所有者及父目录可遍历权限，不用 `chmod -R 777` 解决权限问题。

## 5. 交付离线模型

在已有完整权重的准备机器、仓库根目录打包；Windows 可使用 `tar.exe` 执行相同参数：

```bash
tar -czf vision-shield-models.tar.gz -C models \
  face.jit license_plate.jit yunet_face.onnx hyperlpr_plate.onnx \
  manifest.json cpu-manifest.json README.md NOTICE.md licenses
sha256sum vision-shield-models.tar.gz > vision-shield-models.tar.gz.sha256
```

Windows 没有 `sha256sum` 时，用 PowerShell 的 `Get-FileHash -Algorithm SHA256` 记录并随包交付哈希，接收端计算后逐字比对。尚未准备模型时，可先在联网 Windows 准备机器执行仓库 `scripts/download-cpu-models.ps1`，再打包；不要让服务启动时临时下载。

将模型包及校验值传到目标服务器的约定交付目录。在该目录核验、解包：

```bash
sha256sum -c vision-shield-models.tar.gz.sha256
sudo tar -xzf vision-shield-models.tar.gz -C /data/hos-vision-shield/models
ls -l /data/hos-vision-shield/models
```

四个外部权重和两份清单必须位于模型目录第一层，运行用户可读，父目录可遍历。不要解成 `models/models/...`。

GPU 后端实际使用包内 YOLO 人脸、外部 EgoBlur 车牌、YuNet 和 HyperLPR；`face.jit` 保留给原 `quality` 后端。YOLO 权重随源码 wheel/镜像交付，不从外部模型目录寻找；它与其他模型的许可证不同，参见[包内模型说明](../src/vision_shield/detection/assets/README.md)。外部模型清单也必须保留，真实哈希检查在第 9 节执行。

## 6. 准备 Redis 和网络

### 6.1 Redis 是必需服务

应用 Compose **不创建 Redis**。准备一个从 Docker 网络可达、带认证、持久化且使用 `noeviction` 的 Redis，给本环境分配独立数据库，例如正式 `/6`、测试 `/7`。逻辑数据库不能代替账号或网络安全隔离。

可以复用已有 Redis；若新机器还没有 Redis，可先安装 Ubuntu 软件包：

```bash
sudo apt-get install -y redis-server redis-tools
```

由部署人员编辑 `/etc/redis/redis.conf`，按现场内网地址和容量设置。以下是必须核对的配置片段，不是用来覆盖整个配置文件：

```text
bind 127.0.0.1 REPLACE_WITH_REDIS_PRIVATE_IP
protected-mode yes
port 6379
requirepass REPLACE_WITH_STRONG_REDIS_PASSWORD
appendonly yes
appendfsync always
maxmemory-policy noeviction
```

按机器预算设置 `maxmemory`，为 AOF 和后台操作保留内存及磁盘；满内存写入失败时应告警处理，不能改成淘汰业务任务记录。项目参考配置见 [redis.compose.conf](redis.compose.conf)。Redis ACL 场景使用现有授权用户，并确认应用所需命令/键访问范围。

保存配置后启动并检查：

```bash
sudo systemctl enable redis-server
sudo systemctl restart redis-server
redis-cli -h REPLACE_WITH_REDIS_PRIVATE_IP -p 6379 --askpass PING
```

不要把 Redis 暴露到公网。宿主机只绑定 `127.0.0.1` 时，普通桥接网络容器不能用这个地址访问宿主 Redis；应使用经授权的宿主内网地址，并允许实际 Docker 网段访问。第 9 节会从应用容器再次验证认证与数据库选择。

### 6.2 端口和地址

| 链路 | 默认端口 | 允许范围 |
| --- | --- | --- |
| 业务调用方 → API | TCP 8000 | 可信业务网段或鉴权网关 |
| 播放客户端 → RTSP | TCP 8554 | 获准播放客户端；当前 RTSP 传输使用 TCP |
| Platform 内部代理 → HLS | TCP 8888 | 可信内部代理，配置 CDN 密钥 |
| 容器 → Redis | 按 Redis 配置 | 仅应用使用的 Docker/业务网段 |
| 通知进程 → RabbitMQ | 按 AMQP URL | 启用通知时才需要 |
| worker → 媒体源 | 按实际 RTSP/RTMP/HTTP(S) URL | 网络可达且在应用允许列表中 |
| MediaMTX 控制 API | 容器内 TCP 9997 | 不发布到宿主机 |

`BIND_ADDRESS` 必须是本机已有地址；`RTSP_PUBLIC_BASE_URL` 必须是播放客户端可达地址，不能填容器名。不要把裸 API/媒体端口直接公开到互联网；应用的媒体发布检查不等同于面向公网的 API 用户认证。

Docker 发布端口可能绕过普通 UFW 规则，应同时按网络边界和 Docker 防火墙链验证实际可达范围，见 [Docker 防火墙注意事项](https://docs.docker.com/engine/install/ubuntu/#firewall-limitations)。

## 7. 填写正式环境配置

回到仓库目录，复制模板而不覆盖已有配置：

```bash
cd /opt/hos-vision-shield
umask 077
cp -n deploy/compose.env.example .env.compose
chmod 0600 .env.compose
openssl rand -hex 32
```

将最后生成的随机值用于 HLS 内部代理密钥，妥善保存。编辑 `.env.compose`，下面是一份需要替换现场值的示例：

```dotenv
DEBUG=false
STREAM_OUTPUT_PACING=true
STREAM_E0_DIAGNOSTICS=true
MODEL_DIR=/data/hos-vision-shield/models
MEDIA_DIR=/data/hos-vision-shield/data
BIND_ADDRESS=10.20.0.10
API_PORT=8000
RTSP_PORT=8554
HLS_PORT=8888
RTSP_PUBLIC_BASE_URL=rtsp://10.20.0.10:8554
REDIS_URL='redis://:REPLACE_WITH_URL_ENCODED_PASSWORD@10.20.0.10:6379/6'
GPU_DEVICE_ID=GPU-REPLACE_WITH_FULL_UUID
SOURCE_ALLOWLIST='["10.20.0.50/32"]'
HLS_CDN_SECRET=REPLACE_WITH_GENERATED_RANDOM_SECRET
VISION_SHIELD_NOTIFICATION_AMQP_URL=
NOTIFICATION_TARGETS=[]
NOTIFICATION_LIMIT=1000
MEDIA_DISK_HIGH_WATERMARK=0.95
```

逐项核对：

- `MODEL_DIR`、`MEDIA_DIR` 必须已存在且是绝对路径；Compose 不自动创建这两个宿主目录。
- `GPU_DEVICE_ID` 从 `nvidia-smi` 复制一张卡的完整 UUID。不要复制旧服务器 UUID，不填多卡列表，也不要把 MIG 标识视为当前接口已支持的卡标识。
- Redis 密码中的 `@`、`:`、`/`、`#`、`%` 等需要 URL 编码；使用 ACL 用户时写成 `redis://用户名:编码密码@地址:端口/数据库`。可用 `python3 -c 'import getpass,urllib.parse; print(urllib.parse.quote(getpass.getpass("Redis password: "), safe=""))'` 生成编码值，不把明文密码作为命令行参数。
- `SOURCE_ALLOWLIST=[]` 适合先验收本地上传；使用 URL/实时源时必须授权实际主机及其解析 IP/CIDR，不要为方便设置无范围限制的允许规则。
- 不启用通知时保持 `NOTIFICATION_TARGETS=[]` 和空 AMQP URL；启用时配置独立通知目标及 RabbitMQ 认证/vhost。测试目标不能指向正式消费者。
- HLS 密钥至少 32 字符，并与 Platform 的 `VISION_SHIELD_HLS_CDN_SECRET` 一致。HLS 默认不提供匿名直接播放，具体对接见 [README 的内部 HLS 说明](../README.md#platform-内部-hls-对接)。
- 这里使用 `.env.compose`，不要用本机 `.env` 代替，也不要通过 `source .env.compose` 加载。已有 shell 导出的同名变量可能覆盖 Compose 环境文件，部署前一并核对。

环境变量修改后必须重新创建应用容器；仅 `docker restart` 不会重新注入 Compose 环境。

## 8. 准备 MPS 和正式 CPU/NUMA 约束

### 8.1 检查 MPS 工具

当前 GPU Compose 将宿主机以下驱动工具只读挂载进 MPS 控制容器：

```bash
test -x /usr/bin/nvidia-cuda-mps-control
test -x /usr/bin/nvidia-cuda-mps-server
```

缺少时，先查询发行版与当前驱动匹配的计算工具包：

```bash
apt-cache search '^nvidia-compute-utils-'
```

选择与已选驱动分支一致、提供这两个路径的实际包安装，再执行检查：

```bash
MPS_TOOLS_PACKAGE='REPLACE_WITH_MATCHING_NVIDIA_COMPUTE_UTILS_PACKAGE'
sudo apt-get install -y "$MPS_TOOLS_PACKAGE"
test -x /usr/bin/nvidia-cuda-mps-control
test -x /usr/bin/nvidia-cuda-mps-server
```

不能从其他服务器随意复制驱动二进制。GPU/MPS 支持仍需现场确认。部署脚本不安装这些工具，也不修改 GPU compute mode；不要另行启动全宿主共享 MPS 来代替项目私有实例。

MPS 仅用于流 worker，文件 worker 不加入其 PID/IPC。正式容器只映射一张卡，`CUDA_VISIBLE_DEVICES=0` 指容器内的编号，不是要求使用宿主机 0 号卡。编号规则见 [NVIDIA MPS 文档](https://docs.nvidia.com/deploy/topics/topic_5_2_1.html)。

### 8.2 核对 GPU 邻近 NUMA 节点

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv
nvidia-smi topo -m
numactl --hardware
lscpu -e=CPU,NODE,SOCKET,ONLINE
lspci -D | grep -i nvidia
for node in /sys/devices/system/node/node[0-9]*; do
  printf '%s: ' "$node"
  cat "$node/cpulist"
done
```

也可对照 `lspci -D` 中选定 GPU 的实际 PCI 地址读取内核信息，注意 sysfs 常用 `0000:65:00.0` 形式，不一定与 `nvidia-smi` 的域宽度相同：

```bash
GPU_PCI='REPLACE_WITH_ACTUAL_SYSFS_PCI_ADDRESS'
cat "/sys/bus/pci/devices/$GPU_PCI/numa_node"
```

根据 GPU 邻近节点、在线 CPU 和其他业务分配，确定流 worker 可用的 CPU/内存节点集合。内核返回 `-1` 时不能照抄为内存节点；需结合完整拓扑选择。单节点机器也填写实际 CPU 集合和节点，例如经核实后的内存节点 `0`。

### 8.3 写入部署目标

以下两个变量必须替换为现场核实值；不提供旧机器 CPU 列表作为默认：

```bash
cd /opt/hos-vision-shield
umask 077
mkdir -p .tools
WORKER_CPUS='REPLACE_WITH_ACTUAL_CPU_LIST'
WORKER_MEMS='REPLACE_WITH_ACTUAL_MEMORY_NODE_LIST'
printf '%s|%s\n' "$WORKER_CPUS" "$WORKER_MEMS" > .tools/prod-stream-affinity
chmod 0600 .tools/prod-stream-affinity
```

文件格式是单行 `CPU列表|内存节点列表`，列表支持逗号和范围；三个流 worker 应用相同集合，并非每个 worker 独占一套 CPU。文件缺失或格式无效时正式部署会拒绝；脚本还会在停止旧 worker 前用 CPU 容器检查集合是否可用。这个检查不能判断节点是否与 GPU 邻近或是否适合业务负载。

`.tools/` 被 Git 忽略，迁移目录、备份和回滚时需要单独保存此文件。不要在容器重建后临时手工绑定，再误以为下次部署仍会保留。

## 9. 构建前检查与独立预检

### 9.1 固定当前操作对象

后续正式环境的配置、日志和验收都复用这个 Bash 数组；重新登录后需要重新定义：

```bash
cd /opt/hos-vision-shield
dc=(docker-compose --env-file .env.compose -p vision-shield \
    -f compose.yaml -f compose.gpu.yaml)
"${dc[@]}" config --quiet
"${dc[@]}" config --services
```

预期服务为 `mediamtx`、`api`、`reconciler`、`notifier`、`file-worker`、`stream-worker`、`cuda-mps`。`config --quiet` 通过只证明配置可解析，不证明目录、驱动、Redis 或 GPU 可用。不要把含密码的完整 `config` 输出贴入工单或提交 Git。

确认代码是准备部署的标签，MPS 工具、模型和亲和性文件都已准备好，然后构建：

```bash
"${dc[@]}" build
"${dc[@]}" pull mediamtx
```

首次构建需要访问 Python/uv 基础镜像、PyTorch/Python 包源和 GHCR 制品。GHCR 私有制品按交付账号登录；代理配置须同时覆盖 Docker daemon 拉取与构建网络，不能只看宿主 `curl` 是否成功。

正式和测试镜像都使用项目发布的 **PyAV `19.0.2+hos.1` / FFmpeg `9.0.2`**；MediaMTX 为含 **gortsplib `5.6.7-hos.1` 的 `1.21.2`**。镜像 digest、wheel 哈希及补充锁已固定在仓库，不需要在服务器现编译 FFmpeg 或 gortsplib，也不要安装官方 PyAV 覆盖该补丁版本。来源及升级边界见 [正式媒体依赖发布说明](../specs/009-published-dependencies-parity/production-release.md)。

### 9.2 从容器验证数据目录与 Redis

```bash
"${dc[@]}" run --rm --no-deps api python -c \
  'import tempfile; f=tempfile.TemporaryFile(dir="/data"); f.write(b"ok"); f.close(); print("media write OK")'

"${dc[@]}" run --rm --no-deps api python -c \
  'import os; from redis import Redis; r=Redis.from_url(os.environ["REDIS_URL"],socket_connect_timeout=3,socket_timeout=3); assert r.connection_pool.connection_kwargs.get("db",0)==6; assert r.ping(); print("Redis DB6 PING OK")'
```

示例断言使用 DB6；若分配了其他数据库，应同步修改断言。任何一步失败都先修复，不通过放宽权限、关闭认证或改回默认 DB0 绕过。

### 9.3 验证 CUDA 构建、目标 GPU 与实际算子

先不请求 GPU，检查镜像依赖：

```bash
docker run --rm --network none vision-shield:cu126 python -c \
  'import torch,torchvision,av; assert torch.version.cuda=="12.6"; print(torch.__version__,torchvision.__version__,torch.version.cuda,av.__version__)'
```

再使用与 `.env.compose` 一致的完整 UUID 和模型路径，独立执行无网络预检：

```bash
SELECTED_GPU='GPU-REPLACE_WITH_FULL_UUID'
MODEL_DIR='/data/hos-vision-shield/models'
docker run --rm --network none --gpus "device=$SELECTED_GPU" \
  --mount "type=bind,source=$MODEL_DIR,target=/models,readonly" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e GPU_DEVICE_ID="$SELECTED_GPU" \
  -e DETECTION_BACKEND=quality_yolo -e INFERENCE_DEVICE=cuda \
  vision-shield:cu126 python -m vision_shield.detection.preflight
```

预期退出码 0，报告包含 `device=cuda`、唯一可见设备、GPU 名称/计算能力和已校验模型。入口会检查模型 SHA256、实际加载、合成帧推理和非空 CUDA NMS。CUDA 失败不得换卡或静默回退 CPU。

这一步只验证检测链，不验证 NVDEC/NVENC、Redis 业务写入、流发布、播放器或目标吞吐；第 11 节必须继续执行。

## 10. 首次启动正式服务

确保 `RELEASE_TAG` 仍是第 4 节选择的交付标签，再执行：

```bash
cd /opt/hos-vision-shield
bash scripts/deploy-server.sh --env prod --tag "$RELEASE_TAG"
"${dc[@]}" ps
"${dc[@]}" logs --tail 100 api file-worker stream-worker cuda-mps mediamtx
```

脚本会拉取并核对指定标签、解析配置、构建镜像、处理流 worker 与 MPS 生命周期，在模型启动前应用 CPU/内存约束，最后等待就绪，等待阶段最多 240 秒。前面执行过独立构建时，依赖未变应复用构建缓存。

首次启动和日常更新都使用该入口，不用普通 `compose up -d` 替代其 MPS/NUMA 顺序。若从旧版本升级，先切换到新入口代码再调用；脚本在 Git 更新前已经解析函数，不能依赖旧入口自更新后在同一次调用中采用新逻辑。

启动后检查：

```bash
API='http://10.20.0.10:8000'
curl --fail --silent --show-error --noproxy '*' "$API/health/live"
curl --fail --silent --show-error --noproxy '*' "$API/health/ready"

mapfile -t file_ids < <("${dc[@]}" ps -q file-worker)
mapfile -t stream_ids < <("${dc[@]}" ps -q stream-worker)
test "${#file_ids[@]}" -eq 2
test "${#stream_ids[@]}" -eq 3
```

将 `API` 换为实际绑定地址。`/health/live` 只证明 API 存活；`/health/ready` 检查 Redis、两个 worker 池的有效心跳和 MediaMTX 控制 API，不证明全部 5 个副本或实际播放都正常。逐个检查 worker 日志，不要用副本总数替代模型预检结果。

核对流 worker 实际约束，使用 `/proc/self/status` 检查新执行进程，避免把共享 PID 命名空间中的 PID1 当作该 worker：

```bash
for id in "${stream_ids[@]}"; do
  docker inspect --format '{{.Name}} {{.HostConfig.CpusetCpus}}|{{.HostConfig.CpusetMems}}' "$id"
  docker exec "$id" sh -c 'grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/self/status'
done
"${dc[@]}" exec -T cuda-mps sh -c 'echo get_server_list | nvidia-cuda-mps-control'
```

MPS 健康不表示业务已经使用 MPS；空闲时可没有客户端，运行真实流任务后再核对。目录、PID/IPC 及设备映射以当前 Compose 为准，不接入其他项目的 MPS。

再核对运行容器的媒体依赖：

```bash
for id in $("${dc[@]}" ps -q api file-worker stream-worker); do
  docker exec "$id" python -c 'import av; from importlib.metadata import version; assert av.__version__==version("av")=="19.0.2+hos.1"; assert av.ffmpeg_version_info=="9.0.2"; assert hasattr(av.container.OutputContainer,"flush_mux"); print(av.__version__,av.ffmpeg_version_info)'
done
docker inspect --format '{{.Config.Image}} {{.Image}}' "$("${dc[@]}" ps -q mediamtx)"
```

记录实际镜像 ID/digest 与 Git 提交。只检查磁盘中的源码、镜像标签或 `pip list`，不能证明当前运行进程已经使用新版本。

## 11. 执行真实媒体验收

### 11.1 准备证据目录与素材

使用有权处理、包含可人工核对人脸和车牌的图片、H.264 视频及带音轨的 RTSP 源。不要用原始敏感资料公开测试。

```bash
umask 077
ACCEPTANCE_DIR=".tools/acceptance-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ACCEPTANCE_DIR"
IMAGE_SAMPLE='/REPLACE_WITH_AUTHORIZED_IMAGE.jpg'
VIDEO_SAMPLE='/REPLACE_WITH_AUTHORIZED_H264_VIDEO.mp4'
sha256sum "$IMAGE_SAMPLE" "$VIDEO_SAMPLE" > "$ACCEPTANCE_DIR/input-sha256.txt"
git rev-parse HEAD > "$ACCEPTANCE_DIR/git-commit.txt"
"${dc[@]}" images > "$ACCEPTANCE_DIR/images.txt"
```

### 11.2 图片与离线视频

```bash
curl --fail --silent --show-error --noproxy '*' \
  -F kind=image -F redaction_mode=mosaic -F "file=@$IMAGE_SAMPLE" \
  "$API/v1/file-tasks/uploads" > "$ACCEPTANCE_DIR/image-created.json"

curl --fail --silent --show-error --noproxy '*' \
  -F kind=video -F redaction_mode=mosaic -F "file=@$VIDEO_SAMPLE" \
  "$API/v1/file-tasks/uploads" > "$ACCEPTANCE_DIR/video-created.json"
```

分别从响应读取任务 ID，然后重复查询至 `succeeded`，不能将 HTTP 202 当作处理完成：

```bash
TASK_ID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' \
  < "$ACCEPTANCE_DIR/video-created.json")
curl --fail --silent --show-error --noproxy '*' \
  "$API/v1/tasks/$TASK_ID" > "$ACCEPTANCE_DIR/video-status.json"
# 确认 state=succeeded 后下载；图片任务同理，结果后缀用 jpg。
curl --fail --silent --show-error --noproxy '*' \
  "$API/v1/tasks/$TASK_ID/result" -o "$ACCEPTANCE_DIR/$TASK_ID.mp4"
```

检查输出可解码、时长正常、可见人脸/车牌遮挡符合预期、没有明显误遮/闪烁/拖尾；同时检查视频声音与同步。至少补测一次 `solid` 模式。一次图片成功不能替代视频编解码和连续跟踪验收。

### 11.3 实时流、音频与实际播放

确认源地址可从容器网络读取，并已加入 `SOURCE_ALLOWLIST`。不要将原始验证源发布到受应用发布鉴权保护的 `redacted/` 路径。

```bash
curl --fail --silent --show-error --noproxy '*' \
  -H 'Content-Type: application/json' \
  -H 'X-Detection-Interval: 3' -H 'X-Tracking-Mode: flow' \
  -d '{"url":"rtsp://10.20.0.50:8554/authorized-test","redaction_mode":"mosaic","max_bitrate":5000000,"audio_enabled":true}' \
  "$API/v1/stream-tasks" > "$ACCEPTANCE_DIR/stream-created.json"

STREAM_TASK_ID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' \
  < "$ACCEPTANCE_DIR/stream-created.json")
curl --fail --silent --show-error --noproxy '*' \
  "$API/v1/tasks/$STREAM_TASK_ID" > "$ACCEPTANCE_DIR/stream-status.json"
```

`max_bitrate` 单位为 bit/s。音频验收保留 `audio_enabled=true`，不要用关掉音轨的结果替代音视频验证。等待任务 `ready=true` 且健康正常后，读取响应中的 `playback_url`，在具备播放能力的客户端执行：

```bash
ffplay -rtsp_transport tcp 'REPLACE_WITH_RESPONSE_PLAYBACK_URL'
```

实际观察画面连续性、声音和音画同步。API 的 `ready` 和 worker 的 mux 成功不证明播放器已收到或渲染；HLS 则通过配置好密钥和用户权限的 Platform 内部代理验证，不能以匿名直连 8888 失败判断整体故障。

停止验收流：

```bash
curl --fail --silent --show-error --noproxy '*' -X PATCH \
  -H 'Content-Type: application/json' -d '{"desired_state":"stopped"}' \
  "$API/v1/tasks/$STREAM_TASK_ID"
curl --fail --silent --show-error --noproxy '*' "$API/v1/tasks/$STREAM_TASK_ID"
```

继续查询，直到状态为 `stopped`，并确认输出会话和处理进程结束。PATCH 返回 202 只表示接纳停止请求。

### 11.4 并发和持续运行

按计划容量创建不同任务，确认它们实际同时处理，再至少做一轮 10 分钟的同素材或实际来源验收。分别记录每路输入/有效输出帧率、积压是否增长、最大停顿、音画偏差、遮挡效果、CPU/显存/磁盘使用，并验证文件任务同时运行的资源影响。短片或单路结果不能外推多路容量。

```bash
docker stats --no-stream
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv
nvidia-smi dmon -s u -c 10
df -h /data /var/lib/docker
```

结合真实任务检查所选 GPU、解码和编码活动。只看到 GPU 利用率或模型预检通过，不证明全链路采用预期硬件路径；必要时在独立测试环境开启 DEBUG，使用[诊断说明](../specs/002-compose-cpu-gpu/diagnostics.md)核对实际 codec、逐帧时序及接收证据。

验收记录至少包含：

- 系统、驱动、GPU UUID、CPU/NUMA、代码提交及镜像 ID/digest。
- 模型包/输入哈希、请求参数、任务 ID、输出和实际播放记录。
- 真实并发数与每路持续吞吐，记录失败与未执行项，不沿用旧服务器成绩。
- 图片/视频/流的遮挡效果，音频、同步、长时间播放和源断开后的恢复情况。
- 在验收任务上验证一次 worker 重启；显式停止的流不复活，运行意图的流按既有机制尝试恢复。
- 服务重启后已完成结果仍可查询/下载，数据与 Redis 持久化正常。

重启会中断媒体任务，应使用独立验收任务或维护窗口。未完成的文件任务不会断点续跑；流恢复需要 Redis 状态、reconciler 和源/输出都可用。

## 12. 日常更新、备份和回滚

### 12.1 更新前保存恢复资料

先处理运行中的文件任务，安排流中断窗口。按现有备份机制保存 Redis 与媒体数据的一致快照；不要在 Redis 正在写入时随意复制部分 AOF 文件冒充可恢复备份。

在部署目录保存配置与运行版本，限制目录权限：

```bash
umask 077
BACKUP_STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR=".tools/deploy-backup-$BACKUP_STAMP"
mkdir -p "$BACKUP_DIR"
cp .env.compose .tools/prod-stream-affinity "$BACKUP_DIR/"
git rev-parse HEAD > "$BACKUP_DIR/git-commit.txt"
git describe --tags --always > "$BACKUP_DIR/version.txt"
"${dc[@]}" images > "$BACKUP_DIR/images.txt"
for service in api file-worker mediamtx; do
  mapfile -t ids < <("${dc[@]}" ps -q "$service")
  if [[ "${#ids[@]}" -eq 0 ]]; then
    printf '未找到运行中的服务，备份未完成：%s\n' "$service" >&2
    break
  fi
  image_id=$(docker inspect --format '{{.Image}}' "${ids[0]}")
  docker image tag "$image_id" "vision-shield-backup:$BACKUP_STAMP-$service"
done
docker image inspect "vision-shield-backup:$BACKUP_STAMP-api" \
  "vision-shield-backup:$BACKUP_STAMP-file-worker" \
  "vision-shield-backup:$BACKUP_STAMP-mediamtx" >/dev/null
```

上面的三个服务分别用于取得当前 CPU/GPU/MediaMTX 镜像，备份标签使用时间戳；若服务不存在或命令失败，先补齐备份，不继续更新。保存已确认可用的原正式代码标签，并将旧镜像保留至新版本验收结束。镜像缓存可能被清理，重要恢复点还需按交付方式导出或推送至受控镜像仓库。备份内含凭据及媒体资料，不提交 Git、不公开上传。

### 12.2 按固定标签更新

```bash
NEXT_TAG='REPLACE_WITH_NEXT_VALIDATED_TAG'
bash scripts/deploy-server.sh --env prod --tag "$NEXT_TAG"
curl --fail --silent --show-error --noproxy '*' "$API/health/ready"
"${dc[@]}" ps
```

指定分支更新也受支持，例如 `bash scripts/deploy-server.sh --env prod main`，但它部署的是执行时远端分支最新提交，不是固定版本。

脚本拒绝未提交的跟踪文件修改、本地额外提交/分叉、冲突的未跟踪文件和被改写的同名 tag，不执行强制 reset。构建成功后才进入容器替换；前面的 Git 切换可能已经发生，构建失败不等于工作目录仍在旧版本。日常更新不先 `down`，不使用 `--no-cache`，不清理其他项目或独立回放源。

配置、依赖和入口变更都要重新检查；更新后至少重复就绪、实际运行库、MPS/约束和短程媒体回归，不能只看脚本打印“部署完成”。

### 12.3 回滚

若原正式标签包含同样的 MPS/NUMA 部署能力，可用以下方式恢复：

```bash
PREVIOUS_TAG='REPLACE_WITH_PREVIOUS_WORKING_PRODUCTION_TAG'
bash scripts/deploy-server.sh --env prod --tag "$PREVIOUS_TAG"
curl --fail --silent --show-error --noproxy '*' "$API/health/ready"
"${dc[@]}" ps
```

如果本次改过环境或亲和性，先恢复与原版本匹配的配置，再执行入口。脚本不自动备份、不自动回滚；回滚仍需要源码/镜像可获取，并确认真实运行版本和媒体输出已恢复。

回到不含 MPS 的旧版本涉及旧流客户端和控制容器的退出顺序，不能只套用上面的命令。按[正式发布说明的回退步骤](../specs/009-published-dependencies-parity/production-release.md#回退)操作，不使用 `--remove-orphans` 批量清理。不要运行 `down -v`、`docker system prune` 或清空 Redis 来解决启动问题。

## 13. 如需在新机同时部署独立测试环境

测试路径沿用 CUDA 13.0 和旧服务器兼容设置，**不是任意 GPU 的通用推荐配置**。迁移前检查 `compose.test.yaml`：

- `group_add: 1022` 是旧宿主机设备附加组，先根据新机设备权限核对，而不是把新机所有设备组改成 1022。
- 流 worker/MPS 的 `count: all` 会授予所有 GPU 访问权；`CUDA_VISIBLE_DEVICES=所选UUID` 只限制应用计算选择，不构成设备隔离。多卡或共享服务器须先验证是否仍需要该兼容项，不能将它带入正式环境冒充单卡隔离。
- CUDA 13.0 镜像还需要与新 GPU、驱动、ONNX Runtime 兼容，不能从正式 cu126 结果推断测试路径已通过。

确认测试配置适用于目标机器后，创建独立配置：

```bash
cp -n deploy/compose.env.example .env.compose.test
chmod 0600 .env.compose.test
```

为测试环境分配独立 API/RTSP/HLS 端口，例如 `18000/18554/18888`，同步修改公开 RTSP URL；媒体目录与 Redis 数据库独立，GPU 按现场资源分配，通知目标为空或使用测试专属目标。模型目录可只读共享。

新环境没有旧 worker 约束且没有 `.tools/test-stream-affinity` 时不会绑定 NUMA。需要绑定时，按第 8 节的新机拓扑准备该测试文件；已有 worker 时脚本优先保存其实际约束，修改目标文件不会覆盖优先读取的旧约束，不要据此误判调整已经生效。

```bash
tdc=(docker-compose --env-file .env.compose.test -p vision-shield-test \
     -f compose.yaml -f compose.gpu.yaml -f compose.test.yaml)
"${tdc[@]}" config --quiet
bash scripts/deploy-server.sh --env test --tag "$RELEASE_TAG"
"${tdc[@]}" ps
```

验收命令使用测试数组、测试 API 地址和 cu130 镜像，Redis 数据库断言同步修改。操作哪套环境就始终使用该套项目名、文件组合和端口，不混用正式 `dc` 与测试 `tdc`。

## 14. 常见故障定位

| 现象 | 优先检查 |
| --- | --- |
| `docker-compose: command not found` | 第 3.2 节转发入口是否安装、PATH 是否包含其目录；不要安装 Compose v1 |
| `!override` 无法解析 | Compose 版本是否至少 2.24.4，是否误用了旧二进制 |
| 宿主 `nvidia-smi` 成功，容器 CUDA 不可用 | NVIDIA runtime、驱动/架构匹配、GPU UUID、容器真实设备映射；执行应用独立预检 |
| `no kernel image` / CUDA NMS 失败 | GPU 架构与 torch/torchvision/ONNX Runtime 组合；不能只替换某一个包后跳过效果回归 |
| MPS 容器无法创建/启动 | 两个宿主 MPS 工具路径及执行权限、媒体运行目录权限、所选 GPU 与 MPS 支持 |
| 正式部署拒绝 CPU/内存约束 | 文件名是否为 `prod-stream-affinity`、单行格式、在线 CPU/节点是否存在；不要复制旧服务器编号 |
| 模型预检报角色/清单/哈希错误 | 权重是否解包到第一层，两份清单是否完整，UID10001 是否可读，包内 YOLO 资产是否齐全 |
| API 无法访问 | 实际 IP/端口、`BIND_ADDRESS` 是否属于本机、端口冲突、网关/防火墙和代理；使用 `--noproxy '*'` 排除 HTTP 代理影响 |
| live 成功、ready 失败 | 从容器验证 Redis，查看两池 worker 预检/心跳和 MediaMTX 控制接口；ready 不是仅 API 自检 |
| 远程 URL/流被拒绝 | `SOURCE_ALLOWLIST` 的主机及实际解析 IP/CIDR、容器网络、源认证和协议；不放宽到所有地址 |
| HLS 直接访问失败 | 是否经 Platform 内部代理、密钥一致及用户/设备权限；默认不是匿名 HLS 服务 |
| 修改环境后行为没变 | 容器是否重新创建，当前项目/文件组合是否正确，有无同名 shell 环境变量覆盖 |
| 任务排队不处理 | 两池有效 worker、任务状态、Redis 写入/容量、文件与流各自槽位；容器 running 不代表完成预检 |
| 画面卡顿或延迟持续增大 | 实际源帧率、处理及接收帧率、NVDEC/NVENC、GPU/CPU/NUMA、音频时钟与输出节奏；按单路再并发核对 |
| DEBUG 接口返回 404 / profiling 返回 422 | `DEBUG=false` 下是预期行为；诊断时在受控环境改为 true 并重新创建 API/worker，结束后恢复 |
| 构建中 GHCR/包下载失败 | 镜像读取权限、DNS/代理、磁盘与网络；保留固定 digest/哈希，不临时换成未知镜像或放宽依赖锁 |

## 15. 交付前逐项确认

- [ ] 系统架构、GPU 架构/驱动和 NVDEC/NVENC 能力已核实，记录实际版本。
- [ ] Docker/Compose/NVIDIA runtime 正常，部署账号具有所需权限。
- [ ] 固定代码标签和镜像来源可追溯，模型包哈希正确，无运行时权重下载。
- [ ] 数据目录持久且 UID/GID10001 可写；模型只读可读。
- [ ] Redis 从应用容器认证成功，数据库、持久化、容量与淘汰策略正确。
- [ ] `.env.compose` 和密钥受保护；IP/端口、来源允许列表、HLS 与通知环境均正确。
- [ ] MPS 工具齐全，正式 CPU/NUMA 文件对应新机拓扑，实际内核约束已核对。
- [ ] 2 个文件和 3 个流 worker 均正常，独立 GPU 预检和 ready 通过。
- [ ] 图片、离线视频、实时音视频及实际客户端播放通过。
- [ ] 目标并发持续运行、遮挡效果、恢复和数据保留通过；未通过项明确列出。
- [ ] 原配置/数据/镜像和可用标签已备份，更新及回滚步骤可执行。

首次部署完成的判断是“目标机器上的业务链路通过验收”，不是单独一项健康检查或构建成功。后续配置变更继续以当前 Compose 和部署入口为准；旧服务器实验编号和历史成绩仅供排查参考。
