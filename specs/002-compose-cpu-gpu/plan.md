# 实施计划：服务器 CPU/GPU 部署与离线模型

**实际分支**：main（未创建或切换） | **日期**：2026-10-19 | **规格**：[spec.md](spec.md)

**状态**：服务器 Compose 规划暂缓。2026-10-19 用户明确服务器/GPU 尚未确定，优先实施 Windows 本机 GTX 1080 GPU；以下服务器设计保留为草案，不作为本次本地改动的交付门槛。本机进展见 [local-gpu.md](local-gpu.md)。

## 摘要

复用现有六个运行角色，以 CPU Compose 为基础、GPU 覆盖配置调整 worker。模型目录独立只读挂载，媒体目录和 Redis 数据分别持久化。设备选择独立于检测策略，worker 通过模型校验和真实推理探测后才登记心跳。分离容器内部推流地址与客户端播放地址。

用户确认目标为 aarch64 + NVIDIA。具体平台仍未确定，不指定未经核实的 CUDA 镜像标签或 GPU 依赖版本。

## 技术背景

| 项目 | 核实结果 |
| --- | --- |
| 语言 | Python 3.12，项目限制 >=3.12,<3.13 |
| 依赖 | FastAPI、Taskiq 0.12.4、taskiq-redis 1.2.2、PyAV、OpenCV、PyTorch、torchvision |
| 当前锁定 | CPU torch 2.14.1+cpu、torchvision 0.29.1+cpu；不是 CUDA 可用性证据 |
| 存储 | Redis 保存任务、消息和租约；共享媒体目录保存输入和结果 |
| 测试 | pytest 单元测试、显式真实模型探测、目标服务器 Compose 验收 |
| 范围 | 单服务器，每类任务池默认一 worker，每 worker 并发一任务 |
| 性能 | 六组媒体验证和重复计时，不承诺加速倍数或实时帧率 |
| 目标 | aarch64 + NVIDIA；操作系统、GPU 型号、驱动和显存待核实 |

NEEDS CLARIFICATION：目标是 ARM 服务器还是 Jetson，具体系统、GPU 型号、驱动或 JetPack 版本是什么？据此才能确定 ARM64 GPU 制品、Python 3.12 兼容性和支持矩阵。

## 宪章检查

| 原则 | 研究前检查 | 当前设计检查 |
| --- | --- | --- |
| 脱敏与契约 | 保留现有媒体类型和失败语义 | 不新增业务 API，沿用 live/ready 与任务状态 |
| 异步非阻塞 | 不在事件循环加载模型 | 哈希读取放线程，模型探测放短命子进程 |
| 模块边界 | 复用配置、检测工厂、任务监督 | HTTP 不接触 GPU 库，预检复用检测工厂 |
| 最小实现 | 不更换模型与阈值 | ONNX 保留 CPU，候选 GPU 路径仅迁移 EgoBlur |
| 可验证与资源生命周期 | 分开模拟、CPU、GPU 证据 | 预检失败不登记心跳；验证停止与重启释放资源 |

未发现需要豁免的宪章冲突，但硬件调查未解决，Phase 0 出口未通过，Phase 1 最终检查未执行。

## 项目结构

当前文档：spec.md、plan.md、research.md、checklists/requirements.md。

待调查完成后生成：data-model.md、contracts/deployment.md、quickstart.md。tasks.md 属于后续 speckit-tasks，本轮不创建。

预期实施修改范围：

- deploy/：Compose、Dockerfile、容器专用 Redis/MediaMTX 配置及环境示例。
- .dockerignore：排除模型权重、数据、.env、.git、虚拟环境。
- src/vision_shield/config.py：独立设备设置、内部推流地址。
- detection/__init__.py、quality.py、egoblur.py：设备传递，模型与输入同步选择设备。
- detection/preflight.py：拟新增最小预检入口。
- jobs/worker.py：预检成功后启动 Receiver 和心跳。
- media/publisher.py：使用内部推流地址。
- tests/unit/、tests/integration/：配置、设备、离线预检、地址和真实设备验证。
- README.md、models/README.md、.env.example：同步部署与模型说明，保留 Windows CPU 用法。

pyproject.toml/uv.lock 的修改方式待 GPU 制品确定；禁止普通依赖同步把 GPU 镜像中的 torch 覆盖为 CPU 构建。

## 已明确的设计

### 服务与存储

六个服务为 api、reconciler、file-worker、stream-worker、redis、mediamtx，复用现有 Python 入口。应用角色共享媒体目录的同一容器绝对路径；Redis 单独持久化并保留 AOF always/noeviction。模型只读挂载，不随镜像分发。

仅业务端口和 RTSP TCP 端口对外发布，Redis 和 MediaMTX 控制接口留在容器网络。保留发布鉴权、禁止发布者覆盖和禁录像。固定项目名与持久卷，日常停止不删卷。

API ready 依赖两个 worker；worker 不可反向等待 API healthy，避免启动死锁。容器设置 init 和足够停止宽限期，并在 Linux 实测 SIGTERM、子进程退出及重启恢复。

### 模型设备与预检

候选配置 INFERENCE_DEVICE=cpu|cuda，默认 cpu；DETECTION_BACKEND 保持算法语义。quality 的 EgoBlur 迁移目标设备，YuNet/HyperLPR 继续 CPU。单独 egoblur 保留现有模型路径语义；轻量 cpu 后端搭配 cuda 明确拒绝，避免假 GPU 模式。

EgoBlur 的模型与输入 tensor 必须同设备，保持输入格式、阈值、NMS、分块、局部复核和后处理；复用现有输出转 CPU 的接口。

复用 models/manifest.json 与 cpu-manifest.json，按后端和角色验证必需文件身份、可读性和 SHA-256。短命子进程加载模型并实际执行推理及 NMS，成功退出释放资源后才启动 worker 心跳。预检要有超时及失败清理，诊断包含文件/角色或设备原因，不暴露凭据。禁止下载、模型降级或设备回退。

两个 worker 可能同时持有模型，显存需求须按并发验证，不能以单进程加载成功代替。

### 网络地址

新增内部推流地址，未指定时回退现有 public 地址以兼容本机。Compose 内部使用 rtsp://mediamtx:8554；RTSP_PUBLIC_BASE_URL 仍用于返回客户端可达的服务器地址。控制地址和发布鉴权回调使用容器服务名，保留媒体源 allowlist。

### 验证步骤

1. 单元测试设备默认值、非法组合、工厂传参、内部/外部地址，以及失败预检不登记心跳。
2. 真实 CPU 校验正确模型及缺失、坏哈希、不可读、不可加载；禁止外部模型访问后处理仍成功。
3. 检查 ARM64 基础镜像与二进制依赖，确认镜像不含模型、数据和本机 .env；构建阶段不依赖 GPU。
4. 目标 GPU 执行模型、输入、NMS 与输出探测，记录实际设备和双 worker 峰值显存。
5. 完成规格六组效果、重复耗时、流帧率与丢帧记录。
6. 验证重启持久化、客户端播放、任务停止与资源释放。

每次文件任务都会创建子进程并重新加载模型；后续请求不等于常驻模型热推理。必须区分模型加载、处理和端到端时间。现有 Windows 资源测量不能直接作为 Linux 证据。

## 规划出口

补齐服务器信息并核实 GPU 制品后完成 research.md，再生成 Phase 1 设计及验收命令并复检宪章。目前不标记计划完成。

## 复杂度记录

无新增数据库实体、常驻模型服务、进程管理器或多 GPU 调度。不得通过添加多套未经验证的镜像来绕过兼容性核实。
