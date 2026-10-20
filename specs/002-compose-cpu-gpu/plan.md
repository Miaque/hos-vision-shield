# 实施计划：服务器 CPU/GPU 部署与离线模型

**实际分支**：main（未创建或切换） | **日期**：2026-10-19 | **规格**：[spec.md](spec.md)

**状态**：已实施 Compose、双模式 Dockerfile、GPU 生成锁、模型预检、地址分离、Linux 停机处理与计时日志，并执行本地回归。镜像构建和目标服务器验收待用户在既有环境执行；不在本机安装 Docker。完成范围见 [tasks.md](tasks.md) 与 [实施记录](validation/gates.md)。Git 实际仍为 main，没有创建或切换分支。

## 摘要

复用现有六个运行角色，以 CPU Compose 为基础、GPU 覆盖配置调整 worker。一个 Dockerfile 生成 CPU/cu126 镜像，保留根依赖锁与 Windows 安装方式，额外生成 GPU 部署锁。模型独立只读挂载，媒体和 Redis 分别持久化；worker 通过完整预检才发布心跳。分离内部推流与外部播放地址，并补齐 Linux 停机回收。

用户提供命令输出确认目标为 Ubuntu 24.04.4 LTS、x86_64、7 张 NVIDIA RTX 6000 Ada Generation（每张 49140 MiB）、驱动 595.84。用户确认先指定一张卡供文件与流 worker 共用，不做多卡调度；具体卡号或 UUID 在部署时指定，不扩展 ARM64 或 Jetson 支持。

## 技术背景

| 项目 | 核实结果 |
| --- | --- |
| 语言 | Python 3.12，项目限制 >=3.12,<3.13 |
| 依赖 | FastAPI、Taskiq 0.12.4、taskiq-redis 1.2.2、PyAV、OpenCV、PyTorch、torchvision |
| 当前锁定 | Windows AMD64 选择 CUDA 12.6 的 torch 2.14.1 / torchvision 0.29.1；Linux 等其他平台仍选 CPU 源，不能直接用于服务器 GPU 镜像 |
| 镜像依赖 | CPU 维持原 uv.lock；GPU 使用从根声明与共同版本约束生成的 deploy/requirements-cu126.txt，torch 2.14.1+cu126 / torchvision 0.29.1+cu126 |
| 基础镜像 | python:3.12-slim-bookworm；构建使用 uv 0.13.0；Redis 7.4.3、MediaMTX 1.21.2；已核验 amd64 元数据，digest 见 research.md |
| 存储 | Redis 保存任务、消息和租约；共享媒体目录保存输入和结果 |
| 测试 | pytest 单元测试、显式真实模型探测、目标服务器 Compose 验收 |
| 范围 | 单服务器，GPU 模式显式指定一张卡供两池共用；每类任务池默认一 worker，每 worker 并发一任务 |
| 性能 | 六组媒体验证和重复计时，不承诺加速倍数或实时帧率 |
| 目标 | 用户提供 Ubuntu 24.04.4 LTS、x86_64、7 × RTX 6000 Ada Generation、每张 49140 MiB、驱动 595.84 |
| Compose | 已有 v2.29.2，仅 `docker-compose` 可用；`docker compose` 不可用 |
| 项目类型 | 单服务器异步媒体处理服务，不新增数据库、业务接口或常驻模型服务 |
| 约束 | 模型全离线、GPU 失败不回退、单卡两池各并发 1；预检 120 秒、容器停止宽限 60 秒 |

需求与软件栈已明确。CUDA 12.6.3 采用官方对应的 Linux 驱动 560.35.05 作为交付版本门槛，用户报告的 595.84 满足版本关系；这不是容器运行证明。Docker Engine、NVIDIA 容器运行时、sm_89/NMS 与双 worker 容量作为部署验收项，不再列为待澄清决策。性能仅按同素材实测，不承诺固定帧率或加速倍数。

## 宪章检查

| 原则 | 研究前检查 | 当前设计检查 |
| --- | --- | --- |
| 脱敏与契约 | 保留现有媒体类型和失败语义 | 不新增业务 API，沿用 live/ready 与任务状态 |
| 异步非阻塞 | 不在事件循环加载模型 | 哈希、加载及模型/NMS 探测均放有总超时的短命子进程 |
| 模块边界 | 复用配置、检测工厂、任务监督 | HTTP 不接触 GPU 库，预检复用检测工厂 |
| 最小实现 | 不更换模型与阈值 | 复用既有 EgoBlur GPU、ONNX CPU 路径和 runner，不新增监督服务 |
| 可验证与资源生命周期 | 分开模拟、CPU、GPU 证据 | 预检失败不登记心跳；验证停止与重启释放资源 |

研究阶段检查针对设计，不将 resolver dry-run 或文件检查视为运行测试通过。实施前分析指出并发压力验收未落实为明确任务（C1），用户要求本轮跳过；完整链路零模型下载观测方案（U2）也按用户要求跳过。这两项仍无完整证据，不能据本地回归声明宪章及全部验收已通过，状态保留在实施记录。

## 项目结构

```text
specs/002-compose-cpu-gpu/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── contracts/deployment.md
├── quickstart.md
├── tasks.md
└── checklists/requirements.md
```

tasks.md 已生成并按实际执行情况更新。保留已有 local-gpu.md 等历史验证记录。实施修改范围：

- compose.yaml、compose.gpu.yaml：六服务 CPU 基础部署与单卡 GPU 覆盖。
- deploy/：单个 Dockerfile、GPU 生成锁、容器专用 Redis/MediaMTX 配置及 compose.env.example。
- .dockerignore：排除模型权重、数据、.env、.git、虚拟环境。
- src/vision_shield/config.py：增加内部推流地址；复用已有独立设备设置。
- detection/__init__.py、quality.py、egoblur.py：复用已实现的设备传递与模型/输入设备选择；仅在容器验证发现必要问题时修改。
- detection/preflight.py：按现有清单核验模型并实际推理，供内部 probe 与独立离线验证复用。
- media/child.py：新增内部 probe 分支；Linux 在模型加载前监视父管道 EOF，Windows 保留当前顺序。
- jobs/worker.py、jobs/tasks.py、jobs/reconcile.py：预检后发布心跳，接入 Linux 停止事件并回收任务资源。
- media/publisher.py：使用内部推流地址。
- tests/unit/、tests/integration/：配置、设备、离线预检、地址和真实设备验证。
- README.md、models/README.md、.env.example：同步部署与模型说明，保留 Windows CPU/GPU 用法。

根 pyproject.toml/uv.lock 不改。GPU 锁从根声明生成，共同依赖以原锁导出结果约束；构建空环境时一次同步 GPU 锁，再无依赖安装项目，运行时直接调用 Python，不执行根 uv sync。详细命令和实测证据见 [research.md](research.md)。

## 已明确的设计

### 服务与存储

六个服务为 api、reconciler、file-worker、stream-worker、redis、mediamtx，复用现有 Python 入口。应用角色共享媒体目录的同一容器绝对路径；Redis 单独持久化并保留 AOF always/noeviction。模型只读挂载，不随镜像分发。

CPU 基础配置为四个应用服务显式设置 `build.args.TORCH_BACKEND: cpu`；GPU 覆盖配置同时将两个 worker 的镜像设为 `vision-shield:cu126`、`build.args.TORCH_BACKEND` 设为 `cu126`，API/reconciler 保持 CPU 镜像和构建参数。验证合并配置后，还须检查 CPU 镜像的 `torch.version.cuda is None`、GPU 镜像的 `torch.version.cuda == "12.6"`，这两项不要求实际显卡。

GPU 覆盖配置通过设备预留中的 device_ids 指定同一张卡给两个 worker，不使用 all，不同时配置 count；同一 GPU_DEVICE_ID 也注入两 worker 环境，probe 校验单个卡号/UUID 及唯一可见设备。未设置该部署变量的本机运行不增加可见卡数限制。适配现有 docker-compose v2.29.2，不使用要求 Compose 2.30.0 的 services.gpus 语法。设备选择不提供与其他进程之间的独占保证。

仅业务端口和 RTSP TCP 端口对外发布，Redis 和 MediaMTX 控制接口留在容器网络。保留发布鉴权、禁止发布者覆盖和禁录像。固定项目名与持久卷，日常停止不删卷。

API ready 依赖两个 worker；worker 不可反向等待 API healthy，避免启动死锁。容器设置 init 和足够停止宽限期，并在 Linux 实测 SIGTERM、子进程退出及重启恢复。

### 模型设备与预检

已实现 INFERENCE_DEVICE=cpu|cuda，默认 cpu；DETECTION_BACKEND 保持算法语义。quality 的 EgoBlur 使用目标设备，YuNet/HyperLPR 继续 CPU。单独 egoblur 保留现有模型路径语义；轻量 cpu 后端搭配 cuda 已明确拒绝。服务器交付复用这些行为。

EgoBlur 的模型与输入 tensor 必须同设备，保持输入格式、阈值、NMS、分块、局部复核和后处理；复用现有输出转 CPU 的接口。

复用 models/manifest.json 与 cpu-manifest.json，通过现有 runner 启动 child 的内部 probe；所有模型清单读取、哈希、加载和推理均在该子进程中完成。固定 320×320 合成帧跑完整检测器，再用非空框验证目标设备 NMS，避免空检测未执行 GPU 算子的假通过。总超时固定 120 秒，失败/超时/取消均终止并等待子进程；成功事件和退出码通过后才启动 worker 心跳。仅暴露有限诊断，不输出完整配置，不下载或回退。

两个 worker 可能同时持有模型，显存需求须按并发验证，不能以单进程加载成功代替。

### 退出与恢复

worker 在预检前建立进程内停止事件并注册 POSIX 信号；事件存于既有 broker.state，供 execute 的控制循环检查。SIGTERM 同时停止 Receiver 接单、触发现有任务 cancel 和 runner 回收；不只依赖 Receiver.finish_event，因为持续流会占满接收端信号量。回收后撤心跳并关闭 broker，reconciler 取消主协程后执行已有 finally。

execute 必须区分部署停机与用户取消：部署停止时，回收未完成媒体后停止任务续租，跳过既有 failed/processing_failed 的正常 finish 分支，交由失租约流程生成 worker_lost；不能只设置 cancel 就声称保留原语义。已提交的成功结果保留，未提交半成品不得转成功，run 意图的流允许恢复，不写入用户 desired_state。Linux 在模型加载前启动父管道 EOF 监视，Windows 保留加载后启动以避开已知 DLL 初始化死锁。现有子进程不再派生进程，直接 kill()+wait() 足够，不引入进程组或新进程管理器。

### 网络地址

新增内部推流地址，未指定时回退现有 public 地址以兼容本机。Compose 内部使用 rtsp://mediamtx:8554；RTSP_PUBLIC_BASE_URL 仍用于返回客户端可达的服务器地址。控制地址和发布鉴权回调使用容器服务名，保留媒体源 allowlist。

部署字段、六角色依赖、权限及镜像调用详见 [部署契约](contracts/deployment.md)。不新增 HTTP 接口；[数据与生命周期](data-model.md)仅描述部署对象及既有状态保留。

保留现有心跳租约窗口：干净首次启动时预检失败不得 ready；运行中崩溃后 Redis 中旧心跳可能在默认 LEASE_SECONDS=20 秒内仍有效，超时后 ready 必须失败。正常退出主动撤心跳，不为消除窗口而清空整个池或影响其他本机 worker。

### 验证步骤

1. 单元测试设备默认值、非法组合、工厂传参、内部/外部地址，以及失败预检不登记心跳。
2. 真实 CPU 校验正确模型及缺失、坏哈希、不可读、不可加载；禁止外部模型访问后处理仍成功。
3. 检查 linux/amd64 基础镜像与二进制依赖，确认镜像不含模型、数据和本机 .env；构建阶段不依赖 GPU。
4. 目标 GPU 执行模型、输入、NMS 与输出探测，记录实际设备和双 worker 峰值显存；验证两池仅使用指定卡，错误卡号或 UUID 明确失败，不改用其他卡。
5. 完成规格六组效果、重复耗时、流帧率与丢帧记录。
6. 验证重启持久化、客户端播放、任务停止与资源释放。

每次文件任务都会创建子进程并重新加载模型；后续请求不等于常驻模型热推理。必须区分模型加载、处理和端到端时间。现有 Windows 资源测量不能直接作为 Linux 证据。

| 验证位置 | 必须覆盖的行为 | 需求 |
| --- | --- | --- |
| test_config、publisher/stream API 测试 | 内部地址校验/回落、推流用内部地址、响应仍为公开地址、发布鉴权保留 | FR-007、FR-008、FR-010 |
| test_worker_models 与预检单元测试 | 缺失/不可读/坏哈希/不可加载、probe 非零/超时、失败前不得启动 Receiver 或心跳 | FR-004、FR-005、FR-006 |
| test_inference_device 与 GPU probe | 保留策略，实际模型和非空 NMS 使用 CUDA，不可用不回退 | FR-002、FR-003、FR-006、FR-010 |
| runner/worker 生命周期测试 | 预检中取消、加载中父 EOF、SIGTERM 释放媒体进程且不伪造用户停止 | FR-007 |
| 实际 Compose config 与镜像检查 | 两种配置兼容 2.29.2、无多卡预留、无模型/本机环境泄漏、依赖锁正确 | FR-001、FR-004、FR-008、FR-011 |
| 服务器 CPU/GPU opt-in 验收 | 六组效果、重复耗时、单卡两池峰值、重启与停止、设备标识失败 | FR-003、FR-007、FR-009、FR-011；SC-001 至 SC-007 |

实施时先建立有针对性的失败用例，不为静态配置复制大量同构断言。真实模型、服务和 GPU 验证显式执行；Linux 性能使用接口耗时、docker stats 与 nvidia-smi，不把依赖 PowerShell 的现有测量用例当作 Linux 证据。操作步骤见 [quickstart.md](quickstart.md)。

## 规划出口

Phase 0 已用官方 wheel/镜像元数据、uv 0.13.0 隔离锁生成及安装 dry-run 解决软件栈选择；Phase 1 已生成 data-model.md、contracts/deployment.md、quickstart.md，并复检宪章。后续 tasks.md 已生成，实施前可执行 speckit-analyze；服务器 Docker Engine/Toolkit、真实模型与媒体验收保留为实施验证门槛，不是未定设计。

## 复杂度记录

无新增数据库实体、常驻模型服务、进程管理器或多 GPU 调度。不得通过添加多套未经验证的镜像来绕过兼容性核实。
