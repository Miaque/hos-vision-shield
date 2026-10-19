# 研究记录：ARM64 CPU/GPU 部署与离线模型

日期：2026-10-19。状态：部分完成，目标硬件调查尚未结束。

## 1. 服务与 Compose

**决定**：复用六个服务角色，CPU 基础配置加 GPU worker 覆盖。

**理由**：`scripts/start-application.ps1` 已分离四个应用入口，配合 Redis 和 MediaMTX 即可。`jobs/worker.py` 每进程并发一任务。`main.py` 的 ready 依赖两池心跳，worker 不能等待 API healthy。

**比较方案**：单容器额外引入进程管理器、完整复制两套部署配置均增加无必要维护。

## 2. 设备与算法分离

**决定**：设备单独配置，候选 GPU 路径是 EgoBlur 使用 CUDA、两个 OpenCV 补充模型保留 CPU；不改算法和阈值。

**理由**：`detection/__init__.py` 的 cpu 是轻量算法；`egoblur.py` 固定 CPU 加载且输入留在 CPU，二者必须同时迁移。`quality.py` 可传递设备给主检测器。

**比较方案**：换成轻量算法会改变效果；重编 OpenCV 以全面支持 CUDA 超出最小范围。实际 GPU 使用须独立证明。

## 3. 离线与就绪

**决定**：复用两份模型 manifest 的来源、校验和及许可；模型独立只读交付，worker 心跳之前执行完整性和实际推理检查。

**理由**：现有 worker 仅检查 is_file，坏模型仍可登记心跳。短命预检进程可复用检测工厂并释放模型显存，不在父 worker 常驻副本。

**比较方案**：仅哈希不能验证算子，只有加载不能证明推理；联网下载违反需求；另造来源清单增加漂移风险。

## 4. 地址与数据

**决定**：分离内部推流与外部播放，应用角色共享媒体目录同一绝对路径，Redis 单独持久化，内部控制端口不发布。

**理由**：`media/publisher.py:Publisher.prepare` 与 `api/routes.py` 共享 public 地址。容器内部服务名不能返回外部客户端，宿主机地址也不应成为内部推流的依赖。仅持久化状态或媒体任一侧都无法满足重启要求。

**比较方案**：host network 没有解决地址语义耦合，不作为默认方案。

## 5. ARM64 GPU 软件栈：未决

**已知**：用户确认 aarch64 与 NVIDIA。未确认系统、GPU 型号、驱动、显存或 ARM SBSA/Jetson 类型。

**待决定**：基础镜像、GPU torch/torchvision 对应版本与来源、Python 3.12 兼容性、最低驱动或 JetPack 要求、双 worker 显存容量。

**理由**：ARM SBSA 和 Jetson 需要按对应发行版支持矩阵核实，仅有 aarch64 标识不足以证明任意 CUDA 制品兼容。现有 CPU 锁版本不能证明对应 GPU 制品存在。

**比较方案**：不硬编码 x86_64 wheel，不直接套最新 CUDA 示例，不让 uv sync 覆盖预装 GPU torch，不静默降低 Python 版本。

**完成条件**：取得服务器系统版本和设备型号、GPU/显存/驱动；若为 Jetson，取得 JetPack/L4T 版本。未取得前 Phase 0 不判定完成。

## 6. 计时证据

**决定**：分别记录端到端等待、模型加载和帧处理时间，不声称后续请求就是热模型推理。

**理由**：`jobs/tasks.py` 每次任务启动 `media.child`，每次加载模型；`media/child.py` 现有计时在加载后开始。`tests/e2e/test_measurements.py` 使用 Windows PowerShell 资源计数，不能原样用于 Linux。

**比较方案**：为计时引入常驻模型服务超出范围，只报告处理阶段会掩盖冷启动。

## 官方依据

以下资料本轮已核查；不是目标设备实测证据。

- [Docker Compose GPU 支持](https://docs.docker.com/compose/how-tos/gpu-support/)：设备预留必须设置 capabilities，count 与 device_ids 互斥。
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)：宿主驱动和容器运行时需要正确配置。
- [uv PyTorch 集成](https://docs.astral.sh/uv/guides/integration/pytorch/)：CPU/CUDA 来源需明确选择；是否采用 extras 或独立 GPU 锁定需待 ARM64 制品确认。
- [NVIDIA 支持矩阵](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/)：按框架发行版本及平台核实镜像支持范围。
- [Jetson PyTorch 安装](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)：Jetson 安装关联 JetPack 和对应软件版本，不能由 CPU 架构直接推导。

## 当前验证状态

已完成源码、配置、依赖锁及官方资料调查。未安装依赖、下载模型、构建镜像、连接服务器或运行真实模型测试。仓库没有 extensions.yml，因此没有 before_plan/after_plan 钩子需要执行。
