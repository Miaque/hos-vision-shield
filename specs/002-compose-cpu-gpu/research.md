# 研究记录：服务器 CPU/GPU 部署与离线模型

日期：2026-10-19。状态：Phase 0 研究完成，依赖与镜像方案已确定；源码、官方制品元数据及隔离解析验证通过，不代表容器或服务器运行通过。

## 1. 服务与 Compose

**决定**：复用六个服务角色，CPU 基础配置加 GPU worker 覆盖。

**理由**：`scripts/start-application.ps1` 已分离四个应用入口，配合 Redis 和 MediaMTX 即可。`jobs/worker.py` 每进程并发一任务。`main.py` 的 ready 依赖两池心跳，worker 不能等待 API healthy。

**比较方案**：单容器额外引入进程管理器、完整复制两套部署配置均增加无必要维护。

## 2. 设备与算法分离

**决定**：复用已实现的独立设备配置；EgoBlur 使用 CUDA、两个 OpenCV 补充模型保留 CPU，不改算法和阈值。

**理由**：`detection/__init__.py` 的 cpu 是轻量算法，不能代替设备选择；`egoblur.py` 已按 CPU 加载后迁移参数与缓冲区的流程选择设备，输入同步迁移，`quality.py` 已传递设备。Linux 依赖源仍固定为 CPU，服务器 GPU 交付仍需处理依赖选择。

**比较方案**：换成轻量算法会改变效果；重编 OpenCV 以全面支持 CUDA 超出最小范围。实际 GPU 使用须独立证明。

## 3. 离线与就绪

**决定**：复用两份模型 manifest 的来源、校验和及许可；模型独立只读交付，worker 心跳之前执行完整性和实际推理检查。

**理由**：现有 worker 仅检查 is_file，坏模型仍可登记心跳。预检复用 runner/child，清单、哈希、模型加载和实际推理全部在同一个受总超时约束的子进程中执行，不在常驻 worker 导入模型库或留下哈希线程。固定合成帧加非空 NMS 检查算子，退出后释放显存再启动心跳。

**比较方案**：仅哈希不能验证算子，只有加载不能证明推理；联网下载违反需求；另造来源清单增加漂移风险。

## 4. 地址与数据

**决定**：分离内部推流与外部播放，应用角色共享媒体目录同一绝对路径，Redis 单独持久化，内部控制端口不发布。

**理由**：`media/publisher.py:Publisher.prepare` 与 `api/routes.py` 共享 public 地址。容器内部服务名不能返回外部客户端，宿主机地址也不应成为内部推流的依赖。仅持久化状态或媒体任一侧都无法满足重启要求。

**比较方案**：host network 没有解决地址语义耦合，不作为默认方案。

## 5. 服务器 GPU 软件栈：已确定

**已知**：用户提供的命令输出为 Ubuntu 24.04.4 LTS、x86_64、7 张 NVIDIA RTX 6000 Ada Generation、每张显存 49140 MiB、驱动 595.84；替代此前 aarch64 意向，本次不扩展 ARM64/Jetson。用户确认先单卡验证，部署时指定卡号或 UUID，两类 worker 共用该卡。

**现有 Compose**：独立入口 docker-compose v2.29.2 可用，docker compose 子命令不可用。Docker Compose v2.29.2 源码已将 reservations.devices 转换为 Docker DeviceRequests，可复用设备预留与 device_ids；services.gpus 则要求 2.30.0，不采用该新语法，也不因命令入口不同而要求升级。

**决定**：CPU/GPU 镜像共用 Python 3.12 slim-bookworm，分别使用 torch 2.14.1+cpu / +cu126 和 torchvision 0.29.1+cpu / +cu126；构建用 uv 0.13.0。根 pyproject.toml/uv.lock 保持不变，CPU 镜像沿用根锁，GPU 部署锁从根声明和共同版本约束生成。双 worker 显存容量由目标服务器验收测量，不虚构容量门槛。

**理由**：官方 CPU/cu126 索引均有 CPython 3.12、manylinux_2_28_x86_64 制品，四个 wheel 的 HEAD 均返回 200，小型 METADATA 已核对。bookworm glibc 2.36 满足 glibc≥2.28，避免 Alpine/musl。GPU torch 元数据携带 CUDA 12.6.3、cuDNN 9.10.2.21 等运行依赖，无需再叠加 CUDA devel 基底。

目标 RTX 6000 Ada 计算能力 8.9；CUDA 12.6.3 对应 Linux 驱动 560.35.05，用户报告的 595.84 满足该版本关系。525.60.13 只是 CUDA 12 次版本兼容底线且有功能/PTX限制，不作为本交付的无条件下限。Toolkit 平台表支持 Ubuntu 24.04 x86_64，实际设备、sm_89、NMS 和 JIT 仍须容器实测。

### 官方 wheel 身份

以下 SHA-256 来自官方索引，未下载完整 wheel 重新计算；文件均为 cp312-cp312-manylinux_2_28_x86_64。

| 包与构建 | 字节数 | 索引 SHA-256 |
| --- | --- | --- |
| torch 2.14.1+cpu | 196253677 | `5a6363570c753812540a05eb82380e329469cbe668643e88111414c12627711f` |
| torch 2.14.1+cu126 | 869173899 | `75727245346d8f92224a860bf40c56bd681edcfa0456d01d7dfcb0bf6f0cf86c` |
| torchvision 0.29.1+cpu | 1708098 | `ef220d4b21878c58947851beb7337382897159105f012f0c9c07bf7e84174d7e` |
| torchvision 0.29.1+cu126 | 7194734 | `f846900988cbd64b055886c9ad7e7a44606c5710e8272921062c647f237eae34` |

来源：[CPU torch](https://download.pytorch.org/whl/cpu/torch/)、[cu126 torch](https://download.pytorch.org/whl/cu126/torch/)、[CPU torchvision](https://download.pytorch.org/whl/cpu/torchvision/)、[cu126 torchvision](https://download.pytorch.org/whl/cu126/torchvision/)。GPU 运行依赖依据 [torch METADATA](https://download-r2.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl.metadata)。

### 锁与安装方式

以下过程已在隔离临时目录、使用原始根声明与锁文件通过 uv 0.13.0 实测，只解析元数据，没有安装依赖或下载大型 wheel：

```bash
uv export --locked --no-dev --no-emit-project --no-hashes --no-emit-package torch --no-emit-package torchvision --output-file common-constraints.txt
uv pip compile pyproject.toml --no-sources --torch-backend cu126 --constraint common-constraints.txt --generate-hashes --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --no-build --emit-index-annotation --output-file requirements-cu126.txt
```

结果为 67 项运行依赖：46 项共同依赖按 Linux/Python 3.12 markers 求值后全部与原锁版本相同，另有两项 PyTorch 包和 19 项 GPU 依赖。独立 assert 比较通过；对结果执行 `uv pip sync --dry-run --require-hashes --torch-backend cu126 --python-platform x86_64-unknown-linux-gnu --no-build` 通过。实施时将生成结果保存为 deploy/requirements-cu126.txt，根声明或锁改变时重新生成并核对共同版本；中间 constraints 使用临时目录，不作为第二份人工维护的依赖声明。

实施已生成 `deploy/requirements-cu126.txt`，并重新执行共同版本检查及安装 dry-run；没有安装或替换本机依赖。生成时附加 `--no-annotate --emit-index-annotation --custom-compile-command '<仓库相对路径的生成命令>'`，省去注释中的本机临时目录。维护者从仓库根目录导出共同约束到临时目录，将上述 compile 的输入设为根 pyproject.toml、约束设为临时文件、输出设为 deploy/requirements-cu126.txt；服务器正常部署直接使用交付锁，不需要在宿主安装 uv。

CPU 构建使用 `uv sync --locked --no-dev --no-editable`；GPU 在空 `/opt/venv` 执行 `uv pip sync --python /opt/venv/bin/python --require-hashes --torch-backend cu126 --no-build deploy/requirements-cu126.txt`，再以 `uv pip install --python /opt/venv/bin/python --no-deps .` 安装同一项目。项目构建依赖继续遵循已有 build-system 范围。两类运行入口统一使用安装后的 Python 包，运行镜像不执行 uv sync，避免将 CUDA 构建换回根规则中的 CPU 构建。

**比较方案**：直接把 torch 移到互斥 extras 会使原有默认 Windows 安装缺少模型依赖。保留主依赖再叠加 GPU extra 的隔离实验出现索引冲突；使用 extra 反向 marker 又使默认解析错误选择 CUDA，不能采用。默认 dependency group 虽可运行却改变正式包元数据；uv.toml 不支持 sources。生成一份 GPU 部署锁的影响局限于部署，比改动本机安装或先装 CPU 再覆盖更小。

**完成条件**：Phase 0 的制品与解析调查已满足。Docker Engine、NVIDIA 容器运行时、指定单卡可见性、sm_89/NMS/JIT 和双任务峰值属于实施后的真实运行验证，不以 dry-run 代替。

## 6. 基础镜像与媒体依赖

**决定**：同一 Dockerfile 构建 CPU/cu126 镜像，运行应用用户为 10001:10001；API/reconciler 用 CPU 镜像，两类 worker 按模式选择镜像。构建阶段固定工具和基础镜像 digest，运行镜像只含应用与运行依赖，不复制 uv 缓存、开发依赖、权重或本机环境文件。

2026-10-19 已查询 Docker Hub/GHCR Registry 元数据，下列均存在 linux/amd64 制品；表中为多平台索引 digest，构建/部署显式使用 linux/amd64。元数据存在不表示已拉取、构建或运行。

| 镜像 | 索引 digest |
| --- | --- |
| python:3.12-slim-bookworm | `sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258` |
| ghcr.io/astral-sh/uv:0.13.0 | `sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168` |
| redis:7.4.3 | `sha256:236e397c1d5ab7a94adaf1a51eec3ca8333b05fafcd6d423c6c7cc5987e519a0` |
| bluenviron/mediamtx:1.21.2 | `sha256:c7540465f868a02652118bea5a0daab12a29ac0f6cb8485abd535939ecddf886` |

依据：[Python 标签元数据](https://hub.docker.com/v2/repositories/library/python/tags/3.12-slim-bookworm)、[Redis 标签元数据](https://hub.docker.com/v2/repositories/library/redis/tags/7.4.3)、[MediaMTX 标签元数据](https://hub.docker.com/v2/repositories/bluenviron/mediamtx/tags/1.21.2)、[uv 官方容器说明](https://docs.astral.sh/uv/guides/integration/docker/)。uv 的 tag 和 amd64 子清单已通过 GHCR API 核对。

**理由**：应用通过 PyAV 读写媒体，现有源码没有外部 ffmpeg/ffprobe 子进程；锁定的 av 19.0.1 提供带 FFmpeg 的二进制 wheel。宿主/验收客户端继续提供 FFmpeg/ffprobe/ffplay，不为了测试工具扩大应用镜像。构建后必须检查 av/cv2/torch/torchvision 导入、H.264/AAC 编解码及 RTSP TCP；仅在实际共享库缺失时补最小系统包，不安装编译工具链或重复 CUDA SDK。

**比较方案**：不采用 Alpine，不叠加 NGC/CUDA devel 镜像，不将整个本机目录或多个角色塞进单个容器；不更改现有 Windows 基础服务配置，另提供容器专用配置。

## 7. 启动、停止与恢复

**决定**：Redis healthy 后启动应用；stream-worker 只等待 API/MediaMTX started，不等待 API healthy，避免 API ready 反向依赖 worker 的循环。依赖反向停止使流 worker 在 API/MediaMTX/Redis 之前收尾；应用容器 init、60 秒停止宽限，媒体子进程沿用 10 秒停止与强杀回收。

**理由**：自建 worker 入口未注册 SIGTERM；Taskiq 0.12.4 Receiver 在读取停止哨兵前等待并发信号量，持续流占满槽位时，仅设置 Receiver finish_event 不够。worker 将进程内停止事件放入现有 broker.state，execute 控制循环同时响应并停止媒体子进程，之后撤心跳/关 broker。reconciler 取消主协程后走已有 finally，不另加进程管理器。

execute 必须识别部署停止并在未完成媒体回收后停止续租、跳过正常 processing_failed 的 finish，才能保留失租约 worker_lost；仅设置既有 cancel 会走错失败分支。已提交结果保留，服务退出不伪造用户停止意图，run 流允许恢复。Linux 在模型加载前启动父管道 EOF 监视，覆盖父进程冷加载时退出的窗口；Windows 保留当前顺序以避免 DLL 初始化死锁。

沿用现有 worker 心跳租约：干净首次启动预检失败不 ready；运行中崩溃后的旧心跳可能保留最多默认 20 秒，过期后 ready 失败。正常退出撤销自身心跳，不清空池来掩盖检测窗口。

**比较方案**：仅加 init 或延长 stop_grace_period 都不修复应用退出路径；改用另一套 Taskiq CLI 会扩大现有 receiver 配置和任务语义变更。现有媒体子进程不派生外部进程，直接终止并 wait 足够，不预先引入进程组。

## 8. 计时证据

**决定**：分别记录端到端等待、模型加载和帧处理时间，不声称后续请求就是热模型推理。

**理由**：`jobs/tasks.py` 每次任务启动 `media.child`，每次加载模型；`media/child.py` 现有计时在加载后开始。`tests/e2e/test_measurements.py` 使用 Windows PowerShell 资源计数，不能原样用于 Linux。

**比较方案**：为计时引入常驻模型服务超出范围，只报告处理阶段会掩盖冷启动。

## 官方依据

以下资料用于兼容性核查；不是目标设备实测证据。

- [Docker Compose GPU 支持](https://docs.docker.com/compose/how-tos/gpu-support/)：设备预留必须设置 capabilities，count 与 device_ids 互斥。
- [Compose v2.29.2 设备预留实现](https://github.com/docker/compose/blob/v2.29.2/pkg/compose/create.go)：setReservations 将设备 capabilities、count、device_ids 与 driver 传给 Docker DeviceRequests；属于源码证据，不是目标服务器运行证据。
- [Compose services.gpus](https://docs.docker.com/reference/compose-file/services/#gpus)：该简写要求 Compose 2.30.0，因此现有 2.29.2 使用设备预留语法。
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)：宿主驱动和容器运行时需要正确配置。
- [NVIDIA Container Toolkit 支持平台](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html)：列出 Ubuntu 24.04 x86_64；不能替代服务器实际 GPU 容器探测。
- [uv PyTorch pip 接口](https://docs.astral.sh/uv/guides/integration/pytorch/#the-uv-pip-interface)：使用明确 torch-backend，不启用自动选择。
- [CUDA 12.6.3 发布说明](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html)：核实驱动版本关系，不能替代实际 CUDA 算子验证。
- [PyAV 安装说明](https://pyav.basswood.io/docs/stable/overview/installation.html)：官方二进制 wheel 携带 FFmpeg，区别于外部 CLI 测试工具。
- [Compose 启停顺序](https://docs.docker.com/compose/how-tos/startup-order/)：区分 started 与 healthy，按依赖反向停止。
- [NVIDIA GPU 计算能力](https://developer.nvidia.com/cuda/gpus)：RTX 6000 Ada 的计算能力为 8.9，与用户提供的设备完整型号对应。

## 当前验证状态

本轮重新核对源码、配置、依赖源、既有规格及 Docker Compose、NVIDIA Container Toolkit、uv PyTorch 官方文档。四份本地权重 SHA-256 均与两份 manifest 一致，合计 842,516,699 字节，离线交付还须包含清单与许可证；运行时未发现模型自动下载调用。

本轮完成官方 wheel HEAD/METADATA、Registry manifest、隔离锁生成、共同版本 assert 和安装 dry-run；未安装依赖、下载大型 wheel 或模型、构建镜像、连接服务器或运行真实模型测试。仓库没有 extensions.yml，因此没有本轮 plan 的前置或后置钩子需要执行；既有本机验证仅见 local-gpu.md，不扩展为服务器证据。

服务器环境信息来自用户粘贴的命令输出，本轮未远程复核；容器运行时、真实单卡模型与完整媒体验收仍需在部署阶段执行。
