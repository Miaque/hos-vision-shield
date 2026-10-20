# Compose 实施与验证记录

日期：2026-10-19。本机 Windows，Git 分支 main。实施前仅本功能规格/计划/研究/任务及设计文档存在未提交改动；不覆盖这些内容，不提交或推送。

## 回归基线

`uv run --no-sync pytest tests/unit -q`：96 passed，7.81 秒。未连接真实服务，未运行真实模型或 GPU。

根声明 SHA-256：`895D295F8B4C8A783406CFF95D8D7D00A4BD4832BBAE8204EAB0122178A2B038`。
根锁 SHA-256：`E6ED9001259237798985BE4C76ECB21778B41F239070D57826A8FF1F8A0B29E7`。

## 验证边界

本机有 Python 3.12/uv 0.13.0 与四份本地模型，未安装 Docker Engine。用户要求沿用线上已有 Docker/Compose，明确由用户在服务器执行命令；不连接服务器或修改宿主配置。镜像构建、Linux 信号、CPU/GPU 六组媒体与数据持久化验收须在目标环境执行，未执行不得勾选。

用户明确只修复分析 U1，跳过 C1（补充并发压力验收）和 U2（补充完整链路零下载观测方案）；这两项尚无验收证据，不声明宪章及全部成功标准已通过。

## 实施证据

- 最终执行 `uv run --no-sync pytest tests/unit tests/integration/test_runner_lifecycle.py -q`：139 passed、1 skipped，22.88 秒；包含协调器取消清理用例，跳过项仅为 POSIX 冷加载父 EOF。没有通过调整原有断言或跳过失败来取得通过。
- 地址/停机用例先红后绿：新增内部 URL 与部署停止用例首次 8 failed、1 passed；实现后相关配置、状态边界及卡死进程回收 27 passed。
- 预检用例先红后绿：首次因 preflight 模块未实现而收集失败；实现后预检、监督和停机相关 27 passed。
- 计时事件用例先确认缺失事件失败，再实现业务子进程计时和 worker 任务关联日志。
- 自查发现新停机事件变量与既有 URL 下载分支的同名等待任务冲突；先用远程文件成功用例复现，再改用 deployment_stop，相关 4 passed。未改下载、处理策略或 HTTP 字段。
- `python -m vision_shield.detection.preflight` 在本机 CPU 和 CUDA 分别通过：四份模型身份、实际合成帧推理、非空 NMS；CPU 模型/NMS 为 cpu，CUDA 模型/NMS 为 cuda。真实 CUDA 是本机 GTX 1080（能力 6.1），不是服务器 RTX 6000 Ada。torch 2.14.1+cu126 / torchvision 0.29.1+cu126，CUDA 构建 12.6。
- 实际调用 worker.probe_models 监督 CPU 子进程成功，确认 probe 成功事件与正常退出后才返回，不依赖 Redis/MediaMTX。
- `RUN_GPU_TESTS=1 uv run --no-sync pytest tests/integration/test_gpu_model_probe.py -q`：1 passed，9.92 秒；包含真实模型设备/NMS 与既有人工选定样例区域回归。两项上游 JIT/meshgrid 弃用警告，未变更模型格式。
- GPU 锁重新生成 67 个包；共同版本与根声明的一致性测试通过，`uv pip sync --dry-run --require-hashes --torch-backend cu126 --python-platform x86_64-unknown-linux-gnu --no-build deploy/requirements-cu126.txt` 解析通过，未实际安装/替换本机依赖；根声明/锁哈希不变。
- 在收到“不在本机安装 Docker”前，曾临时下载官方独立 Compose v2.29.2，校验 SHA-256 `59cd2bd789ab2e5920674b8ac5d17a19a684b1622f17c847cc7361e832508d25` 后，两种文件组合 config --quiet 均退出 0；仅证明语法解析，不证明构建或运行。临时 exe、校验文件与目录已清理，后续 Docker 操作全部由用户在服务器执行。

## 尚未验证

- Docker Engine / NVIDIA Container Toolkit 版本与配置，CPU/cu126 镜像构建、共享库导入、H.264/AAC 编解码和容器内容。
- RTX 6000 Ada 指定 UUID/卡号的实际映射、CUDA/NMS、双 worker 同卡容量和错误设备场景。
- 六组图片/视频/至少 60 秒流效果、客户端播放、端到端及重复计时、人工漏遮/误遮核对。
- Linux SIGTERM、模型冷加载父 EOF 与进程回收、Redis/媒体持久化、重启恢复及会话/显存释放；Windows 模拟不得代替这些证据。
- C1/U2 按用户要求跳过，未新增压力验收或完整链路零下载观测方案。

## 交付核对

源码、Compose、离线模型清单及模型包交付均围绕 FR-001–FR-011；设备/预检/失败逻辑已有本地证据。SC-001–SC-007 均涉及真实目标环境，保持待验收。T018/T026 只完成配置解析部分，不能勾选镜像构建完成；T030 已提供指南及效果表，授权服务器素材哈希仍待用户填写。

`.specify/extensions.yml` 不存在，没有 before_implement/after_implement 钩子。未修改规格质量检查清单，未提交或推送。

## 部署交付包

本次包位于被 Git 忽略的 `data/deployment/`：`vision-shield-source.tar.gz`、`vision-shield-models.tar.gz` 和 `SHA256SUMS`。源码包不含本机 .env、虚拟环境、原始媒体或权重；模型包独立包含四份权重、两份原清单、模型说明及完整许可。已对压缩包中四份权重逐一流式重算 SHA-256，与既有清单一致；权重不进入应用镜像。

由用户将三个文件上传服务器，在同一目录执行 `sha256sum -c SHA256SUMS`，再按验收指南解包、准备权限并填写 `.env.compose`。源码包包含本次未提交的实现，不能用当前远端 main 代替本次源码包。实际交付镜像 digest、素材和效果记录待服务器执行后补充。
