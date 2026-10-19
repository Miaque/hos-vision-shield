# 本机 GTX 1080 GPU 切换

2026-10-19 用户调整优先级：线上服务器与 GPU 尚未确定，先使已安装 GTX 1080 的本机使用 GPU。

## 范围

- 保留 Windows 原生启动和既有媒体任务接口，不实施服务器 Compose、地址拆分或新预检服务。
- 本机驱动实测 582.78，显卡 GTX 1080，显存 8192 MiB。
- Windows x64 固定 PyTorch 2.14.1 与 torchvision 0.29.1 的 cu126 构建；CPU 运算仍可选择，其他平台保留 CPU 包源。
- 新增 INFERENCE_DEVICE=cpu|cuda；设置默认 cpu，本机 .env 与示例选择 cuda。
- quality 中 EgoBlur 使用目标设备，ONNX 补充保持 CPU；不改模型、阈值、输入尺度和复核策略。
- 模型从现有本地文件加载；不下载权重，不变更模型清单。

## 依据

- [PyTorch 旧架构支持说明](https://dev-discuss.pytorch.org/t/introducing-cuda-13-2-and-deprecating-cuda-12-8-release-2-12/3337)：CUDA 12.6 保留 Pascal 等旧架构支持。
- [CUDA 12.6 官方制品索引](https://download.pytorch.org/whl/cu126/torch/)：存在 2.14.1 Windows CPython 3.12 制品。
- [EgoBlur Gen1 官方加载流程](https://github.com/facebookresearch/EgoBlur/blob/main/gen1/script/demo_ego_blur_gen1.py)：先以 CPU 加载，再通过 `.to(device)` 迁移模型参数及缓冲区。直接 `map_location=cuda` 会把尺寸计算常量也迁到 GPU，而运行时尺寸张量仍在 CPU，真实推理已复现设备冲突；采用官方流程后通过。

## 验证记录

新增设备选择、参数传递和 GPU 不可用不回退的单元测试，先验证四项失败，再实现。
真实 GPU 验证使用显式 RUN_GPU_TESTS=1 开关，校验四份权重并运行已标注图像。

- 用户完成依赖同步后实测为 torch 2.14.1+cu126 / torchvision 0.29.1+cu126，CUDA 12.6 可用，设备为 GTX 1080，构建包含 sm_61。
- 单元测试、视频管线、取消检查与真实 GPU 模型验证合计 103 项通过，2 条上游 Torch 警告，14.45 秒；日志 `data/validation/local-gpu/tests.log`。
- 另以 CUDA 运行既有弱目标、轮胎误遮和按类复核质量检查：7 项通过，8 条上游警告，22.07 秒；日志 `data/validation/local-gpu/quality-gpu.log`。该命令仅选择这七项，未运行依赖缺失历史素材的完整集合检查。
- 真实 CPU 质量回归：7 项通过，1 项因缺少历史素材 `samples/robot-quality/rtsp-green.png` 失败；没有修改或跳过该断言。
- `uv sync --frozen --offline --dry-run` 确认锁文件匹配，环境不需要更改；未再次下载包或模型。
- 原有 `media.runner → media.child` 链路分别完成一张图片和一个三帧测试视频，退出码均为 0；视频帧数、尺寸和源时间戳检查通过。输出为 `data/validation/local-gpu/child-image/result.jpg` 与 `child-video/result.mp4`。

## 同机单图耗时

以下是首次设备切换的历史基线；最新两路全图检测重叠执行的完整数据集结果见 [GPU 同效果优化记录](gpu-performance.md)。

固定 `image-7.jpg`（3840×1080）、quality、最长边 1280、并发 1。每种设备加载一次模型后运行四次检测与遮挡，后续中位数取后三次；计时不含模型加载、文件编解码、API 或队列。数值见 `data/validation/local-gpu/verification.json`，验证脚本为同目录 `verify.py`。

| 项目 | CPU | GTX 1080 |
| --- | --- | --- |
| 模型加载 | 3.205 秒 | 1.250 秒 |
| 首次检测与遮挡 | 13.597 秒 | 2.467 秒 |
| 后续三次中位数 | 10.423 秒 | 2.154 秒 |
| 最后一次检测框数 | 21 | 21 |

该样例后续处理约快 4.84 倍，CPU/GPU 最后一次遮挡范围逐像素一致；不能外推为全素材准确率或完整服务吞吐量。GPU 测得 PyTorch 峰值分配约 0.95 GiB，不包含 CUDA 上下文、显示及其他进程占用，不能作为双 worker 容量承诺。

实际媒体子进程另测：图片从启动到退出约 8.35 秒，其中图片处理 2.56 秒；三帧测试视频总计约 12.32 秒，其中视频处理 6.81 秒。每次任务重新加载模型，不把前述模型驻留计时当作用户端到端耗时。

输出示例：`data/validation/local-gpu/image-7-cuda.png`。

## 限制

现有 worker 启动仍只检查模型文件存在，健康检查不能代替真实模型验证；需按 README 在首次启动前运行 GPU probe。
服务器、Docker、长时间流性能和多任务峰值显存不属于本次切换的已验证范围。
