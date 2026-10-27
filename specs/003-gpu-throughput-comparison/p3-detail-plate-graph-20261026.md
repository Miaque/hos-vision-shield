# HyperLPR 固定地址与执行图候选（2026-10-26）

在 `7ba2d5f` 实时默认三帧/flow 修正之后独立验证；rfm 目标为至少 25 FPS。
固定已采纳的原图车牌规则、现有 FP32 权重、batch 1/2 与解码后处理，继续使用服务器两条原任务。
不修改人脸 session、工作池并发或媒体输出；不能用降低并发获取收益。

## 实现范围

只修改 `cuda_detail.Detector` 的补充车牌执行。为 batch 1/2 分别保留输入/输出 CUDA 张量和 I/O binding，
每次将新输入复制到固定地址后重放同一 session 的图，输出独立复制回 CPU 供既有后处理。
构造期间完成捕获，避免在业务帧与并行人脸检测同时进行首次捕获。
非 Tensor 的局部放大输入继续使用原来的 CUDA session 推理，显式 gpu_graph_id=-1，不触发错误地址捕获。
图或推理失败继续抛出，不返回旧输出；没有 CPU 推理回退。

这是现有 320×320 补充车牌的固定 batch 1/2 执行复用，非此前被撤回的主车牌骨干融合/共享索引。
目标依据为旧设备采样的 407 次 kernel 发射、33 次复制 API；已知显式同步时间较少，不以去同步为主方向。
当前 rfm 为 1920×1080、G3 为 3840×1080，和旧 640×480 rfm 测量不同，收益必须在实际在线负载验证。

按 [ORT CUDA Graph 限制](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview)，
固定地址/形状和 I/O binding 是必要条件；同一 session 不能并发 Run。当前调用链中车牌由单一调用方执行，
并行分支只做人脸，两个文件和两个流各自保留独立模型/缓存，不以跨任务串行锁满足该条件。
代价包括构造时捕获、固定缓冲与 session 显存；须检查 worker 预检、重启和常驻模型复用。

本地通过只说明固定地址复用、独立返回、错误传播及 session 选择等语义；
实际 CUDA 捕获、数值/完整输出一致性、显存和实时 25 FPS 收益均待服务器证明。

本地完整 unit 集合 794 通过、96 项按 CUDA 条件跳过；另补三个形状/精度拒绝回归。
禁止固定地址复用的变异被缓存/独立输出测试捕获，恢复源码后通过。完整 GPU 执行仍未本地验证。

## 服务器结果：失败并撤回

候选 `30286c0` / `exp-20261026-a-detail-plate-graph-01` 通过 worker 预检，但没有通过实际混合负载。
G3 运行约 69 秒后在 frame_process 阶段失败；堆栈经过局部放大复查，最终由
`cpu.decode_plates` 的非正宽高校验抛出 `ValueError: Invalid plate dimensions`。
这是模型输出异常，不能归为源 EOF，也不能跳过校验或丢弃问题帧来维持运行。
两个并发 GPU 文件均在处理 512 帧后失败，媒体耗时分别为 24.245 / 22.584 秒，没有完整输出。
rfm 的短时速度不能作为双流双文件条件下的有效收益。

已通过受限部署脚本撤回到 `7ba2d5f` / `exp-20261026-a-stream-default-flow-01`，保留三帧检测与光流修正。
本地同步撤回执行图和固定地址缓存改动，恢复该版本的检测实现与测试；未修改检测阈值或输出校验。
确切根因尚未定位，现有堆栈不能证明是哪一种执行图/绑定交互造成异常；后续须先完成独立 CUDA
完整文件验证，再考虑在线候选。CPU 测试及 worker 预检不足以证明实际 CUDA 输出正确。

失败任务与 attempt 诊断保存在忽略目录 `data/validation/route-a-rfm25-20261026-01/graph/` 及
`graph-g3-failure-diagnostics.json`，不提交带签名的播放地址。用户重新打开平台观看会话后，
新的 G3、rfm 任务均恢复 running；G3 已出现 flow_updates，rfm 出现 skipped_frames / face_rechecks。
后续恢复观测保存于 `rollback-reopened/`，不与候选或旧任务的窗口合并。

恢复后约 300 秒 API 观测中，两路均保持同一 attempt、running；实际 FLV 分别解码 4519 / 7529 帧，
约 301.6 秒内没有缺失 PTS、倒退 PTS 或损坏帧。G3 / rfm 输出计数增量对应 14.906 / 24.881 FPS，
尚未达到各自稳定下限。采样前已有输出，启动段缺失标记保留，不把未采到的旧窗口当成零输出。
两路累计 dropped_frames 均为 1，本轮增量为 0，不能据此宣称整个 attempt 无丢帧或完整验收通过。
两个 GPU 文件同时处理并均成功，各 1776 帧，媒体耗时 124.087 / 126.117 秒；完整输出 SHA256
均为已核验的 `8724c29e7d1a8080d50e228719e72e81d1b6ff575163d1bf32975d0e4c7cf5e7`。
本地回撤实现的 75 项相关回归通过；回撤提交 `6e465ea` 已推送，服务器保持已验证的 `7ba2d5f`，
没有为记录文档再重启服务。30 分钟连续播放、稳定 25 FPS 及失败候选根因仍未完成。

## 本地 CUDA 混合输入定位（2026-10-27）

本地现有 GTX 1080 / PyTorch 2.14.1+cu126 / ORT 1.20.2 可以运行实际 CUDA。
本次从 Git 单独加载 `30286c0` 检测器到忽略目录，不恢复主干执行图，不操作服务器任务。
新的验证条件为隔离真实 CUDA、交错输入和完整文件，不重复旧 CPU mock 或 worker 预检。

三张实际图片的最小复现中，首次固定绑定重放与普通 CUDA 对照逐位一致；
同一单张 session 插入一次 `gpu_graph_id=-1` 的 NumPy 普通 Run 后，再重放的 94500 个
预测元素全部变化，最大绝对差约 432.6353，原非正宽高校验抛出 `Invalid plate dimensions`。
官方接口允许 [通过 -1 禁用本次图执行](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview)，
但当前模型的混用正确性必须实测，不能由接口可调用推定。已定位可复现调用组合，
尚未证明 ORT 内部究竟是哪一处内存/执行计划交互造成破坏。

只在隔离原型中将 NumPy blob 转为 `torch.from_numpy(blob).to(self.device)`，
再使用原固定 GPU 输入/输出绑定和重放；不对同一 session 混用 CPU feeds/CPU outputs 的普通 Run。
权重、FP32、batch、预处理、阈值、后处理和校验均不变，无 CPU 推理回退。
120 轮交错校准中，CUDA 结果和 NumPy 复查结果分别与对应普通对照逐位一致，未出现解码异常。

使用同一原始车牌视频（输入 SHA256 `85603d67...80c3e3`）顺序检查旧候选、当前普通路径、修复原型：

- 旧候选完成 511 帧后，第 512 帧在局部复查链路抛出相同错误；失败和部分输出保留。
- 普通路径与修复原型均完成 1776 帧、零丢帧，实际 NVDEC/NVENC。
- 每份均有 3552 次 CUDA batch 2、142 次 CUDA batch 1 和 142 次 NumPy 复查；调用数及检测/跟踪计数一致。
- 1776 帧最终遮罩矩形逐帧完全相同，完整输出 SHA256 均为
  `b691711f5085f7dcf928a68558b3e5b3d5b5438d688c7957ee6a4d0e41edf07c`。
- 两份均完整解码；原分辨率、全部有理 PTS、943 个 AAC 包内容及 PTS/DTS/时长与输入一致。

GTX 的主车牌为现有 FP32 路径，不能将本次输出哈希与 Ada FP16 的历史哈希混为同一配置。
本次是单文件顺序校准，没有三次交错吞吐或 RTX 双流/双文件收益结论，原型未进入主干执行路径。

新增真实 CUDA 回归覆盖 NumPy 局部复查与单张/双张 CUDA 交错；当前普通路径通过，
替换为旧候选时失败，替换为隔离修复时通过。命令为 `RUN_GPU_TESTS=1` 下执行
`tests/integration/test_gpu_model_probe.py::test_plate_predictions_survive_interleaved_numpy_confirmation_and_cuda_batches`。
证据在 `data/validation/route-a-detail-graph-isolation-20261027-*`、
`route-a-detail-graph-full-20261027-01/` 及同名前缀的采集/核验脚本。
后续是否投入 RTX 图执行收益实验，按当前两路新测量决定，不以修复故障本身宣称提速。
