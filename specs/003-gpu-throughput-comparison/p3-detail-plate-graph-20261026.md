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
