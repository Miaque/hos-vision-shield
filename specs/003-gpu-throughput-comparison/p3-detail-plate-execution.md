# 补充车牌执行成本与下一候选条件（2026-10-25）

当前服务器为原路径 `e83e898` / `exp-20261025-a-shared-mask-index-rollback-01`。
骨干融合、共享布尔索引等无稳定整体收益的改动均未继续启用；两路流与两个 GPU 文件并发约束保持。
本节只记录新设备证据和候选条件，没有启用新执行后端或精度。

## 原路径设备证据

使用现有 `X-Detection-Profile: forward-detail-plate` 在双流双文件负载下采样，每份三个 checkpoint。
两份均成功处理 1776 帧、零丢帧、NVDEC/NVENC，计数与原骨干一致，完整输出均复现
`231cf162a86c3f945b4c2be30b710c235e8bacf41766c7c9f0e7d5d2c9cc2132`；
素材与输出精确哈希一致，因此复用已核验的原分辨率、全帧 PTS 和 943 个 AAC 包完整性。
同卡 UUID、CUDA 参数/provider、实际源码哈希和停止原因已核验，收尾活动任务及队列为空。
带 profiler 的媒体耗时 115.173/117.527 秒只作诊断，不是正式吞吐结果。

六个采样均为原有两视图合批，源帧为 32/212/378，输入形状为 2×3×320×320。
每次 407 个 kernel、457 次被该 forward 拥有的 CUDA API：

| 项目 | 六份独立采样范围 |
| --- | ---: |
| host forward | 6.860–11.955 ms |
| 关联 GPU busy | 1.091–1.364 ms |
| 显式同步 API 数 | 2 |
| 同步 host 时间 | 0.0525–0.0668 ms |
| cudaLaunchKernel 次数 / host 时间 | 407 / 1.354–3.087 ms |
| cudaMemcpyAsync 次数 / host 时间 | 33 / 1.026–4.797 ms |
| HToD | 32 次、53568 字节，GPU 活动约 0.0207–0.0273 ms |
| DToH | 1 次、756000 字节，GPU 活动约 0.0309–0.0637 ms |

API 统计按与现有 profiler 相同的 scope、External id、线程和时间边界归属，排除并行人脸模型和
profiler 结束同步。不能统计整份 trace 的全部 kernel/API 后归给车牌，也不能把 host−GPU 差值
全部当作可回收时间；主机运行时开销、其他任务竞争和采样干扰仍混在其中。
当前结果不支持继续以减少两次显式同步或复制带宽作为主要方向。

已有 host 诊断的一份整片中，补车牌调用 4262 次、主机累计约 24.904 秒；
这是单阶段成本线索，与主模型及其他并行阶段不相加为整片总耗时。
实际调用链为 `cuda_detail.Detector._view_detections` → `_plate_input` → `_plate_forward`
→ `cuda_pixels.run`：维持原两视图合批，运行时逐次创建 I/O binding，并把输出绑定到 CPU。

证据在 `data/validation/route-a-detail-plate-execution-20261025-01/baseline-device/`，
包括六份原始 trace、完整输出、verification.json；API 归属统计另存 scoped-api-investigation.json。

## 下一项优先核对固定形状执行复用

在不改变权重、FP32 精度、模型调用/视图合批/质量规则的前提下，先评估 HyperLPR 的固定形状
I/O 与执行图复用，目标是主机发射及逐次小拷贝提交成本，不预设所有等待都可消除。
这是 320×320、现有 batch 1/2 的独立补车牌模型，与此前 160px 主人脸图、EgoBlur 骨干冻结不同；
407 个 kernel 和小拷贝的设备证据是新的条件，不原样重试先前无收益的主模型方案。

按 [ORT CUDA Graph 官方限制](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview)，
须确认算子分配、固定输入输出形状及地址，并保证同一 session 不并发运行。
此前已复现运行中首次捕获与并行补检冲突，因此若进入原型，捕获须在初始化期间完成，
不能将第一次捕获放进业务帧；冷启动增加、额外显存及非 Tensor 输入调用契约也必须核验。
先用原始 FP32 和批次，确认 GPU 图实际 replay、完整输出/计数/媒体一致，再做普通混合对照。
这不改变两个文件任务的实际 GPU 并发，不借流优先级或顺序处理实现收益。

## 精度转换只做静态资格检查

另核对现有 ORT 自带转换器对 HyperLPR 的支持：保持输入输出 FP32，188 个 initializer 可为 FP16，
Resize 的一个 FP32 常量及 46 个 int64 常量保留，85 个 Conv 和其他原算子保持，增加两个 Cast。
拓扑排序后 checker 通过，原文件和模型未改；转换 initializer 与直接 FP16 舍入一致，未增加裁剪差异。
该检查未运行模型、未证明 CUDA 数值/质量/收益，也未接入产品。
[官方精度说明](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)明确转换需评估精度影响；
当前 GPU busy 与 host 成本的比例，更支持先保持精度核验执行复用，不将 FP16 静态可转换等同于应采用。
本地没有 CUDA，所有实际执行和性能结论继续以服务器验证为准。
