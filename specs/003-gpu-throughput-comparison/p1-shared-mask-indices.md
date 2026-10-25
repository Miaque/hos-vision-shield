# 主车牌共享布尔索引调查（2026-10-25）

该方向已实现最小实验候选，尚未部署或通过 CUDA 性能验证。仅处理相同帧内的重复索引计算，
不改变业务并发、模型、阈值、排序、ROIAlign、NMS 或检测调度。

## 已有设备证据与调用链

六份骨干融合 CUDA trace 均有两组重复的布尔过滤：

| 固定 Gen1 路径 | 现有 nonzero 次数 | 可复用方式 |
| --- | ---: | --- |
| RPN proposal_utils 的 nonempty mask | boxes、scores、level_ids 各一次，共 3 次 | 同一 mask 生成一次整数索引，三个张量复用 |
| fast_rcnn 的 score filter_mask | 显式 filter_inds 一次，boxes/scores 布尔索引又各一次，共 3 次 | 复用已有 filter_inds 的两列整数索引 |
| 四级 ROI 选择 | 4 次 | 保持原实现 |
| NMS 输出压缩 | 1–2 次 | 保持原实现 |

这里的重复来自 aten::index 内部再次计算 nonzero，不是已有单图 ROI 的 repeat_interleave 实验。
三个 proposal 布尔索引累计 host sync 约 0.136–2.561 ms，两次末端布尔索引约 0.286–5.503 ms；
数字来自六次独立局部采样，包含 GPU 竞争等待，不能全算为可回收成本，也不能外推为整片收益。
每个 forward 预期只减少四次 nonzero，仍须实际 CUDA trace 和普通双流双文件测量确认。

来源为 candidate-device 的原始 trace、cpu-load-prototype/product-graph.txt，逐项关联及哈希另存
`data/validation/route-a-backbone-fusion-ada-20261025-01/shared-mask-index-investigation.json`。

## 独立 CPU 原型

原型只处理已核对的两组固定 Gen1 图结构，复用已有二维 filter_inds，另为一维 proposal mask
生成一次索引；异常组数或现有索引位置不匹配即失败。原图复制后修改，不直接覆盖模型。
相同 nonzero 的行优先顺序保持，因此候选与旧布尔过滤在每个位置收集同一数值，没有改变阈值或权重。

CPU 小例覆盖 0/1/17 个候选、FP16/FP32/FP64、非连续二维输入、空/全选/混合 mask，
全部输出逐位相同，实际 CPU profiler 的 nonzero 从 6 次变为 2 次。
产品骨干图的 FP32 CPU 对照又覆盖 32×32、319×321 空图和 360×1280 七车牌输入，
坐标、分数、类别及尺寸元数据全部逐位相同。原型及检查在
`data/validation/route-a-shared-mask-index-20261025-01/` 与对应 prototype.py。

骨干融合返回对照已完成且未形成稳定收益，当前保留原骨干。
本项只在现有 `_async_device_moves` 的图副本内复用两组索引，不折叠骨干；
固定组数、已有索引位置不匹配即失败，原模型和 CPU 尺寸元数据保持。
产品原骨干的三个 FP32 CPU 输入再次确认全部输出逐位相同；17 个内部异步迁移、
四个 ROIAlign、112 个 BatchNorm 均保留。测试先复现缺失实现，再验证空/全选/混合 mask、
非连续二维输入、三种 dtype、原图不变、实际 nonzero 6→2 和异常结构拒绝。
完整 unit 598 项通过、48 项按 GPU 条件跳过，没有新增产品开关或依赖。

这些检查不能证明 Ada 半精度结果或同步、整片收益。下一步仅以该索引改动为变量，
在双流双 GPU 文件负载下采样原完整视频，核验完整输出、计数、设备路径及同步次数；
若输出发生变化先校准，不因减少算子就判定采用，也不将带 profiler 的耗时当作正式收益。

## RTX 6000 Ada 校准结果

`9373af1` / `exp-20261025-a-shared-mask-index-01` 已成功部署并完成两个完整 GPU 文件与两路流的
独立采样。两份均为 1776 帧、零丢帧、NVDEC/NVENC，检测/跟踪计数与原骨干一致，
输出全部复现已核验的 `231cf162a86c3f945b4c2be30b710c235e8bacf41766c7c9f0e7d5d2c9cc2132`，
以精确相同素材及输出哈希复用完整帧、分辨率、PTS 与 943 个 AAC 包的既有完整性结论。
同卡 UUID、CUDA 参数及 ONNX provider、实际 egoblur 源码哈希均已核对；加载分别 1.229/1.449 秒。

| forward 调用 | 原 nonzero / 新 nonzero | 原同步 API / 新同步 API | 新 GPU busy 两份范围 |
| --- | ---: | ---: | ---: |
| 64 | 11 / 7 | 15 / 11 | 7.189–7.589 ms |
| 424 | 12 / 8 | 17 / 13 | 4.092–4.105 ms |
| 768 | 12 / 8 | 17 / 13 | 8.192–8.198 ms |

六份原始 trace 均保留原骨干的 112 个 BatchNorm；本次不是已终止的骨干折叠或单图 ROI 改动。
减少四次重复 nonzero 的机制已被 CUDA trace 证实，不能把全部 host 等待当作可回收时间。
四个任务共同推进 95.122 秒，停止原因均持久化，采样收尾任务和队列为空。
带 profiler 的媒体耗时 111.624/111.057 秒只保留作诊断，不作为吞吐收益。

九份可用固定 PNG 与前一基线逐字节一致；8 项区域通过、7 份缺失、1 项既有 rtsp-green 失败，
绝对质量不因相对一致而通过。证据在该实验目录的 candidate-device/ 与 candidate-quality-png/。
普通 candidate 已用与上一原骨干返回对照相同的新采集器启动：正式 G3 外部路径、rfm 原路径、
三张图片和三组完整双文件，不启用 profiler；正式整体收益与逐路实时结果仍待实测。
