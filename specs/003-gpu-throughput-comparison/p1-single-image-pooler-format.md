# 单图 ROI 格式转换实验（2026-10-25）

接续 `eed1331` 的异步迁移候选，在 `cd0633c` 的执行路径上只改变 ROI batch 索引构造，
保持两路实时流与至少两个 GPU 文件任务的并发契约。attempt 原因记录已独立部署，
后续性能测试自然遇到切换时读取记录，不为等待重连暂停核心优化。

## 选择依据与范围

固定 Gen1 每次接收一张 CHW 图像，其 `_convert_boxes_to_pooler_format` 仍将单个候选数迁到 GPU，
用 `arange(1)` 与 `repeat_interleave` 生成全零 batch 索引，再与原始候选框拼接。
已有 trace 在该格式转换的复制与 repeat_interleave 内分别记录同步等待。

候选以 `zeros_like(boxes[:, :1])` 直接生成相同零列，复用原始框拼接。仅改该固定图的对应节点，
不改权重、原尺寸元数据、RPN、候选数、NMS、四层分组、ROIAlign 或 ROI head。
模型结构匹配失败时仍失败关闭，原图不被修改；沿用当前固定 TorchScript 的内部图接口限制。

独立原型先保留所有 ROI 在四层上计算，再选择原层，以避免四次动态分组同步。
0/1/17/128/1000 个候选的 CPU 池化输出全部逐位相同，但原模型 adaptive sampling=0，
将大框送往细层会大幅增加采样：测试尺寸的样本工作量比值为约 1.67–85 倍；
1000 候选 CPU 约 0.350→10.662 秒。该值不是 GPU 测量，但已暴露明确的额外计算问题，
因此四层全量原型未接入运行路径、未部署。当前候选保留原分组和全部原 ROIAlign 工作。

## 验证

原型及输出保留在 `data/validation/route-a-roi-pooler-sync-20261025-01/`。
`single-image-format.py` 固定从 `cd0633c` 提取前一候选实现，避免基线随 main 改变。
CPU 原型在空画面 32×32、319×321 和有七个车牌输出的 360×1280 图片上，
完整框、类别、分数及 CPU 尺寸元数据均与原图逐位相同。

产品回归先在旧实现失败，再检查 0/1/17 个候选的原输出、CPU 阻塞下载、输入不变、
原方法图不变和 repeat_interleave 被消除；18 项定向检查通过。
产品实现的 CPU 原模型对照也覆盖上述三种输入，完整输出逐位相同，四个 ROIAlign 与四个
分层 nonzero 保留，repeat_interleave 为零。完整 unit 集合 584 项通过、48 项按既定 GPU 条件跳过。
本地没有 CUDA，CPU 图等价不能证明 Ada 半精度结果或性能收益；仍需服务器独立采样和完整任务比较。
只有整体收益超过波动且输出与质量不退化才保留。

实现语义参考 [PyTorch repeat_interleave](https://docs.pytorch.org/docs/stable/generated/torch.repeat_interleave.html)
对输出形状同步的说明，以及 [Torchvision ROIAlign](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/roi_align.html)
对 batch 索引列和自适应采样数量的定义；性能结论仍以本项目服务器实测为准。

## RTX 设备采样

服务器固定 `b2e6ca7` / `exp-20261025-a-single-image-pooler-02`，同时运行两路流和两个 GPU 文件。
两份文件各 1776 帧、零丢帧，媒体耗时 119.205/117.544 秒；本轮带 profiler，且播放传输条件与
此前内部 RTSP 采样不同，这两个耗时不用于判定整体收益。

六个对应主车牌采样的同步次数均再减少三次：从异步迁移版本的 15/17/17 变为 12/14/14，
HtoD 次数减少一次；稳态主车牌主机 P50 为 32.452/30.941 ms，P95 为 47.217/46.613 ms。
两份完整输出 SHA256 仍为 `231cf162a86c3f945b4c2be30b710c235e8bacf41766c7c9f0e7d5d2c9cc2132`，
与已有完整视频逐字节一致，检测/跟踪计数、NVDEC/NVENC 均核验；完整帧、PTS 和 943 个 AAC 包
的等价结论按精确哈希复用。六份原始 trace、profile 与输出保存在
`data/validation/route-a-single-image-pooler-20261025-01/candidate-device/`。

采样证明目标同步确实消除，尚未证明收益超过运行波动。未通过性能、绝对质量或最终验收。

## 普通混合负载与返回对照

不带 profiler 的三轮双流双文件已完成：六份媒体耗时为
114.864/113.549、114.126/114.292、116.011/117.455 秒，平均 115.050 秒。
六份均为 1776 帧、零丢帧、NVDEC/NVENC，检测/跟踪计数相同，输出仍与上述 SHA256 完全一致；
逐帧 PTS 与 943 个 AAC 包的完整性也已复核。每组两文件与两流同时推进的实测重叠时间
分别为 107.509、113.685、115.311 秒，未降低并发。

证据保存在 `route-a-single-image-pooler-20261025-01/candidate/verification.json`。
此前 `cd0633c` 普通混合回归六份均值约 117.349 秒；目前约 1.96% 的观察差值尚不能排除波动。
下一阶段仅将本实验的 ROI 格式转换恢复至 `cd0633c`，保留内部异步迁移与 attempt 诊断、
外部停止顺序修复，使用同一素材、两流播放和三组并发文件返回对照。该恢复是测量控制变量，
不是宣告候选失效；只有可重复的整体收益才决定是否恢复候选。恢复后 14 项定向检查通过。
