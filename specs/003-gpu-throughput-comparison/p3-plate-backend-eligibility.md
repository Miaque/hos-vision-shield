# 主车牌后端条件复核（2026-10-25）

本轮仅核对既有失败条件，没有升级依赖或部署另一推理后端。两路实时流与两个 GPU 文件任务的
并发、模型、分辨率和质量规则保持不变。

## 旧 ONNX 实验为何落到 CPU

历史 `egoblur-onnx-20261021/plate.onnx` 使用 opset 17；四个 RoiAlign 的实际 schema 版本为 16，
模式为 half_pixel、7×7 输出、自适应采样。原始 ORT trace 中这四层均为 CPUExecutionProvider，
十次调用合计 40 个 RoiAlign，累计主机时间约 7.265 秒。原包装器还存在 CPU 输入与输出往返，
不是当前 GPU 固定 I/O 路径的完整对照。

根据 [ORT v1.20.1 CUDA 内核注册](https://github.com/microsoft/onnxruntime/blob/v1.20.1/onnxruntime/core/providers/cuda/object_detection/roialign.cc)
和 [内核版本匹配实现](https://github.com/microsoft/onnxruntime/blob/v1.20.1/onnxruntime/core/framework/kernel_registry.cc#L120)，
该 CUDA RoiAlign 只注册 v10，未指定结束版本时匹配规则要求 schema 起始版本相等，
不能据算子表的“10+”推断覆盖 v16。这与旧 trace 的 CPU 分配吻合；float16 的注册也被注释。

逐份下载核对 v1.21.0、v1.22.0、v1.23.2、v1.24.2 的相同源文件，仍为 v10、float/double，
没有获得足以重跑全模型的变化条件。上游 main 已有 v16/v22 及半精度注册，但这不能证明
某个可安装发布版、服务器现有二进制或当前模型能够使用，不能直接据此升级全项目 ORT。
版本源文件、下载状态与旧 trace 保存在 `route-a-ort-kernel-audit-20261025-01/`。

本地枚举 CUDA 内核受 Windows provider DLL 依赖缺失阻断，未据此宣称服务器算子不可用；
当前服务器仍为 ORT 1.20.2。任何后续候选必须先确认实际二进制的算子分配，禁止静默回退 CPU。

## 不采用简单降 opset

不能为命中旧 CUDA 注册而直接改算子版本或删除 half_pixel 属性。
显式 CPU 小模型检查 v16 half_pixel 与 v10 坐标平移补偿：普通框输出逐位一致，
亚像素框与边界框最大绝对差分别约 1.080、0.661，因为旧定义还包含最小 ROI 尺寸处理。
证据 `opset-downgrade-cpu.json` 仅为语义反例，不是 CUDA 性能测量。
因此该降级方案不进入产品；目前不原样重跑旧 ORT 全模型实验。

## 下一项执行成本候选的条件

已有骨干冻结融合只在历史 GTX 1080 上测过：112 个 BatchNorm 归零，但完整前向中位仅改善
0.37%，另有小幅坐标/分数变化，因此未接入。当前本地没有 CUDA，不重复该本地 GPU 实验。

RTX 6000 Ada 的新 trace 仍包含 112 个归一化及其小算子，GPU 卷积与主机发射的成本比例
不同于历史 GTX 1080。若继续该方向，先仅折叠骨干图，保留原 RPN、候选、ROI 与异步迁移，
把 Ada 上减少发射的收益作为新假设验证；不将嵌套主机耗时求和当作可消除成本。
需要独立核对数值、框与遮罩差异、初始化成本，以及同一混合负载的完整耗时。
这是具备不同硬件条件的待测候选，尚未证明收益；后续实现和测量见
[骨干融合实验](p3-plate-backbone-fusion.md)。

独立 CPU 图原型已准备：仅内联折叠后的骨干，保留原模型输入、CPU 尺寸元数据、
17 个内部设备迁移及四个 ROIAlign，原模型图不被修改。32×32、319×321 空图及
360×1280 七车牌输入完成检查：空图输出逐位一致，七车牌的坐标最大差约 0.000031 像素、
分数最大差约 0.00000122，类别与尺寸元数据一致。该 FP32 CPU 检查不证明 Ada 半精度结果、
遮罩一致或性能收益；原型及结果位于 `route-a-backbone-fusion-ada-20261025-01/`。
