# Research: YOLO CPU 对比

日期：2026-10-19。本文记录选型与实施门槛，不是运行报告。

## 1. 模型与执行引擎

**Decision**：首轮用现有 OpenCV DNN CPU 执行两个专用 ONNX，不新增 Ultralytics 运行依赖，不训练。固定初选如下：

| 目标 | 权重与来源 | SHA-256 |
| --- | --- | --- |
| 人脸 | [akanametov release 1.0.0 / yolov8n-face.onnx](https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx) | `06b941fd5792be624ad18f2df9ede0a021c4df165dd418204d978c20fd555928` |
| 车牌 | [morsetechlab / license-plate-finetune-v1s.onnx](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/251a30d7daedca065f56e04b0af04052c907c68f/license-plate-finetune-v1s.onnx) | `301e4239442b3a533c868ad1cc2ec38933e75ca53e5c94c58755cff2cfb294b7` |

**Rationale**：专用类别符合需求；v8 人脸 nano 优先避免较新注意力算子引入额外适配，不代表质量最优；车牌 small 在 CPU 开销与效果之间作为首个候选。现有 OpenCV 已承担两类 ONNX 推理，无须先建立新的推理服务。

**Alternatives considered**：通用 COCO 检测器不能代替人脸/车牌专用检测；Ultralytics Python 推理增加依赖，现阶段不采用；YOLOv12 人脸引入更多算子兼容不确定性，首轮不选；训练新模型超出本轮范围。

**Evidence**：[人脸原作者仓库](https://github.com/akanametov/yolo-face)、[实际 release 资产](https://github.com/akanametov/yolo-face/releases/expanded_assets/1.0.0)、[车牌模型卡](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)、[车牌文件页](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/blob/251a30d7daedca065f56e04b0af04052c907c68f/license-plate-finetune-v1s.onnx)。下载链接与元数据已核实；权重尚未下载和执行。

## 2. ONNX 契约与模型交付

**Decision**：实施第一步校验模型哈希、实际输入输出 shape、类别、内置 NMS 及 CPU 真实推理，再固定适配器。模型 manifest 记录所有信息；下载先写 .part，哈希匹配后原子替换，模型与原始媒体不进 Git。

**Rationale**：人脸作者导出示例含 `nms=True`，不能凭 YOLO 版本假定裸输出 `(1,5,8400)`。车牌作者提供 ONNX 及推理示例，也不足以证明当前 OpenCV 可以执行该具体文件。实际 shape 属于权重交付验证，不需要用户猜测。

**Alternatives considered**：默认猜张量、吞掉不支持算子或自动改模型会污染比较，均不采用。若固定 ONNX 不兼容，先报告具体错误，研究并记录最小替代方案后更新设计，不能直接开始全量运行。

[OpenCV 官方 YOLO 教程](https://docs.opencv.org/4.13.0/da/d9d/tutorial_dnn_yolo.html) 强调预处理和模型输出需要匹配。应用帧契约是 RGB，ONNX 输入颜色、归一化、letterbox pad 与缩放均须依实际导出契约验证。

## 3. 许可与适用边界

**Decision**：保存上游许可和来源，作为本地比较候选；不把“可以下载”写成“已获生产部署许可”。

**Rationale**：人脸仓库标记 GPL-3.0，车牌模型卡标记 AGPL-3.0；这些均非项目现有模型全部适用的宽松许可。本文只记录上游声明，不作生产使用的法律结论，后续采用/发布前单独核对适用义务。

**Alternatives considered**：忽略权重许可或把 ONNX 格式等同于许可豁免均不采用。

车牌作者承认其公开数据存在训练/测试混入，公开指标不能用作本项目效果证据；作者也说明小目标和非拉丁车牌的效果未保证。因此只用本地 demo 得出有限样本结论，不预设胜出。

## 4. 基线与流程复用

**Decision**：固定 `0664be4ae9edb6d0eac7bb2635d5563dad9d7956`、quality、CPU、最长边配置 1280、padding 0.15；不复用旧 evaluate-robot-media.py 的固定 EgoBlur 对照及前三帧采样。

**Rationale**：`config.py` 默认 quality；`quality.py` 组合 EgoBlur + YuNet/HyperLPR，CPU 串行执行，含分块与局部确认。`redaction.py` 保留至多两次检测、200ms 历史遮挡。只比孤立帧会丢失视频实际语义。

**Alternatives considered**：轻量 cpu backend、单独 EgoBlur 和历史 GPU 报告均不是当前主干 CPU 基线。

复用 `load_detector`、`process_image`、`process_video`、`FrameRedactor`、媒体子进程与 worker 生命周期。新 detector 返回相同源坐标框；worker 须按 yolo 选择两份模型路径，否则只在离线脚本可跑而业务启动失败。

保持主干原后处理；YOLO 采用人脸 15%、车牌横向 30% 纵向 15% 扩框。两者共用参数值但主干不同来源车牌框有不同扩框，报告必须披露，不为了统一而改基线。网络固定输入尺寸与全图/分块处理分别记录，不能声称两种模型的实际网络分辨率完全相同。

## 5. 对比方法

**Decision**：完整处理 21 份媒体，质量复核使用全部图片及每段视频每 5 秒首个达到该时刻的帧，补末帧并去重。先冻结清单，之后处理两组。输出对齐原始 PTS，判定遮挡包含 FrameRedactor 历史框。

**Rationale**：本地读取元数据，5 段视频分别约 60.160、60.698、61.897、60.858、60.339 秒，帧数元数据均为 0，需实际解码。抽样降低人工复核量，但不减少媒体处理量。按遮挡区域并集判断 full/partial/missed，误遮以不对应真实目标的连通遮挡区域计数，避免重复框膨胀。

**Alternatives considered**：前三帧、固定一秒片段不能代表全视频交付；每帧人工标注超出本轮需要；检测框数量不能代替脱敏覆盖质量；原始黑像素不能作为遮挡证据。

统计以人工确认状态为门槛，代理可辅助准备但不冒充人工复核。新增漏遮、误遮及过大范围案例必须可定位。无法归因类别的基线误遮保留 unknown，不改基线检测协议来猜测类别。

## 6. 测量与资源

**Decision**：相同解释器与锁定依赖、顺序运行两组，单模型实例串行使用；OpenCV 2 线程，基线 Torch 4 线程。初始化与单次媒体处理分开记录，单次耗时为辅助指标，不声称稳定性能分布。

**Rationale**：使用现有资源边界，避免不同进程互相争抢 CPU。完整视频处理可能较慢，但不擅自降格为片段。保存逐素材完成状态，中断后重跑未完成的整个素材。

**Alternatives considered**：多进程同时跑两组会混入资源竞争；调参后覆盖原记录或忽略失败项会失去可复核性。

## 实施修订（2026-10-19）

实际 OpenCV 4.14.0 加载固定 yolov8n-face.onnx 失败：NonZero 算子不支持。保留权重及其哈希，改用 ONNX Runtime CPUExecutionProvider；不重新导出、不静默换模型。新增 onnxruntime 运行依赖，以 uv.lock 固定实际版本；双模型共用顺序推理，intra_op_num_threads=2、inter_op_num_threads=1，禁用 provider fallback。官方接口依据：https://onnxruntime.ai/docs/api/python/api_summary.html 。本修订替代上述 OpenCV DNN 执行选择，OpenCV 仍用于预处理/NMS。
