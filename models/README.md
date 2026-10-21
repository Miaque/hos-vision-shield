# 模型文件

## 服务器离线交付

默认 quality 的 CPU/GPU 两种部署都使用同一组 `face.jit`、`license_plate.jit`、`yunet_face.onnx`、`hyperlpr_plate.onnx`。离线包还须包含 `manifest.json`、`cpu-manifest.json`、本文件、`NOTICE.md` 与完整 `licenses/`；四份权重合计 842,516,699 字节，设备切换不更换模型或阈值。归属与许可见 [NOTICE.md](NOTICE.md)。

GPU Compose 现在选择 `quality_yolo`：人脸主网络改用随应用包交付的 `yolov8n_face.onnx`，车牌 EgoBlur 及两份补充 ONNX 继续读取本目录；`face.jit` 保留供原 `quality` 回退。新权重的 GPL-3.0 许可、哈希及导出契约见[组件说明](../src/vision_shield/detection/assets/README.md)。该后端只支持 CUDA，CPU 部署保持 `quality`。

在已有模型的准备机器、仓库根目录打包，无需重新下载：

```bash
tar -czf vision-shield-models.tar.gz -C models face.jit license_plate.jit yunet_face.onnx hyperlpr_plate.onnx manifest.json cpu-manifest.json README.md NOTICE.md licenses
```

服务器解包至 `MODEL_DIR`，清单和权重须处于目录第一层。文件对 UID 10001 可读，Compose 只读挂载 `/models`；缺目录、文件、坏哈希、不可读、加载/推理失败均禁止 worker 就绪。`python -m vision_shield.detection.preflight` 可在独立容器内验证，具体 CPU/GPU 无网络命令见 [验收指南](../specs/002-compose-cpu-gpu/quickstart.md)。只有 quality 需要四份模型；轻量 cpu 仅核验两份 ONNX，既有 egoblur 后端核验其两条模型路径旁的 `manifest.json`。

## 本机策略与准备


当前默认`quality`效果优先组合使用四个检测模型：EgoBlur全图人脸/车牌，加OpenCV Zoo的YuNet人脸（MIT，约233KB）与HyperLPR3的320车牌检测ONNX（Apache-2.0，约2.34MB）作分块补充。只加载检测网络，不加载或运行OCR、车牌字符识别及人脸身份识别。

本机.env已配置`DETECTION_BACKEND=quality`、`FACE_MODEL_PATH=models/yunet_face.onnx`、`PLATE_MODEL_PATH=models/hyperlpr_plate.onnx`、`EGOBLUR_FACE_MODEL_PATH=models/face.jit`、`EGOBLUR_PLATE_MODEL_PATH=models/license_plate.jit`。quality的遮挡候选门槛为EgoBlur人脸0.3/车牌0.1、YuNet人脸0.6及HyperLPR车牌0.25；EgoBlur得分至少0.7直接保留，其他候选需另一模型同类框IoU至少0.2佐证，否则对周边局部复核。全图人脸佐证至少0.3，否则转局部复核。短边至少128源像素的大人脸候选需至少一路高分结果：弱EgoBlur须有YuNet至少0.9佐证，低于0.9的YuNet须有EgoBlur至少0.7佐证；全图与局部采用相同分数要求。YuNet自身至少0.9时允许EgoBlur全图0.3或局部0.1佐证。较小人脸保留原有低阈值局部复核，以保护远处侧脸；不会单独从低分人脸新增遮挡。复核失败也可能过滤真实弱目标；任一模型运行异常则整个处理失败，不降级。CPU上明显慢于单独ONNX；Windows 本机已增加 INFERENCE_DEVICE=cuda：EgoBlur 使用 GPU，ONNX 补充仍使用 CPU；CPU 模式设置为 cpu，检测策略不变。GTX 1080 使用 cu126 依赖，验证命令见根 README。验证见[效果优先验证](../specs/001-media-redaction/validation/quality-optimization.md)。

局部复核仅执行待确认类别的检测网络，初检仍运行全部四个模型；Torch每个媒体子进程4线程、OpenCV2线程。权重、输入尺度、阈值和后处理保持原方案，计时及像素一致性记录见[同效果CPU推理优化](../specs/001-media-redaction/validation/inference-performance.md)。

固定版本、下载地址、字节数与SHA-256见[cpu-manifest.json](cpu-manifest.json)，许可与来源说明见[NOTICE.md](NOTICE.md)。在项目根目录运行 `./scripts/download-cpu-models.ps1`；已存在且哈希正确时不重复下载，下载失败不会覆盖原文件。

单独轻量方案使用 `DETECTION_BACKEND=cpu`与上述两条ONNX路径，不加载EgoBlur。ONNX全图检测最长边由DETECTION_MAX_SIDE控制；另外在源图上做约640像素的重叠分块，每轴至多8块，大图扩大分块而不跳过区域，最多65个视图含全图。quality复用这些视图，另加EgoBlur一次全图。重叠遮挡保留，避免去重时丢失较宽的遮挡范围。
单独轻量方案的YuNet阈值仍为0.3，偏向补上模糊侧脸，代价是增加误遮；quality的人脸种子门槛为0.6，由EgoBlur补充部分弱人脸，较低分只参与佐证。HyperLPR使用objectness×两种车牌版式得分的最大值，阈值0.25、NMS0.3。DETECTION_BOX_PADDING默认0.15，EgoBlur和YuNet每侧外扩15%，HyperLPR车牌水平方向每侧30%、垂直15%。版式得分只用于检测，不提供识别结果。分块和扩框无法保证从未检出的目标被遮挡。

## 保留的EgoBlur对照基线

原Project Aria官方组织发布的EgoBlur Gen1权重保留于本机；quality显式组合使用，cpu与egoblur单独后端不自动回退：
https://huggingface.co/projectaria/EgoBlur/tree/9c0b319ad09d0346e8294d7bf569593d3ab865c0

固定revision下载ego_blur_face.zip与ego_blur_lp.zip；压缩包SHA-256必须与官方LFS元数据一致，
解压后分别保存为face.jit与license_plate.jit，原文件名及权重哈希记录在manifest.json。
权重与压缩包不提交Git。Apache-2.0许可及原始下载来源随manifest保留。

验证：`uv run pytest tests/integration/test_model_probe.py -q`。
加载成功只证明权重可运行，不证明普通监控或中国车牌的准确率。

切换对照须同时设置 `DETECTION_BACKEND=egoblur`、`FACE_MODEL_PATH=models/face.jit`、`PLATE_MODEL_PATH=models/license_plate.jit` 并重启进程。EgoBlur固定阈值0.1、NMS IoU0.3，两类每侧外扩15%；关闭TorchScript运行时优化，避免冷帧分析开销。640历史性能见[性能报告](../specs/001-media-redaction/validation/performance.md)，1280机器人素材基线见[补充验证](../specs/001-media-redaction/validation/robot-media.md)。新CPU结果另见[CPU优化验证](../specs/001-media-redaction/validation/cpu-optimization.md)。
