# 检测模型来源与许可

项目仅执行人脸与车牌目标检测，不加载身份特征、车牌字符识别网络、字符字典或 OCR。

- EgoBlur Gen1 人脸及车牌：Project Aria 官方 `projectaria/EgoBlur`，Apache-2.0；来源为 <https://huggingface.co/projectaria/EgoBlur>，固定 revision、原始文件名及 SHA-256 见 `manifest.json`。许可全文复用 `licenses/HyperLPR-Apache-2.0.txt` 中的通用 Apache-2.0 文本，EgoBlur 的来源归属以本条及清单为准。
- YuNet 人脸：OpenCV Zoo / Shiqi Yu，MIT，许可全文见 `licenses/YuNet-MIT.txt`。
- HyperLPR3 320 车牌：HyperInspire / szad670401/HyperLPR，Apache-2.0，许可全文见 `licenses/HyperLPR-Apache-2.0.txt`。RGB letterbox 与检测张量解码沿用上游 `Prj-Python/hyperlpr3/inference/multitask_detect.py` 模型契约。

ONNX 模型的固定 revision、下载来源、字节数与 SHA-256 见 `cpu-manifest.json`。四份模型及两个清单、本文和完整 `licenses/` 随离线包一起提供；权重保留在本地，不提交 Git，不进入应用镜像。

GPU Compose 的 `quality_yolo` 另使用随源码包交付的 YOLO 人脸 ONNX（akanametov / yolo-face，GPL-3.0），替代 EgoBlur 人脸；其清单、完整许可、来源与转换方法见 [模型说明](../src/vision_shield/detection/assets/README.md)。此权重进入 Git 和应用镜像，不适用上段离线权重的交付方式；其余三份模型仍从离线目录读取。
