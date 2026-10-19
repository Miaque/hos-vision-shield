# CPU detection models

The CPU backend uses only object detection. It does not load face embeddings,
license-plate recognition/classification networks, character dictionaries, or OCR.

- YuNet face detector: OpenCV Zoo / Shiqi Yu, MIT. The complete license is in
  `licenses/YuNet-MIT.txt`.
- HyperLPR3 320 plate detector: HyperInspire / szad670401/HyperLPR, Apache-2.0.
  The complete license is in `licenses/HyperLPR-Apache-2.0.txt`.
  RGB letterboxing and detection tensor decoding follow the upstream
  `Prj-Python/hyperlpr3/inference/multitask_detect.py` model contract.

Exact source revisions, original download URLs, sizes and hashes are recorded in
`cpu-manifest.json`. Model binaries remain local and are excluded from Git.

## YOLO 对比候选

人脸权重来自 https://github.com/akanametov/yolo-face/releases/tag/1.0.0 ，上游仓库标记 GPL-3.0；车牌权重来自 https://huggingface.co/morsetechlab/yolov11-license-plate-detection ，模型卡标记 AGPL-3.0。固定来源与 SHA-256 见 yolo-manifest.json。保留上游许可声明，不把本地对比视为生产部署许可结论。车牌模型卡披露训练/测试污染，公开指标不能作为本项目效果证据。

权重实际 metadata 补充：yolov8n-face.onnx 内部声明 AGPL-3.0，与来源仓库 GPL-3.0 标记不同；两者均保留在 yolo-manifest.json，不将仓库标记作为权重许可的唯一依据。
