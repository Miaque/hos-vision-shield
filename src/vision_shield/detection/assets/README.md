# YOLO 人脸检测权重

`yolov8n_face.onnx` 随 Python 包和应用镜像交付，仅用于 `quality_yolo` 的人脸候选检测。来源为 akanametov / yolo-face 的 1.0.0 发布，原权重 `yolov8n-face.pt`；原文件与转换结果的 SHA-256、字节数及工具版本见 `manifest.json`。使用前核验哈希，启动及推理不联网下载。

上游源码：<https://github.com/akanametov/yolo-face>；原权重：<https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt>。此组件遵循 GPL-3.0，完整许可见 `LICENSE.yolo-face.txt`，并非其他模型的 Apache/MIT 许可。上游许可来源固定于清单所列 revision。转换未修改原权重数值，未训练新模型；输出中的关键点不参与处理，也不运行身份识别。

在独立的 Python 3.12 导出环境安装 `torch==2.14.1`、`ultralytics==8.4.183`、`onnx==1.23.2` 后，将上述原权重放入当前目录，执行：

```python
from ultralytics import YOLO

YOLO("yolov8n-face.pt").export(format="onnx", imgsz=1280, dynamic=True,
                             opset=17, nms=False, simplify=False, device="cpu")
```

将生成的 `yolov8n-face.onnx` 重命名为 `yolov8n_face.onnx`，按清单核验 SHA-256 后交付。

原始输出为 `[1, 20, anchors]`，前五通道为中心坐标、尺寸及人脸分数，其余为关键点。应用保持 RGB、除以255、114补边、32倍数矩形输入，按源坐标还原并作 IoU 0.3 NMS；全图最长边1280，局部人脸确认最长边160（较大的局部输入会漏掉已核对侧脸）。推理仅使用现有 ONNX Runtime CUDA，不引入 Ultralytics 运行依赖。

切回 `DETECTION_BACKEND=quality` 并保留原离线 `face.jit` 即可恢复原人脸网络。更换权重或导出工具后，必须更新清单并重新执行真实 GPU 效果与速度验证。
