# 保存点验证（2026-10-19）

- 命令：`.venv/Scripts/python.exe -m pytest tests/unit/test_yolo_detection.py tests/unit/test_yolo_comparison.py tests/unit/test_detector_selection.py tests/unit/test_inference_device.py tests/unit/test_worker_models.py tests/integration/test_yolo_model_probe.py -q`
- 结果：21 passed in 9.31s；日志 data/validation/yolo-setup/checkpoint-tests.txt。
- compileall：YOLO 检测器、比较脚本和时间戳修复草稿语法检查通过。
- git diff --check：通过。
- 两份实际 ONNX 权重哈希及 CPU forward 已验证。OpenCV NonZero 不兼容由 ONNX Runtime CPU 执行器解决。
- 广泛回归未取得最终输出，不计通过；全量媒体比较、修复副本像素一致性和浏览器检查未完成。
- 人工标注未完成，不能宣称准确率或效果提升。详细待办见 progress.md。
