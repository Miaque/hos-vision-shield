# 保存点验证（2026-10-19）

- 命令：`.venv/Scripts/python.exe -m pytest tests/unit/test_yolo_detection.py tests/unit/test_yolo_comparison.py tests/unit/test_detector_selection.py tests/unit/test_inference_device.py tests/unit/test_worker_models.py tests/integration/test_yolo_model_probe.py -q`
- 结果：21 passed in 9.31s；日志 data/validation/yolo-setup/checkpoint-tests.txt。
- compileall：YOLO 检测器、比较脚本和时间戳修复草稿语法检查通过。
- git diff --check：通过。
- 两份实际 ONNX 权重哈希及 CPU forward 已验证。OpenCV NonZero 不兼容由 ONNX Runtime CPU 执行器解决。
- 广泛回归未取得最终输出，不计通过；全量媒体比较、修复副本像素一致性和浏览器检查未完成。
- 人工标注未完成，不能宣称准确率或效果提升。详细待办见 progress.md。

## 主干合并验证（2026-10-20）

- 合并 main 的模型来源说明冲突时保留 EgoBlur/YuNet/HyperLPR 与 YOLO 的原有来源、许可及 metadata 差异声明。
- 先确认 GPU 锁一致性测试因缺少 onnxruntime 失败，以及 YOLO 启动预检的两项用例失败；补齐模型清单/CPU provider 兼容和 GPU 锁后，相关 35 项测试通过。
- `uv sync --locked` 安装本分支已锁定的 onnxruntime 1.31.0、flatbuffers 与 protobuf；未下载权重，未安装 Docker。根 pyproject.toml/uv.lock 与合并前 YOLO 分支一致。
- `uv run --no-sync pytest tests/unit tests/integration/test_runner_lifecycle.py -q`：153 passed、1 skipped，14.72 秒。跳过项为 POSIX 冷加载父 EOF，Windows 结果不替代 Linux 实测。
- GPU 部署锁重新生成 70 项，安装 dry-run 与共同版本检查通过；没有实际在本机安装 Linux CUDA 依赖。
- 主工作区没有 YOLO 的两份真实权重，未执行 test_yolo_model_probe.py；没有重跑媒体全量比较、人工效果复核或服务器 Docker 验收，不以单元测试代替。
