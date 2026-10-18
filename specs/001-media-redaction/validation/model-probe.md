# T005 官方模型验证

2026-10-18，Windows本机Python3.12.14、PyTorch2.14.1+cpu、torchvision0.29.1+cpu。
两个Gen1压缩包来自Project Aria官方Hugging Face仓库的固定revision，压缩包与官方LFS SHA-256完全匹配，提取后权重哈希已记录models/manifest.json。
命令：`uv run pytest tests/integration/test_model_probe.py -q`。
结果：1 passed；两个真实权重均在CPU成功加载，并分别对官方人脸、OpenCV车牌样本执行推理，输出框/类别/得分维度匹配且数值有限。

PyTorch报告TorchScript加载弃用提示和旧模型meshgrid将来需要indexing参数提示；未屏蔽警告，当前固定版本可运行。
该验证不等于真实监控或中国车牌准确率验收。
