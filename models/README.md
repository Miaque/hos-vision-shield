# 模型文件

已选择Project Aria官方组织发布的EgoBlur Gen1权重：
https://huggingface.co/projectaria/EgoBlur/tree/9c0b319ad09d0346e8294d7bf569593d3ab865c0

固定revision下载ego_blur_face.zip与ego_blur_lp.zip；压缩包SHA-256必须与官方LFS元数据一致，
解压后分别保存为face.jit与license_plate.jit，原文件名及权重哈希记录在manifest.json。
权重与压缩包不提交Git。Apache-2.0许可及原始下载来源随manifest保留。

验证：`uv run pytest tests/integration/test_model_probe.py -q`。
加载成功只证明权重可运行，不证明普通监控或中国车牌的准确率。

当前检测固定阈值0.1、NMS IoU0.3；默认输入最长边1280、每侧外扩15%，可通过DETECTION_MAX_SIDE和DETECTION_BOX_PADDING配置。两模型同时执行，不提供目标类别选择。CPU推理关闭TorchScript运行时优化，避免冷帧分析开销；640的历史性能基线见[性能报告](../specs/001-media-redaction/validation/performance.md)，机器人素材对照见[补充验证](../specs/001-media-redaction/validation/robot-media.md)。
