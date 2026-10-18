# T007 媒体探针

环境：Windows、Python3.12.14、PyAV19.0.1、FFmpeg/ffprobe本机安装。
命令：`uv run pytest tests/integration/test_media_probe.py -q`。
全量运行中6种组合全部通过：AAC/无音轨/VFR × 完整保留/丢弃一半解码帧。

单源demux、H.264重编码、AAC包直接remux，经PyAV与ffprobe分别检查：

- 完整输出24帧；丢帧输出12帧。
- 每个保留帧时间戳与输入对应时间相差不超过1ms，未按新帧序号加速时间线。
- AAC编码包内容逐包相等，ffprobe确认H.264与AAC/无音轨结构。
- 合成源为320×240、12fps、2秒，视频包含B帧；测试无真实目标检测参与。

机制探针已含VFR；真实业务与模型分别见videos.md和performance.md，未验证真实摄像头或长流音画漂移。
