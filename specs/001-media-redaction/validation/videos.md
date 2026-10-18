# 文件视频验证

入口：tests/integration/test_media_probe.py、test_video_pipeline.py、tests/e2e/test_videos.py。
合成源320×240、12fps、24帧，H.264 B帧；分别有AAC和无音轨。VFR前12帧12fps、后12帧6fps；旋转夹具使用FFmpeg的display_rotation=90。

PyAV/ffprobe检查完整输出24帧、每个PTS与源对应，编码flush后的尾帧不丢失；AAC编码包逐包相等。
指定矩形的遮挡检查与真实模型视频链路分开，不能把合成testsrc当作检测效果验收。
旋转结果240×320，无残留旋转标签；预先取消抛出中断；MKV/3GP中即使是H.264也拒绝，损坏/时间戳非法不会透传原图。

真实双模型E2E覆盖上传/URL三帧MP4、成功下载，以及处理中取消和强制终止文件worker后的worker_lost；失败或取消结果不可下载。
性能入口另使用完整24帧AAC视频，产物在data/validation/video-redacted.mp4。原文件和失败临时文件按要求保留。
每个解码帧在转RGB前再次检查尺寸，不能依赖初始流元数据绕过宽高上限。尚未做长达60分钟或4K视频的容量/效果验收。
