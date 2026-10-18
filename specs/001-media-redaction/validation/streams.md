# RTSP验证

入口：tests/e2e/test_stream_recovery.py，真实API、Taskiq、Redis、MediaMTX、FFmpeg源及官方双模型。
320×240、4fps输入，经另一PyAV客户端读取输出帧；其中一轮带48kHz AAC，其余无音轨。
每轮依次断开/恢复源、强杀/替换stream worker、停止后再次重启worker；三轮重复的最终结果见acceptance.md。
worker替换以Redis中generation增加为稳定断言，不要求轮询恰巧看到短暂recovering状态。

五项发布隔离探针覆盖已批准但ANNOUNCE未结束、批准后尚未RECORD、旧worker控制租约消失但媒体连接仍活着，后者重复3次。
应用使用同一协议：冻结发布门、kick精确session ID并等待该session GET=404，再清空已确认ID并开放新generation。仅看路径为空不视为撤销成功。
控制接口不可用时，停止意图保留、发布门关闭，但不会虚报stopped。

2帧队列始终有界，满时丢旧帧，取样时选最新一帧；所有输出均经过双模型和遮挡，不存在原始流旁路。
AAC保持编码包/原时钟，当前超过16秒或4MiB缓存则重连。4秒为640检测的历史校准值；机器人1280检测较慢，已调整为16秒。单元测试仍以2秒验证配置上限。
输入容器存活到输出mux结束，避免读线程失败后使用已关闭的音轨模板；RTSP探测时间缩短到0.1秒。

当前检测最长边1280、CPU两线程，每侧外扩15%，短时补遮挡每次会话重新建立。640历史速度见performance.md，机器人对照见robot-media.md；本机CPU不足以保持4fps全帧输出。回归中AAC轮次增加为1728×536超宽输入，不能以小尺寸测试替代该验证。
本机启动脚本已验证2个stream worker注册；不能据此推导两路真实摄像头吞吐。用户真实3840×1080源的播放、首帧无PTS兼容与效果问题见[真实RTSP实测](real-rtsp.md)；长时间音画漂移和多机共享目录未验收。
MediaMTX所有录像选项关闭，应用流分支没有落视频文件。
