# 阶段一专项采样

只在实验分支保留，不合入main，不执行部署或任务停止。原始数据在本机`data/diagnostics/rtsp-interleave-phase1-20261029/`，约68MB；Git只保留精简证据和清单。

- `capture-av-acceptance.py <输出目录> <秒数>`：并行消费原流/脱敏流，记录可用PTS/DTS、实际解码、AAC信号强度及少量画面。单流32MiB上限。它没有播放器呈现时钟，不能把解码完成称为渲染。
- `capture-rtp-av.py <输出目录> <秒数>`：同一RTSP会话订阅视频/音频，保留实际视频图片、AAC AU哈希和RTCP sender report；TCP接收缓存256KiB、视频图片8MiB、每流输出64MiB上限。有keepalive。AAC只接受该测试流SDP的13位size/3位index非分片格式，无法解析时明确报错，不默默生成音频统计。
- `acceptance-analysis.py`：本轮固定的`default-current-load`和`candidate`目录分析。将它复制到本机数据根，并从上一规格的experiments复制`analyze.py`、`supplement-analysis.py`及`phase-analysis.py`作为同目录依赖，先执行两组phase/supplement分析，再运行。它核对对应Git blob指纹，输出窗口帧率、长停顿频率、进度及独立播放器观察。

任务ID和URL在采样脚本顶部，复用前必须核对。本轮输入为本机18555，脱敏任务为`27c39007-0aee-4185-b8fa-9d1ece4de058`；修改采集目标不代表获得停止其他任务或部署的授权。

负载对照还使用上一规格的`capture-live.py`和`capture-rtp.py`，将目标ID替换为本轮恢复任务。负载采样对`GET /v1/tasks`每15秒保留当前/排队任务的ID、状态和遮挡模式。服务器内接收器由现有DEBUG接口按需启用，结束查询后自行闲置退出。

脚本只在本机读流、存储证据。`source_pts`、服务器单调时钟、本机perf_counter及墙钟保持分列；没有跨主机相减。采样上限和边界帧不能被解释成业务丢帧。
