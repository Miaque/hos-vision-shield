# 专项诊断脚本

这些脚本只保存在 `feature/current-stream-stall-audit`，不进入 main。它们针对本次任务ID和本机端口，复用时先核对文件顶部的任务ID、输入/输出URL和数据目录，不自动停止任务或部署。

运行环境：项目 `.venv`。先将本目录Python脚本复制到独立的 `data/diagnostics/<本轮目录>/`，分析器以脚本所在目录为数据根。原始采集不提交Git；完整材料的文件大小和SHA256保留在 `../evidence/raw-artifact-manifest.json`。本轮原始数据位于 `data/diagnostics/stall-3c66326a-20261029/`，约159MB。

- `capture-live.py <输出目录> <秒数> [任务ID] [HTTP地址]`：HTTP GET任务与诊断，默认600秒，文件上限96MiB。默认仍为本轮旧任务及`http://127.0.0.1:18000`，复用时显式传入当前任务；按墙钟检查停止，可超出一轮请求时间。须先确认服务器DEBUG已开启；查询实时诊断会启动有20秒闲置退出的内部只读接收器。
- `capture-rtp.py <绝对输出目录> <秒数>`：原流/脱敏流并行只读RTSP/TCP接收，包含keepalive、完整图片VCL哈希，单流文件上限64MiB；不解码、不发布。使用前编辑顶部URL。
- `phase-analysis.py internal baseline`、`phase-analysis.py interleave-b 100000`、`phase-analysis.py restored-a restored`：分别分析三组；先验证服务器时钟身份，然后用VCL哈希与媒体/RTP偏移匹配。
- `supplement-analysis.py <分组目录名>`：队列、主机计时增量、资源及PTS修复统计。它的非process/render操作总量只含>=100ms选择样本。
- `analyze.py`：本轮早期baseline与observed数据的补充分析，有明确目录约定，不是通用命令行产品。
- `mux-repro.py`：**仅本机隔离实验**，用临时localhost RTSP接收端及真实PyAV复现音频稀疏时交织等待，比较默认、flush_packets及100ms交织选项。无需访问服务器。
- `export-evidence.py`：按本轮三个固定Git版本导出指纹核对、CSV、摘要与HTML；输出目录固定为本规格的evidence。更新数据后再运行，不能套用到未知版本而跳过指纹检查。

小型自检包含H.264 NAL/FU-A/STAP-A哈希一致性、分位数/阈值计算和运行源码指纹断言。原始画面、包内容、权重文件不进入Git。本轮时间线只保存元数据和哈希。

生产可复用的代码位于 main 的 `media/live_diagnostics.py`、`media/rtsp_observer.py` 及对应API，受DEBUG控制。100ms实验分支仅用于对照；不从采样脚本隐式启用任何实验参数。
