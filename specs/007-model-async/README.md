# 主模型异步关键路径与车牌成本

2026-10-30方向更新：当前验收对象固定为既有RTSP任务 `27c39007-0aee-4185-b8fa-9d1ece4de058`，保留另一条流同时运行。已停止文件测试调度，两个已提交文件任务均自行完成；相关实时数据标为混合负载，不用于干净双流基线。最新结果见 [live-report.md](live-report.md)。下文文件实验为历史辅助成本证据，不是实时验收，也不继续创建文件任务。

本轮不修改模型、阈值、检测频率、前视或发布政策。服务器保留 E0 `21bd631d5a3d02de052974b57f73428560944804`；新的任务关联观测只由 DEBUG 下显式视频请求 `X-Detection-Profile: async-host` 启用。E1 在独立分支 `feature/stream-e1-no-rebase`，固定 `d6f24d0ac4679b1ea09bbc0a5efe0fe2fec99d10`，不包含在此候选。

## 证据口径

E0 实机18ms/帧是 `quality.py` 两处 `pending.result()` 的主线程独占 wall，不能再加异步线程的模型 forward。现有部署没有任务关联事件，尚不能精确拆出这18ms的 GPU/CPU/调度占比。本地是 GTX970（5.2，Gen1 float32），服务器 RTX 的 Ada 路径使用 half；不得把本机耗时按比例套到服务器。

新增关联事件覆盖 submit、worker 开始/返回、future 完成回调、result 进入/退出，及关联 model_call 的准备、forward、后处理区间。submit→worker_start 包含观测提交开销、executor排队及OS派发，不能单称线程池排队；worker_return→result退出包含观测收尾、Future发布与唤醒，不能单称OS调度。完成回调说明已经完成，时间位于状态发布之后，可能晚于result返回。

同线程嵌套区间先做差，主线程wait再与对应worker的非重叠区间求交；不同线程只报告重叠关系，不能相加为端到端耗时。forward是host调用边界，包含GPU及隐式同步；thread CPU不是纯预处理，wall减CPU也不是纯GPU。独立CUDA采样单列，不使用采样FPS作吞吐结果。

Windows Python3.12的 `monotonic` 是15.625ms tick；新观测采用单调 `perf_counter`（本机声明分辨率100ns），不改变应用已有时钟。每任务最多25000个异步任务、64MiB事件；达到上限明确失败，不截断后冒充完整诊断。已有host观测另有250000测量预算。常驻worker的已有池也能关联；诊断结束恢复模型方法与StageTimes。普通请求不安装这些钩子，也不新增CUDA同步。

## 源码已排除或收敛的方向

- `plate_proposals` 已经并行 primary face 与 detail，并缓存结果供后续 `detect_regions` 使用；不能按两个函数名认定重复执行全图 face/detail。
- uncached CUDA `detect_regions` 已共享一次上传后的源像素，primary与detail实际并行。两者的最终输入不同：Gen1为BGR uint8 CHW，HyperLPR有320尺寸letterbox/float准备，YOLO又是独立归一化。不能直接共用最终模型输入。
- CUDA `prepare`/`upload` 对已在设备上的像素仍验证并同步；YOLO再调用Gen1 plate分支时再次prepare。值得按事件核算，但跨线程/stream生产者依赖存在，不能删同步后只凭输出一次正常就认定安全。
- cached proposal分支的primary plate是在质量门控确定需要全图检测后才执行。提前做它可能变成不必要的推测计算，不是无代价的“并行化”。局部复核的目标、镜像与裁剪也必须一致才能讨论缓存，不能只按320形状复用结果。

## E1与恢复问题保持独立

E1唯一应用变量是取消明显迟到后的due重设，保留 `max(previous_release+d, previous_actual+d/2)`、.5s间隔限幅、PTS/音频/前视和停止。独立真实媒体调度测试两轮约5.99→5.35秒（10.7%减少），全套unit915 passed/96 skipped、最终相关122 passed、真实媒体6 passed（含本机真实NVDEC/CUDA/NVENC）。收益测试用固定ROI和注入成本，不是rfm真实模型或RTX结果；真实GPU测试只作正确性证据。完整报告在独立工作树 `specs/007-async-critical-path/e1/local-validation.md`。

EOF与InvalidDataError不在本候选改动内。`jobs/tasks.py:run_stream` 完成媒体协议收尾、publisher关闭后，等待 `min(30,2**retry)` 再验证来源/准备发布/重新打开RTSP；稳定输出后才把retry归零。`media/runner.py:MediaProcess` 正常复用同一个resident模型进程，`child.process` 对正常EOF与可恢复解码错误可返回reusable；媒体attempt重建不等于每次重新加载全部模型。只有配置变化、不可复用或进程终止等才重建进程并预热。

保留E0实测正常EOF最后mux到下次首mux4.155s，其中协议/事件收尾1.225s、发布关闭.123s、退避1.001s、新attempt到首输出1.709s；实际RTP空窗5.075s。InvalidDataError对应mux4.387s/RTP5.148s。新attempt的1.709s仍含发布准备、源重开、首帧计算及前视暖机，不能全部归为网络或模型加载。早期3.365/5.370s证据继续保留。AAC raw回退54次，修复后/mux时钟未回退，RTP/RTCP/同AU映射未完成归因，不混入推理修改。

## 实验入口

- `experiments/file-run.py`：完整同源真实质量模型/NVDEC/前视/NVENC，核对2208帧与694个AAC包的PTS/内容；文件路径没有E0 StreamProbe，不能用于宣称E0开关成本。
- `experiments/analyze-tasks.py`：完整序号检查、主线程独占预算、对应future等待交集、调用目的/输入形状分布。
- `experiments/kernel-probe.py`：独立固定像素CUDA采样，和端到端吞吐分开。
- `experiments/observer-cost.py`：CPU夹具的新增观测增量，不能外推GPU/HTTP/Redis成本。
- `experiments/verify-current.py`：只读HTTP核对当前E0的20个模块与配置，无SSH探查。

本轮尚未覆盖部署、推送tag、修改main、修改原流或进行E2重构。
