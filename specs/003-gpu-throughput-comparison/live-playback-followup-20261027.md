# 浏览器计数解释与补人脸长尾定位（2026-10-27）

本轮服务运行代码为 `7f0014f`；报告提交 `6a71203` 与其运行代码相同，固定算法对照仍为 `581c175`。
继续使用原 G3 `86ecd4f1` 和 rfm `d22ec295`，未创建、停止或重建流任务。
原任务为 mosaic，输入输出 960×268 / 640×480。当前目标仍未通过，不以浏览器少丢帧替代稳定 FPS。

## 为什么 d22ec295 的 droppedVideoFrames 少

该字段来自浏览器 HTMLVideoElement.getVideoPlaybackQuality()，不是服务器统计。
[规范定义](https://w3c.github.io/media-playback-quality/)包含浏览器解码前丢弃、解码后错过显示期限而丢弃的帧，
在媒体重新 load 时归零。没有收到新帧而等待、上游未产生的帧、服务端缺 PTS 丢弃，都不能直接从它读出。
例如等待 2 秒但之后继续播放，仍可能只增加极少 droppedVideoFrames。

浏览器最初约 271 秒的日志中，rfm 为 dropped=12、total=6573（约 0.18%），
同时出现 13 次 waiting，呈现回调相邻最大间隔约 2.503 秒；少丢帧与卡顿并不矛盾。
G3 同期 dropped=2644、total=3828，而呈现回调已有 3792 次；不能把 total-dropped
直接认定为已验证的有效呈现数。requestVideoFrameCallback 的 presentedFrames 指提交合成的计数，
并不保证画面已经物理显示；主线程繁忙也可能漏收到回调，见 [回调定义](https://wicg.github.io/video-rvfc/)。

现场截图一度显示页面已滚到统计区，G3 离开视口，rfm 仍部分可见；另有页面隐藏采样。
[Chromium 合成器](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/platform/media/video_frame_compositor.cc)
会在正常模式下对尚未被消费的前帧报告 OnFrameDropped，后台模式采用不同规则。
这支持将页面/元素呈现条件作为独立影响因素，不能据累计差值推断服务器丢失了几千帧。

随后回到页面顶部，两次确认两路视频均完整位于视口内、文档可见，约 233.776 秒的计数增量为：

| 流 | 新增 totalVideoFrames | 新增 droppedVideoFrames | 比例 |
| --- | ---: | ---: | ---: |
| G3 | 3486 | 11 | 约 0.316% |
| rfm | 5829 | 1 | 约 0.017% |

两次检查间各自 segment 均未改变，未刷新或重建播放器；原累计值保留。
这支持先前巨大差值受呈现条件影响，但两次端点检查不证明整个区间每一刻的元素可见性，
也不能从一次观察推出两路不同编码、源时间轴和浏览器合成条件的唯一因果解释。
原始数据在 `data/validation/route-a-browser-20261027-02/` 的 both-visible-start/after.json。

rfm 后来出现一次 loading_complete 后自动重新打开原 URL，浏览器 segment 由 1 变成 2，
计数随 load 重置；故不同 segment 的累计值不能直接相减。服务端 attempt 也单独记录，
浏览器 segment 不等同于服务端 attempt。窗口评估仍以服务端 attempt 起点为准。

## 浏览器和混合文件证据边界

浏览器使用固定 mpegts.js 1.8.0、播放速度 1，关闭 liveSync 和跳播追延迟；仅清理已播放的旧缓冲。
后续还发现页面隐藏时两路视频均 paused，恢复 play 而未重新 load；该中断保留于
-02/playback-resumed.json，不能将恢复前后相加称连续播放通过。
保留 waiting、播放/暂停、重连事件、每秒质量快照、逐次呈现回调和原服务端观测。
当前计数只是合成/显示阶段证据，不能据此认定源采集至播放的绝对延迟。

第一段本地采集进程随后中断，browser.jsonl 留下约 271 秒已落盘数据，
原两条流及浏览器视频元素继续运行；恢复后另存 -02，不拼接成无缺口的 30 分钟证据。
补采已改为独立隐藏进程，继续接收同一页面的统计。原始时间间隔保留，未补造缺失记录。
浏览器播放时另提交四份文件，均成功；中断时视频结果下载未完成，
随后按原 task_id 恢复查询及下载，未重复创建任务，半成品保留并以 recovered 文件名另存完整结果。
完整核验为 1776/2450 帧、原尺寸、全部原 PTS、943 AAC 包及其内容/时钟相同。
四份输出与已验证对照 SHA256 一致，具体数据在 route-a-browser-20261027-01/file-verification.json。
中断后的首次成功查询只给出终态观察上界，不把它作为精确处理或排队时长。

## 下一优化方向：补人脸模型长尾

恢复后的每秒观测同时保存现有 Server-Timing 和检测计数，按同一 attempt 的服务器
sampled_monotonic 括住目标帧，而不是拿客户端时钟或任意前后行归因。
G3 attempt a929a2b2、sequence=15622 的两次快照只增加一帧：

- 帧包收到至成功输出 3.381 秒，其中解码后排队约 1.687 秒，开始处理至成功输出约 1.694 秒。
- 同一帧 frame 约 1692.327 ms、inference 1678.848 ms；forward-detail-face 1626.940 ms，
  local-confirm 1622.529 ms，encode 1.030 ms。阶段嵌套，不相加。
- 主车牌/补车牌前向约 19.372/12.130 ms，session-detail-face 约 0.056 ms。
  当前证据优先指向局部补人脸执行的长尾，不支持把这次等待归于编码或 session 构造。

完整原始括区间保存于 -02/single-frame-15622-stages.json。
尚不能仅凭 host 墙钟区分 GPU 执行、隐式等待和输入形状首次运行成本，也不能断定是硬件算力上限。

源码中 NumPy 局部确认走动态 YuNet CUDA session，全图 Tensor 走有界固定形状 session。
独立本地探针因此比较动态普通 Run 与现有四项固定形状缓存，保持输入、权重、FP32、CUDA
和确认解码不变，不重新尝试已无整体收益的 CUDA Graph 或固定 I/O 方案。
GTX 1080 上 64 对交错调用的输出在 1e-5 绝对差内、解码决定一致；
动态总 2.256 秒，固定缓存总 2.559 秒，后者 49/64 次未命中，未形成整体收益，未改入主干。
动态最大约 0.145 秒，未复现 RTX 的 1.6 秒长尾；该结果不证明 RTX 的根因或收益。
证据 route-a-detail-face-numpy-20261027-02/report.json；第一探针因未处理合法 None 解码结果中止，
单独保留 probe-error.json，不计有效测量。

下一步应先在实际长尾中核对局部输入形状、普通 Run 的隐式等待及 GPU 执行成本，
再选择最小执行修复。不得削减局部确认、放宽阈值或降低分辨率来消除这个等待。


## 独立 RTX GPU 诊断补充

为区分 host 墙钟与 GPU 成本，使用已有 `X-Detection-Profile: forward-detail-face`，
以原始完整人脸视频另建诊断文件任务 `d722906e-3521-4741-9e29-bfcee0469ee1`；
原两条实时任务继续运行，不改变它们的配置。这段会引入 profiler 与同步开销，
从 manifest 标记起属于独立诊断负载，不将其 FPS 当正式吞吐结果。
证据 `data/validation/route-a-detail-face-ada-profile-20261027-01/`。

完整 2450 帧诊断任务成功、dropped_frames=0，输出 SHA256 与此前已完整核验的
`98e50a7b...ba5124` 相同，证明该文件的全部输出字节等价；媒体处理 91.307 秒，
模型加载另计 2.275 秒，这些诊断耗时不用于声称性能提高。
四份模型权重哈希与本地资产一致，十份关键运行源码 SHA256 与 `7f0014f` 的 Git 规范 blob
逐项一致。使用规范 blob 避免将 Windows CRLF 与 Linux LF 的差异误报为部署漂移。
实际 RTX 6000 Ada、Torch 2.14.1+cu126、CUDA 12.6、ORT 1.20.2，
主人脸与补模型为 CUDA provider，主车牌参数 FP16；配置 quality_yolo/CUDA、solid、3/flow。
这份元数据证明本次诊断 worker 的实际状态，不替代原实时任务自身的模型运行证据。

补人脸前向总计 3890 次，host P50/P95 约 3.613/5.445 ms；
发现阶段的局部确认 650 次，P50/P95 约 2.027/40.977 ms，
跟踪阶段内的局部确认 25 次约 39.589/51.750 ms。不同用途单独保留，不能只报全模型平均值。

| 实际 forward 序号 | 用途 | host 前向 ms | GPU busy ms | kernel 累计 ms |
| --- | --- | ---: | ---: | ---: |
| 64 | 全图 discovery | 5.361 | 0.537 | 0.490 |
| 424 | 全图 discovery | 5.327 | 0.542 | 0.490 |
| 768 | 全图 discovery | 4.685 | 0.563 | 0.511 |

三份原始 CUDA trace 已下载，采样实际都属于全图调用；未命中局部确认或实时 1.6 秒长尾，
因此不能拿这三个亚毫秒 GPU 数值解释那次慢调用。host/gpu 指标与嵌套局部确认计时分别报告。
下一轮最小补充是针对实际慢局部调用关联输入形状、线程及等待证据，再决定是否调整执行路径；
当前没有据此新增缓存、替换模型或削减确认规则。浏览器长观察仍保留暂停、可见性变化、
重连和本地采样缺口，正式连续播放与稳定逐路 FPS、质量验收均未通过。
