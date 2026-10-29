# E0 本地验收与覆盖部署审查

2026-10-29，UTC+8。本文件是部署前交付；**没有部署授权，没有执行部署、推送或恢复操作**。

后续更新：用户随后明确授权部署，E0已部署并完成观测；当前状态及结果见[E0实机报告](e0-server-report.md)。本文以下内容保留部署前审查记录，不作为最新运行状态。

## 固定版本与结论边界

- E0 候选：`21bd631d5a3d02de052974b57f73428560944804`。
- 本地新 tag：`cand-20261029-stream-e0-01`，指向上述提交，尚未推送。
- 基线：`1c9be543c30fed7a2f4263b85c447ba079d438f8`。
- 本地已验证的恢复 tag：`fix-20261029-rtsp-interleave-100ms-01`，精确指向基线；本轮未通过服务器 Shell 或远程 Git 查询其服务器状态。
- 分支：`feature/stream-e0-observations`。未合并 main、未改历史 tag、权限、部署脚本或外部原流。

**专项本地验收通过；完整单元套件仍存在一项接收器退出测试失败，不能报告全套全绿。** E0 已具备提交审查和申请受控实验的条件；其 RTX 开销、真实可避免等待量与计算瓶颈还需授权后的实机采样。没有准备 E1/E2 处理行为修改。

## 源码差异

候选改动23个文件，其中9个应用源码文件；其余为环境开关、说明、专项脚本和测试。可本地审查：

```powershell
git diff 1c9be543c30fed7a2f4263b85c447ba079d438f8 21bd631d5a3d02de052974b57f73428560944804 -- src/vision_shield compose.yaml .env.example deploy/compose.env.example
git show --stat 21bd631d5a3d02de052974b57f73428560944804
```

| 文件/位置 | 新增观测 | 不作为何种证据 |
|---|---|---|
| `media/stream_pacing.py` | 对每次实际决策保留原媒体间隔、限幅间隔、previous due/actual、media/anti-burst due、重设判定时刻、最终 due、remaining、实际等待起止与停止结果 | 不更改公式或关闭等待；观测运行开销本身不可能为零 |
| `media/stream_e0.py` | 全部Pacer累计计数、条件反事实等待分解、固定容量元数据环、原始/修复/输出时钟链、各线程独占主机预算 | 条件重设贡献不等于真实运行E1后的吞吐收益；主机墙钟不等于GPU内核时间 |
| `media/streams.py` | demux包、decode帧、FIFO full/入队/出队、时钟修复、render、encode、音频/视频mux、EOF/error/drain/close/join边界；保留VCL哈希关联 | mux成功只表示封装器接受，不表示RTP已经到达 |
| `media/plate_lookahead.py` | 进入、media/wall截止、释放原因、深度、未完整前视窗口及补查区间 | 不更改.5s媒体窗口、.75s wall deadline、16帧上限或补遮决策 |
| `detection/timing.py`、`detection/quality.py` | 既有stage加入按线程嵌套扣除；单独计量两处future.result等待 | 不新增CUDA同步；隐含同步仍属于具体host调用，不能进一步声称已定位设备内核等待 |
| `media/live_diagnostics.py` | E0门禁、20个源码指纹、事件大小控制、恢复原observer | 不依赖短调用选样计算总体成本 |
| `jobs/tasks.py` | 来源校验、发布准备、媒体执行、首有效输出接收、关闭及退避边界；连接历史保留最终E0累计摘要 | 父进程收到事件的时间与子进程实际输出时间分别保存，不能混为一谈 |
| `config.py`、Compose、示例env | `STREAM_E0_DIAGNOSTICS=true`，须同时DEBUG和STREAM_DIAGNOSTICS开启才生效；可关闭E0 | 不改变输入/前视/编码队列容量、模型或媒体配置 |

编码器配置、CUDA帧处理、媒体PTS修复规则、帧序、检测与跟踪逻辑和100ms交织选项保持原有路径。人脸/车牌模型及LK均未做优化。部署脚本、`media/video.py`、`media/cuda_frames.py`、`media/stream_metrics.py`与基线逐文件无差异。

## 数据完整性与口径

- 每个attempt的`live.e0.pacer_totals.decisions`覆盖全部已完成Pacer调用，包括不等待及停止中断；不从环形样本推算总量。
- `conditional_rebase_request_seconds`：在观测到的就绪时刻和前次实际发布时间保持不变、保留半帧防突发下限时，不再重设计划时间可少请求的等待。`conditional_rebase_wait_seconds`只计实际已等待的相应区间；停止提前返回、剩余防突发等待、剩余媒体等待及定时器超时分别计量并验算总和。这是条件归因，不能保证实际E1保持相同就绪序列。
- 内存环上限：帧元数据256、Pacer128、音频输入128、视频包128、音频输出128、生命周期64；encoder身份映射受既有512个待匹配帧保护约束。元数据不持有原始画面、音频内容或GPU张量。
- `ring_discarded`是内存环淘汰；`wire_omitted`是响应字节预算省略；`discarded`是累计记录数减当前响应保留数。逐帧采集器按frame_id/revision保存更新，其他记录按序号去重；缺失覆盖量单列。
- E0响应小于128KiB，与旧观测合并的device_profile事件小于192KiB；总事件仍受现有256KiB子进程管道约束。所有报告限频沿用既有每秒至多一次。
- 各线程独立栈，父区间减直接子区间；主线程按计算调用、显式同步/future等待、Condition等待、Pacer实际等待、编码、mux、report及未覆盖区间列账。主线程总和加未覆盖时间必须与同窗口wall时间闭合，异步工作线程不相加到主线程预算。
- EOF最终累计摘要进入`stream_end.e0`及`attempts[].e0`，历史仍最多32个连接；指纹随摘要保留。跨重连不能相减不同attempt的计数器。缺少最终摘要或进程被强制回收时，明确存在尾部缺口。
- 输入/音频raw记录是worker实际从PyAV demux/decode取得的时钟；input_clock是读线程修复后，normalized_clock是主线程最终时间，encoder_frame/packet_clock是编码实际时基，音频输出保存减origin前后时钟。这能核查异常是否进入worker及发布计算。**E0没有新增AAC RTP/RTCP接收器**，不会据worker时钟正常就断定接收端重建算法错误；完整AAC根因仍需同AU、同连接/SSRC的双边证据。

## 本地验收结果

| 检查 | 结果 |
|---|---|
| 固定`1c9be543…`基线、E0关闭、E0开启，1000次相同抖动/极短PTS/大跳变/回退序列 | due、actual和等待请求逐次完全一致 |
| 停止打断、定时器超时、.5s等待上限、环淘汰及嵌套/跨线程预算 | 定向断言通过 |
| 真实CPU编解码开启/关闭对照，AAC/G711转码、B帧、单/双轨时钟回退 | 输出包字节、PTS/DTS/time_base一致，既有像素遮挡及音画检查通过 |
| CUDA路径合同测试 | 前视、最终帧序、原始/编码时钟及全部Pacer身份保持；不是实际RTX运行 |
| 本机RTSP/TCP真实接收，连续/时钟重置两组开关对照 | 2项通过；视频/AAC字节与轨道时钟一致，RTP/RTCP映射差<2ms |
| 最终专项单元命令 | **71 passed**，13.54s；包括采集器只用GET和残缺Pacer环仍从累计计数归因 |
| 完整unit，首次运行 | 910 passed、96 skipped（当时尚无后补采集器/差分测试） |
| 完整unit，最后一次记录 | **1 failed、911 passed、96 skipped**；失败为`test_real_rtsp_reader_is_bounded_and_stops_its_socket`中的接收线程退出时序断言 |
| 接收器文件核对/单独复跑 | 接收器与其测试同基线无差异；单独3项通过。未改该用例、延长超时、跳过或删除断言，未宣称失败已经根治 |
| 额外定向复跑 | 推理设备及CUDA合同55项通过；新增分析器累计预算测试3项通过 |
| 静态与配置 | py_compile、git diff --check、Compose YAML及E0开关映射通过 |

RTSP测试锁定了两配置均存在的固定素材teardown少一个最终AAC AU现象，未以测试通过宣称完美AAC交付。96项既有GPU条件用例跳过，不把CPU/合同检查替代RTX或播放器、真实遮挡质量验收。

本地开销基准3轮、每轮2000帧，每帧含24个嵌套stage调用：增量CPU中位约 **0.328ms/帧**；E0快照中位 **2.694ms**、最大 **4.801ms**；与旧观测合并快照中位 **9.763ms**、最大 **10.348ms**；合并序列化响应 **189441 bytes**；tracemalloc保留约 **1.095MB**、峰值约 **1.140MB**。快照最多每秒一次。它不包含实际GPU竞争、Redis、子进程管道与API往返，不能当作RTX低干扰的最终证明。

复跑命令及原始日志哈希见[e0本地证据](evidence/e0/local-validation.json)、[开销基准](evidence/e0/local-overhead.json)。

## 当前服务器的可验证基线

本轮仅通过已授权HTTP GET核查任务`27c39007-0aee-4185-b8fa-9d1ece4de058`及诊断：`running/ready`，17个运行源码SHA256全部匹配`1c9be543…`，mux实际选项为`{rtsp_transport:tcp,max_interleave_delta:100000}`，`solid/3/flow`、音频开启、输入FIFO=2。

[核查原始值及采集Unix时间](evidence/e0/baseline-verification.json)保留了attempt、有效配置和时钟身份；这是模块/参数核验，不声称读过服务器Git HEAD或完整文件系统。[候选20模块指纹](evidence/e0/candidate-source.json)是本地预期值，当前还未部署生效。

## 覆盖部署的影响与恢复

部署脚本会构建镜像并更新整个`vision-shield`应用栈，不能只更新一个观测函数。现有文件/流worker、API等可能重建，流任务会恢复并产生新attempt；客户端可能断开、重新缓冲，正在执行的任务可能中断重试。没有无停机或恢复耗时保证。外部原始流不修改；任务地址、业务请求合同和现有媒体队列保持，但DEBUG下E0会随现有流一起启用，增加CPU、管道与Redis观测流量。

以下指令**仅供审批，未执行**。当前规则只允许HTTP访问；若由本agent执行部署，需要用户另外明确授权下面这一条既有部署脚本命令，不能据此执行任何其他SSH/Shell操作。也可由用户自行执行。

先将固定本地候选tag发布到Git远程（本轮尚未推送；需要授权）：

```powershell
git push origin refs/tags/cand-20261029-stream-e0-01
```

覆盖部署：

```powershell
ssh -T gpu-server "cd /data/hos-vision-shield && sudo -n /bin/bash /data/hos-vision-shield/scripts/deploy-server.sh --tag cand-20261029-stream-e0-01"
```

如需恢复，使用原100ms版本，**恢复也必须取得单独明确授权**，不自动执行：

```powershell
ssh -T gpu-server "cd /data/hos-vision-shield && sudo -n /bin/bash /data/hos-vision-shield/scripts/deploy-server.sh --tag fix-20261029-rtsp-interleave-100ms-01"
```

不修改远程sudo、SSH、Git权限或部署脚本；tag不可用、脚本拒绝或部署失败时停止并报告，不改用其他远程命令。

## 授权后的HTTP生效检查与采样

1. 等HTTP入口恢复，确认目标任务`running/ready`；不自行停止/创建任务。若任务未恢复，先报告状态。
2. 读取`GET /v1/tasks/{id}/diagnostics`：20个源码哈希逐个对固定候选Git blob核验；`effective_config`保持原任务参数且E0=true，100ms mux不变；E0 schema=1，attempt指纹和boot ID/namespace一致。
3. 同一attempt中，确认全部决策累计单调递增、wait分解闭合、FIFO计数和帧身份匹配、主线程独占预算闭合；检查新attempt最终摘要可回溯。
4. 生效验证后采集**600秒墙钟**；每轮GET尽量间隔1s，实际耗时按记录报告，不把请求轮数当秒数。文件上限96MiB。若600秒内没有EOF，不人为制造；覆盖更多连接需要另定观察窗，不修改原流。

```powershell
.venv/Scripts/python.exe specs/006-stream-backlog-pacing/experiments/e0_capture.py data/diagnostics/e0-server-20261029-01 --ref 21bd631d5a3d02de052974b57f73428560944804 --seconds 600
.venv/Scripts/python.exe specs/006-stream-backlog-pacing/experiments/e0_analyze.py data/diagnostics/e0-server-20261029-01/e0.jsonl 21bd631d5a3d02de052974b57f73428560944804
```

采集器在源码/配置不符、计数回退、时钟身份变化、连续3次HTTP失败、15秒没有新内部RTP观测、96MiB上限或600秒结束时停止；它只停止本机采集，不停止任务、不部署/回滚。单轮请求可能使时长略超600秒，最终按真实时间报告。观察到健康降级、音画/遮挡异常或显著恶化时立即报告，先申请恢复授权。停止HTTP查询后内部只读接收器沿原有20秒闲置退出；E0随服务继续运行，关闭开关或恢复也需要重建及授权。

## 服务器E0后如何决策

最终报告应分别给出：

- 累计覆盖的Pacer次数、迟到重设次数；条件重设新增请求/已等待区间的次数与秒数，以及防突发、媒体节奏、定时器超时各自贡献。存在采样缺口时，累计与逐帧覆盖分开。
- 主线程最大的非重叠host调用、显式同步/future等待、输入等待、前视补查、编码、mux和report成本；其他线程另列，未覆盖区间不可强行归入GPU。
- 同一稳定输入时间段的raw/normalized媒体推进、落后增长斜率、FIFO纯队列等待、满载/背压时长和前视驻留趋势；不只比较平均FPS。
- 实际内部RTP到达的1s/5s低窗口FPS、间隔P50/P95/P99/max、500/1000ms停顿及跨连接空窗；保留播放器未观测、AAC接收未增加的限制。
- EOF前后demux、decoder drain、final render、encoder flush、close/join、worker退避、下次来源准备和首有效mux的各段贡献；输入异常在raw、修复或encoder哪一步出现。

若重设贡献显著，先准备E1单变量候选，再单独授权真实对照；若非发布等待成本已经超40ms，针对最大的独占host计算/同步调用准备独立优化实验。E1实际收益不能由条件分解直接相减预测。只有这些路径均不足且真实音视频/前视正确性和解耦收益成立后，再考虑E2。不得删光等待、扩大缓存、隐性丢帧或重复帧。

**当前没有实机E0结果，不能给出可避免等待的真实服务器总量或决定已应实施E1/计算优化。阶段二持续积压与播放器缓冲仍未解决。**
