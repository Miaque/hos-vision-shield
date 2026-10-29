# 仅诊断候选：覆盖部署申请

## 固定版本与差异

- 候选commit：**`41ba7e3210cc9d8e0d2a34c06d77be49ef2ee0f5`**。
- 本地固定tag：**`cand-20261029-model-async-diag-01`**，尚未push或部署。
- 当前服务器：E0 **`21bd631d5a3d02de052974b57f73428560944804`** / `cand-20261029-stream-e0-01`。
- E1另在 `feature/stream-e1-no-rebase`，commit **`d6f24d0ac4679b1ea09bbc0a5efe0fe2fec99d10`**；不包含在本次覆盖范围。

相对服务器版本，应用/本地入口差异仅6文件、241行增加/14行删除：新增 `detection/async_profile.py`；`host_profile.py`关联任务并恢复常驻worker观测；`media/child.py`安装显式文件诊断；`api/routes.py`和`jobs/store.py`接受DEBUG专用 `async-host` 并提供固定目录的NDJSON下载；本地CLI增加同一模式。无模型、阈值、频率、输入队列、前视、Pacer、时间戳、音频、mux、Compose或部署脚本修改，100ms参数保持。

```powershell
git diff 21bd631d5a3d02de052974b57f73428560944804 41ba7e3210cc9d8e0d2a34c06d77be49ef2ee0f5 -- src scripts
```

普通任务不安装新观测；显式文件任务诊断结束恢复resident模型方法和StageTimes，已有线程池也被关联。每文件任务最多25000个异步任务、64MiB新事件、250000个已有计量；超限拒绝成功报告。新观测不额外执行CUDA同步。

## 本地证据与当前基线

完整unit926 passed/96 skipped；随后仅补report预算耗尽拒绝成功的守卫，相关7 passed。真实quality_yolo/NVDEC/CUDA/PlateLookahead/NVENC完整2208帧、694 AAC包，在普通/async-host两臂的最终MP4字节完全一致；单轮总墙钟差1.50%，不能作精确开销结论。该后续守卫及实测源指纹的差异记录在 `evidence/post-run-change.json`，没有改写实测hash。

完整证据和限制见 [local-report.md](local-report.md)。本地为GTX970 float32，服务器Ada half与资源竞争不同。观察到主模型等待主要覆盖真实plate forward，串行化减少wait但不改善8帧端到端时间；没有已证明的模型调度修复。

最后一次只读HTTP核验的Unix时间 `1793287367.452496`：目标任务running/ready，20个源码SHA256匹配E0，100ms、solid/3/flow、音频与FIFO2保持。新核验及候选Git blob指纹在 `evidence/deployment-source.json`。这证明可见模块/配置，未读取服务器Git HEAD或完整文件系统。

## 影响、执行及恢复

部署脚本会更新整个应用栈，现有API、流/文件worker可能重建；既有任务可能中断重试，客户端可能断开、增加积压或重新缓冲。不能承诺无停机。RTX文件实验会与既有流竞争GPU/CPU；每次只运行一个新增文件任务，按停止条件中止本轮任务。外部源不需变更。

**以下命令均未执行，需取得用户对本次覆盖的明确授权。** 请求授权范围包括发布该新tag、仅执行下列一条既有部署脚本，以及按下一节通过HTTP创建/停止本轮文件任务和读取结果；后续E1覆盖、配置切换及恢复分别授权。

```powershell
git push origin refs/tags/cand-20261029-model-async-diag-01
ssh -T gpu-server "cd /data/hos-vision-shield && sudo -n /bin/bash /data/hos-vision-shield/scripts/deploy-server.sh --tag cand-20261029-model-async-diag-01"
```

如需恢复当前E0版本，以下是精确指令；**恢复也另行授权，不自动执行**：

```powershell
ssh -T gpu-server "cd /data/hos-vision-shield && sudo -n /bin/bash /data/hos-vision-shield/scripts/deploy-server.sh --tag cand-20261029-stream-e0-01"
```

脚本、tag拉取或权限失败时停止并报告；本次申请不包含额外SSH/Shell探查、修改权限或`.env.compose`。本地/远程已有tag均不重写。

## HTTP生效检查与RTX文件对照

1. 等HTTP入口恢复，读取既有任务状态及诊断；与固定commit的20个源码blob逐个比对，核对100ms及原有效配置。OpenAPI应出现 `async-host` 与 `artifact=async-tasks`；API合同出现本身不能证明worker已更新。
2. 同一源文件SHA256 `846204eb03f4cd0aeec78f22e1fa1ca4bf0ab2bb2fd866c9012bc32f956fb42a`，通过上传接口顺序执行 **normal→async-host→async-host→normal**，总共4个完整文件任务。统一 `X-Detection-Interval: 3`、`X-Tracking-Mode: flow`、solid；普通臂省略profile头。只在上一个任务结束并检查结果后提交下一个。
3. 每个任务核对2208帧/零丢帧、694个AAC包及完整PTS/帧序。显式任务另核对GPU UUID、权重/源码hash、精度、实际配置、模型复用和 `async_tasks.source_sha256`；普通任务不返回完整运行指纹，按相邻显式任务、原请求及已有worker证据标注限制，不填造逐任务身份核验。核对任务/调用数是否因真实内容一致而可比；**不把本机4837写成必须相同的服务器常量**，结果差异需解释。下载 `GET /v1/tasks/{id}/diagnostics?artifact=async-tasks`，只在完成后取全量文件，核对末尾计数、序号及预算，按任务ID分开分析。
4. 输出优先核对整片字节、逐帧像素、音频载荷及时钟；不同先定位编码差异、检测/前视差异，不给收益结论。记录完整wall及模型/主线程独占预算，区分冷形状初始化与后续调用；前10次模型调用仅作单列统计，输出帧仍完整保留。4个任务差异在噪声内时报告无确定收益。
5. 依据关联结果中最大forward阶段，最多追加 **2个独立 `forward-*` 文件采样任务**，使用既有三调用CUDA采样，不使用采样FPS作吞吐验收。GPU busy、同步API和host wall分别报告，不重复累计；未采样的总体GPU份额不给伪精确数字。本次总上限6个文件任务。

文件任务每个最长300秒，完整实验上限35分钟；到期通过HTTP停止本轮创建的任务并报告，不停止既有任务。共享流状态/已有E0诊断尽量1秒查询一次，按真实观测时长计量；资源快照以接口实际可用为准，不把缺失曲线补成GPU利用率。

以下条件立即停止提交任务，能访问API时停止本轮活动文件任务并报告：既有流失败或not-ready持续15秒、15秒无有效RTP输出、连续3次HTTP失败、模型/CUDA错误、诊断预算触顶、输出帧/PTS/音频/遮罩不一致。停止文件实验不等于恢复服务；需要恢复时先请求前述E0恢复授权。

本次4臂文件路径没有E0 StreamProbe，**不冒充E0开/关对照**。关闭E0影响实时流配置且需服务重建，既有部署脚本没有热切换参数；另准备明确配置与恢复方案后单独申请，不能在服务器Shell临时改环境。

## 后续实时验证单独申请

RTX文件结果用于选择模型/同步的最小单变量实验，不直接作为长期流畅性结论。E1随后若申请覆盖，仍用其独立commit，保留E0计量；固定配置至少600秒，与匹配条件的E0窗口比较真实RTP间隔P50/P95/P99/max、1秒/5秒低窗口FPS、>500/1000ms停顿、媒体落后增长、FIFO/前视等待、长期延迟、音画/AAC、恢复和实际播放器缓冲。

输入PTS跳变/连接边界单列，源条件不等价时不给伪因果对照；没有播放器接收/渲染证据时明确缺项。持续积压需可复现改善，平均FPS略增不构成通过。EOF/AAC另案不混入本次代码。
