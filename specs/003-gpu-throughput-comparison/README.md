# 双路 GPU 吞吐与技术路线对比

目标为 RTX 6000 Ada 上双路实时视频各稳定至少 15 FPS，继续争取 30 FPS。
离线结果不能替代实时验收。当前已完成代码审计、工具验证和两条路线的本地人脸整片诊断，
RTX 6000 统一 P0 及最终路线选择尚未完成。

## 固定版本及实际配置

| 项目 | A 固定对照 | B 初始基线 |
| --- | --- | --- |
| 提交 | `581c175a0fad21efe10a0caf08a01e37b25fd3ad` | `d577ee7c5460afe18ca4efa310400e7554c70c12` |
| 运行后端 | `quality_yolo`、CUDA | **显式 `quality`、CUDA** |
| 主人脸 | YOLOv8n-face，ORT CUDA FP32 | EgoBlur face，TorchScript CUDA FP32 |
| 主车牌 | EgoBlur plate，Ada 上 FP16 参数/autocast，输出框/分数 FP32 | EgoBlur plate，TorchScript CUDA FP32 |
| 补人脸 | YuNet，ORT CUDA FP32，阈值 0.3 | 相同 |
| 补车牌 | HyperLPR detection-only，ORT CUDA FP32 | 相同，无字符识别 |
| 全图/分块 | 主最长边 1280；补全图及源图 1920 重叠分块 | 相同 |
| 调度 | `detection_interval=3, tracking=flow` | 逐帧检测，原模板补偿 |
| 扩框/遮罩 | padding 0.15，原始分辨率逐帧不透明深灰色 | 相同 |
| 池复用 | quality/补检测复用单线程池 | 每次调用/视图创建线程池 |

A **不是所有模型每三帧运行一次**：`plate_proposals` 在间隔帧也执行整帧人脸及补模型，
复用同帧候选，再按变化或跟踪失败复核主车牌。实际计数以 forward 为准，不能用帧数除三。
YOLO 的局部 160px 与全图最长边 1280 也必须分开统计。

B 基点的 `compose.gpu.yaml` 原本设置 `DETECTION_BACKEND=quality_yolo`，不能直接沿用它建立 B 基线。
本实验分支将两个 GPU worker 的部署配置显式固定为 `DETECTION_BACKEND=quality`、`INFERENCE_DEVICE=cuda`，
作为 B 基线的配置覆盖，不改检测实现；测量脚本也直接传递这些参数，
不读取 `.env`，并在初始化及完成后检查没有导入 YOLO 模块。
基点虽然包含 `_HalfPlate`，`quality` 没有调用它，不能把其 FP16 效果当作已经实现。
表内 FP32 指参数/张量 dtype，不表示禁用了 TF32 数学路径；实际全局、matmul、cuDNN 卷积策略
另记在运行报告。本地两条路线的 cuDNN 卷积策略均为 `tf32`，GTX 1080 不能验证 Ada 的对应数值路径。

权重来源和预期 SHA256 见 [基线清单](baseline.json)，每次运行另记录实际文件哈希、
实际参数 dtype、CUDA provider、设备 UUID、源码 SHA256 和工作区状态。
禁止 CUDA 失败后静默回退 CPU。

## 指定版本差异审计

`git diff --stat d577ee7 581c175` 为 19 文件、1182 行新增、34 行删除。
检测与媒体差异：quality 增加同帧候选复用及人脸路径；cpu/quality 复用执行池；
yolo 支持整帧人脸入口；redaction 增加分类降频、光流及旧轨迹复核；video 传递降频/跟踪参数；
timing 增加相关计时。测试增加降频、稀疏光流及 GPU 帧对照。
接口/任务差异涉及请求头、幂等参数及 worker 关闭资源；其余是报告及 CHANGELOG。
B 未移植这些调度实现，未合并 A，也未修改现有 main 或其未提交文件。

## GPU 路径与剩余搬运

适用的 H.264、yuv420p 输入启用 NVDEC，硬件解码器设置 `allow_software_fallback=False`。
NV12 经 DLPack 接入 Torch，颜色转换、缩放、归一化、模型、遮罩在 GPU；输出 NV12 进入 NVENC。
不支持该硬件帧分支的输入仍可能走原 CPU 解码路径，因此每次必须核对实际 decoder/encoder。

这不是全程无搬运：模板跟踪每帧下载整幅灰度图及缩略图；局部复核下载 RGB crop，
CPU 放大/镜像后再次交给 GPU 模型；ORT 输出绑定 CPU，后处理/NMS及少量框管理在 CPU；
EgoBlur 还含图内小标量及网格传输。`input-copy`/`output-copy` 不能覆盖所有框架内部复制。
`explicit-sync` 是显式等待，模型 forward 的墙钟还含隐式等待。GPU 格式转换不等于 CPU/GPU 复制。

离线 CUDA 输出使用 `h264_nvenc`，原视频全部帧/PTS和 AAC 包保留；
**实时 `add_video(live=True)` 仍选择 libx264**，尚未证明实时 NVENC。
初始化和每个输入形状首次创建 ORT 会话属于冷启动成本，现状没有独立的模型预热阶段。

异常时当前模型及媒体链路抛错、阻止任务成功，不返回部分检测结果，也不静默输出原始失败帧。
当前不是整帧遮挡保护；发生异常的未完成输出不能当作成功结果。
后续正式测试仍需保留异常注入证据及输入/处理/编码/丢弃/保护分别计数。

## 历史实验清单及重试边界

以下均为历史记录，**不能代替本次统一环境重测**；证据见
[历史调优记录](../002-compose-cpu-gpu/temporal-optimization.md)及
[早期并行测量](../002-compose-cpu-gpu/gpu-throughput.md)。

| 方向 | 既有证据 | 本轮处理 |
| --- | --- | --- |
| 两路主/补检测重叠、补模型并行 | 早期 GTX 1080 固定样例有效，B 已具备 | 保留，不重复作为新收益 |
| GPU 补检测、固定形状 ORT、NVDEC/NVENC | 已进入基点，减少过 CPU 路径成本 | 核对实际运行而非再实现 |
| EgoBlur plate 原完整图 FP16 | 在 `quality_yolo` Ada 路径有效；B 的 quality 未启用 | P0 后若成本支持，作为 B 独立精度实验，不能直接继承 A 质量结论 |
| 仅骨干 FP16/冻结 BN/CUDA Graph | 全管线收益不足；Graph 仅本地原型 | 没有新设备瓶颈证据前不重复 |
| TF32 矩阵乘 | 服务器整片无可靠收益，已撤回 | 不盲目重试 |
| ROI 1000→512 | 本地小段改善，RTX 双路整片略慢，已撤回 | 不重复以局部算子收益承诺提速 |
| HyperLPR 跨视图合批 | 补检测变快，RTX 双路端到端无收益，已撤回 | 只有 B 的新计数证明它转为关键路径时才重评 |
| 锚框网格缓存 | HToD/同步减少，RTX 整片无稳定收益，已撤回 | 同上，先测总贡献 |
| 模型常驻/跨流合批/TensorRT/NvDCF | 未完成统一双路质量与端到端验收 | 未验证，均非必做项 |
| 降频+光流 | A 有历史离线提速，也存在背景误遮及不确定车牌帧 | 固定 A 重测，不扩张质量容忍范围 |

## P0 测量方法

独立脚本 [profile-video.py](../../scripts/profile-video.py) 包装实际生产调用，**不修改生产检测器**。
用 `PYTHONPATH` 指向要测的 worktree `src`。A 会检查其 `src/vision_shield` 与固定提交无差异；
B 在本实验分支运行。所有输出必须使用新目录，禁止覆盖历史证据。

```powershell
$env:PYTHONPATH = '<B工作区>/src'
python scripts/profile-video.py --route B --mode host `
  --source '<原始完整视频.mp4>' --target '<新目录>/output.mp4' `
  --model-dir '<四份权重目录>' --report-dir '<新报告目录>'
```

计量分开运行：

- `host`：四模型每个实际 forward 的 JSONL（帧序号、源时间、全图/分块、局部放大/确认、
  镜像、跟踪复核来源、耗时、失败），并汇总次数/每帧次数、P50/P95、首次及前十次之后统计。
  并行执行器仅传递诊断上下文，不增减推理调用。各阶段为墙钟包含时间，嵌套/并行不能相加。
  编解码包装保留原 Packet、VideoStream 和音轨，另记 demux、decode、encode、mux 及关闭成本；
  这些是主机调用耗时，不声称为 NVDEC/NVENC 硬件引擎的纯执行时间。
- `device`：通过 `--device-stage forward-primary-face` 等选择四个实际前向之一，
  `--checkpoints 64,424,768` 按该模型真实调用序号采样，每份至多 32 MiB。
  复用现有 CUDA trace 相关性分析，区分关联内核、GPU 活跃并集、HtoD/DtoH/DtoD、
  主机同步/分配；缺少关联内核直接失败，不将主机等待误报为纯 GPU 时间。
- `throughput`：不安装包装器/设备采样，使用原生产处理器。媒体 FPS 与脚本测量开始至媒体处理完成 FPS 分列；
  后者从导入推理依赖前计时，不含解释器启动、输出校验及报告写入。另记录工具文件 SHA256。
  CLI 没有 API 任务/队列边界，`task_creation_to_finish_fps=null`，正式该指标另从隔离 API 测量。

首次前向与其后统计不等于所有尺寸均已预热。设备采样和 host 诊断 FPS 不计有效提速，
正式吞吐需另测无设备采样的双路整片；完整帧/PTS/AAC内容核验、连续片段及人工质量检查均不能省略。

GPU 服务器可对视频上传显式设置 `X-Detection-Profile: host`，使用同一观察器记录真实调用，
完成后从原 `GET /v1/tasks/{id}/diagnostics` 取得四模型分组次数/耗时、首次与稳定统计、
实际权重/源码 SHA256、设备/精度/执行器信息及模型加载成本。逐条记录与报告保存在该次
attempt 的 `host-profile/`。仅支持本实验的 quality/CUDA，不允许静默 CPU 回退。
上限 250000 个阶段观测，超出时诊断任务失败且不发布成功结果；三份现有整片均在上限内。
普通请求不导入观察器；原 `X-Detection-Profile: 1` 的主车牌 CUDA 时间线保持。
host 模式不与 CUDA 采样混用，诊断 FPS 不计验收；不改既有 stats 和任务 JSON 合同。
需要区分内核与等待时，独立任务将该头设为 `forward-primary-face`、`forward-primary-plate`、
`forward-detail-face` 或 `forward-detail-plate`，在该模型实际第 64/424/768 次前向采样，
不额外调用模型。`profile.device_samples` 保存关联内核、复制及同步成本，原诊断接口
`?sample=64` 等可下载对应原始 trace。调用不足三个采样点时状态为 partial，不能当作完整采样。
每份 trace 上限 32 MiB；诊断失败终止本次任务，不返回未处理帧。
373 项单元检查及 4 项真实编解码集成检查通过；本地实际子进程的非空 GPU 遮罩短片
与普通处理逐字节一致，全部 PTS/尺寸保持。最初短片诊断暴露默认遮罩参数未写入元数据的
KeyError，已按媒体处理器的有效默认值修复；失败样本保留且不计成功。

## 验收工作清单

- [x] 独立 B worktree/分支，保留现有修改；固定两条参考提交。
- [x] 核对模型、精度、调度、GPU/CPU数据路径、旧失败方向。
- [ ] P0 两条固定路线的逐模型完整诊断及单/双路 RTX 6000 统一基线。
- [ ] 根据 P0 总贡献选单项优化；异常/媒体完整性/质量回归。
- [ ] 最终候选和固定参考各至少三次双路完整视频，报告每次及波动。
- [ ] 明确实时延迟上限、丢帧策略、窗口门槛；稳定源/输出地址及时间戳。
- [ ] 候选双路至少 30 分钟，每秒/10秒有效 FPS、延迟分位、队列及资源/恢复。
- [ ] 输出、关键片段、质量判定、离线/实时报告及路线结论。

用户确认复用同一服务器顺序测试，并授权扩展部署脚本支持指定分支。
新版推荐 `scripts/deploy-server.sh --tag exp-20261023-b-deploy-02` 按准确 tag 部署及回退，
不移动本地分支、不覆盖已有 tag；也支持显式分支参数的安全切换和快进。
无参数仍仅部署 main。服务器须先取得新版脚本，旧版会忽略参数，不能直接据此执行分支部署。
该变更尚未完成真实 Docker/服务器验证，不阻塞本地 P0。
性能实验仅推送 `feature/gpu-throughput-optimization` 和不可覆盖 tag，不合并 main。
按用户后续明确授权，仅部署入口、其测试和使用说明另以 `51333af` 合入 origin/main，
tag `deploy-20261023-01`；未包含 B 的后端配置、性能诊断或模型变更。
固定 A 的部署包装 tag `exp-20261023-a-baseline`（`940e368`）仅同步部署脚本和其测试；
运行源码、Compose、依赖及构建文件均已与 `581c175` 逐项比对一致，固定参考定义不变。
回退使用保留的固定基点/实验版本重新部署，不移动原 tag、不重置其他分支及未提交文件。

清晰正脸/可读车牌必须遮挡，中大目标不能整体退化；仅容忍 1–2 帧侧脸边缘/背脸/出入画残影。
自动差异只定位复核，不能充当准确率；对照自身的误遮/漏遮和不可判读目标分别保留，不强行通过。

本地工具验证及其限制见 [P0 本地验证](p0-local.md)。

## CUDA 采样关联修复

RTX 主人脸第 64/424/768 次原始 trace 均已保存，前两份正常，第三份因 CUPTI 运行时事件
沿用旧执行线程 tid，旧筛选器误报无关联内核。改用选定父范围及其同线程 CPU 算子的 External id
关联 CUDA 调用；未携带该 ID 的 ORT 调用保持原线程/时间限制，不合并并行模型或整个时间窗的内核。
回归用例先复现旧归因错误，修复后 54 项定向检查通过；三份真实 trace 离线重算分别为
2289/2289/2266 个关联内核，busy 13.760/14.852/14.319 ms。第三次失败的原任务不计整片或 FPS。
另保存采样所在源帧、全图/局部和输入区域上下文，避免不同视图成本混为一谈；将重新运行整片采样。
