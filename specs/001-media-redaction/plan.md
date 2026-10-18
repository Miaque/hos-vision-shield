# Implementation Plan: 媒体人脸与车牌脱敏

**Branch**: `feature/cpu-detection`（当前CPU优化；主干为`main`） | **Date**: 2026-10-18 | **Spec**: [spec.md](spec.md)

**Input**: `specs/001-media-redaction/spec.md`

## Summary

FastAPI 负责接入、状态和文件结果；Taskiq 文件与流两个 worker 池执行处理；Redis 同时承担队列、任务状态和运行意图持久化。
受控媒体子进程使用人脸/车牌检测模型和 PyAV 完成逐帧检测、实心遮挡及音视频输出；MediaMTX 提供 RTSP 播放。当前按用户效果优先要求，默认quality组合EgoBlur全图与YuNet/HyperLPR分块补充；上一轮cpu轻量方案保留对照。
首版内部使用，无调用方鉴权，无自动文件删除，不额外录像。流恢复由持久运行意图驱动，不依赖永久未确认消息。

## Technical Context

**Language/Version**: Python 3.12（符合现有 >=3.12，首版锁定验证解释器，不直接承诺后续所有 Python 版本）。

**Primary Dependencies**: FastAPI、Uvicorn、Pydantic Settings、Taskiq、taskiq-redis、redis.asyncio、HTTPX、PyAV、Pillow、NumPy、OpenCV DNN；PyTorch/torchvision用于quality中的EgoBlur及单独基线；外部 Redis、MediaMTX 与支持 H.264 的 FFmpeg 库。具体兼容版本锁定到 uv.lock；不使用 latest 部署标签。

**Storage**: Redis 单实例持久卷，AOF appendfsync=always、noeviction；任务元数据无 TTL。共享媒体目录保存原文件与结果，当前本机进程使用同一目录，扩到多机时再提供共享文件系统；模型为只读本地文件，不在请求中下载。

**Testing**: pytest、pytest-asyncio、HTTPX ASGI 测试；真实 Redis/MediaMTX 集成测试；已授权样本及 ffprobe 验证。

**Target Platform**: Windows本机原生进程（用户已明确要求），直接使用Python3.12、Redis和MediaMTX可执行程序；Docker/WSL2不再是实现或验收前提。

**Project Type**: Python REST 服务与后台 worker，无前端。

**Performance Goals**: 按 SC-006/007 出具可复现实测报告，不设并发或准确率保证；默认每个 worker 进程一个活动媒体任务，扩进程增加槽位。

**Constraints**: 仅 JPEG/PNG、MP4/H.264/AAC 或无音轨、RTSP/H.264/AAC；两个模型必须同时可用；任何失败不得旁路原始画面；同步媒体操作不得运行在 API/Taskiq 事件循环上。

**Scale/Scope**: 文件和流隔离容量；每池最多100个未结束业务任务，物理消息队列各100条，流帧队列2帧，音频最多16秒且最多4MiB，文件单帧顺序处理。机器人高分辨率检测在本机CPU更慢，因此从640基线的4秒缓存调整为16秒；仍超限重建，不承诺实时帧率或固定延迟。

## Constitution Check

| 原则 | 研究前 | 设计后证据 |
|---|---|---|
| 媒体脱敏与 RESTful 契约 | 通过 | contracts/rest-api.md 明确输入、状态、失败和停止 |
| 异步非阻塞 | 通过 | 异步网络/Redis、受控文件线程与媒体子进程；有界队列、取消、超时 |
| 模块职责 | 通过 | API → jobs → media；检测与遮挡无 Web 依赖 |
| 最小实现与必要注释 | 通过 | 一个服务包，无插件系统、用户系统、OCR或模型训练；所有权和时间戳处说明不变量 |
| 验证与生命周期 | 通过 | quickstart.md 覆盖正常、取消、重投、故障恢复、音画同步及真实模型 |

无原则豁免。通过的是设计检查，不代表实现已验证。技术验证必须先证明模型可加载、音频可保留、消息确认可用和发布接管可隔离；失败时调整计划，不以桩实现通过。

## Project Structure

### Documentation (this feature)

```text
specs/001-media-redaction/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/rest-api.md
└── checklists/requirements.md
```

### Source Code (repository root)

以下是计划结构，尚未创建源码：

```text
src/vision_shield/
├── main.py                  # FastAPI lifecycle
├── config.py
├── api/{routes,schemas}.py
├── jobs/{service,store,broker,tasks,reconcile}.py
├── media/{runner,images,video,streams,source,publisher}.py
├── detection/egoblur.py
└── redaction.py
tests/{unit,integration,e2e}/
```

**Structure Decision**: 保持现有 src 包布局；jobs.store 直接封装项目所需 Redis 原子操作，不建通用仓储框架；media.source 统一外部地址校验和下载；redaction 共享矩形裁剪与实心遮挡。

## Execution Design

1. 上传使用 UploadFile 分块读取并在线程池写入任务目录，限制请求体与实际字节数；URL 由文件 worker 用 HTTPX 流式下载。完整接收后才将文件标记为可处理。
2. 先原子保存任务和待投递标记；单一逻辑 reconciler（Redis 租约允许备份进程）投递 Taskiq 消息，消息只含任务 ID。投递后崩溃允许重复消息，但不能丢失持久意图。
3. 文件池和流池分别使用 RedisStreamBroker；任务取得 owner/generation 后手动 ACK，业务状态独立于 Taskiq result backend。已验证依赖组合为 Taskiq0.12.4/taskiq-redis1.2.2；broker保存AckableMessage回调供任务显式确认，该版本无Context.ack。文件 owner 丢失则标记 worker_lost 失败，不承诺断点续跑；流 owner 丢失则恢复调度。
4. worker await 受控媒体子进程；心跳、取消和控制仍在异步循环运行。子进程通过专用管道接收输入，不把源凭据放在命令行；父管道 EOF、停止或租约失效时退出，超时强制终止并 wait，禁止孤儿媒体进程继续发布。
5. worker 每 5 秒续租，租约 20 秒；reconciler 每 5 秒检查持久运行意图。流读超时 10 秒，重连退避 1/2/4/8/16/30 秒，用户停止终止退避。时长为实现初始值，故障测试需覆盖边界。
6. 文件输出先写 generation 独立临时结果，完成 flush/校验后原子改名，再通过 generation 条件提交成功；取消或旧执行者不得覆盖新状态。残留文件保留供排查，不进入下载接口。
7. 有界消息积压不能仅靠业务计数：发布适配层以 Redis 原子操作检查物理 stream 长度并入队，满时保留待投递标记；单消费者组，ACK 回调以 Lua 原子完成 XACK + XDEL，通过broker显式确认方法触发，避免崩溃遗留已确认消息。禁止用近似 MAXLEN 裁掉未确认消息。业务准入限制未结束任务，reconciler 同一任务只保留一个待投递意图。
8. 无权限管理不取消输入安全：HTTP/HTTPS 来源按域名/IP/CIDR allowlist 放行，每次重定向和实际连接重验地址；保留原 Host/TLS 校验并固定已验证地址，RTSP 默认 TCP 且结合本机出站规则限制目标，防止 DNS 重绑定及服务内部地址探测。

### 媒体处理与时间线

JPEG/PNG 使用 Pillow 解码、应用方向、裁剪检测框后填黑；不复制输入 EXIF/缩略图以免旁路保留原始敏感图像。
两类检测模型在子进程初始化时加载，不增加公共模型选择接口。原EgoBlur Gen1为首版CPU基线；2026-10-18按用户CPU优化要求，默认改为YuNet人脸与HyperLPR3的320车牌检测ONNX，通过OpenCV DNN显式选择CPU。仅使用检测网络，不运行OCR或身份识别；后端由本机配置选择，无自动模型回退。来源、许可、固定版本和哈希见models/cpu-manifest.json。

后续同日按效果优先补充，默认quality使用EgoBlur1280全图，再追加YuNet/HyperLPR补充候选。人脸种子阈值分别0.3/0.6，车牌0.1/0.25；EgoBlur得分至少0.7保留，其他候选需要另一模型同类框IoU至少0.2佐证，或在候选周边局部复核。局部取3倍最长边且至少96像素，调整到最长边320；佐证人脸门槛为EgoBlur0.1、YuNet0.3，不单独产生低分遮挡。保留被确认的重叠遮挡，不跨模型NMS丢弃边缘；任一模型异常整个处理失败。单独egoblur/cpu后端保留对照，四个模型路径、异步进程边界和REST接口不变。不扩展CUDA或下载新模型，效果/代价见validation/quality-optimization.md。

CPU实现使用全图与约640源像素重叠分块，每轴至多8个位置，大图扩大分块以保持区域覆盖，每次顺序处理一个视图。人脸阈值0.3、车牌0.25；人脸每侧扩框15%，车牌水平每侧30%/垂直15%，保留重叠遮挡。该策略增加误遮，不能补偿从未检出的目标；实际效果和性能另见validation/cpu-optimization.md。其余任务、音视频时间线、缓存和异步边界保持原实现。

机器人补充：默认检测最长边1280、Lanczos缩小、不上采样，每侧外扩15%，原图像素不作锐化/重建。按样本比较而非盲目降低0.1阈值。视频/流每会话一个FrameRedactor，仅缓存两组实际框与一张36×64 RGB缩略图，补偿200毫秒源时间内的短暂漏检；明显缩略图差异、尺寸或时间异常清空，永不以保留框刷新历史。快速运动不保证可补偿，当前CPU流跳帧较长时通常不会沿用旧框。

视频与流使用一个输入容器 demux 音视频，视频解码后检测和填黑，再编码 H.264；AAC 包 remux，不做音频解码或修改内容。
保留源 PTS/time_base 并统一音视频时间原点，按 muxer 时基换算，不以处理帧序号重造固定 FPS 时间线。
实时流的解码线程只向最多 2 帧的队列放最新帧，推理不阻塞源读取；编码/mux 由单一所有者串行执行。
音频缓存达到时长或字节上限时结束当前发布并重连，不能无限积压，也不把未经脱敏的视频作为同步补偿。
输入时间戳无效、时间线倒退或中途格式变化无法安全处理时明确失败/重建会话；文件不静默加速或丢帧。

### 发布所有权与故障隔离

稳定播放路径为 redacted/{task_id}，MediaMTX 禁止 publisher 覆盖已有发布者，关闭录像。
Redis 租约只保护状态，不能撤销旧 RTSP socket。接管前先设置 publish_allowed=false 并撤销旧 generation，
通过 MediaMTX Control API 踢掉已登记的旧发布 session，确认路径无旧 publisher 后才授予新 generation 发布资格。
MediaMTX 内部 HTTP publish 回调校验 task_id、generation、owner 和有效租约并登记 session ID；读流无需凭据。
这是 worker 所有权隔离，不新增调用方登录或权限系统。回调/控制接口只开放服务内部网络。

回调与接管必须串行化，阻止校验与连接建立之间的旧请求重新生效：接管期间拒绝全部发布，清除已批准及建立中的旧 session，确认其终止后再开放。
不能证明旧会话已结束时保留 recovering 或当前状态并设置 desired_state=stopped，不允许新发布、不返回 stopped。必须用“旧 worker 断开 Redis 但仍连媒体服务”的故障注入证明该协议。
单独锁过期、path 暂时为空或 session 查询返回404都不算通过；已批准但仍建立中的连接可能稍后出现。该验证失败时阻断后续流实现，重新评估媒体端代次隔离方案，不以轮询空路径替代证明。

## Implementation Order

1. 技术验证：固定权重与依赖，单图双类检测、H.264/AAC 时间线回环、Taskiq 手动 ACK、MediaMTX 发布接管竞态；形成通过证据后再展开业务。
2. 图片薄切片：输入校验、持久任务、文件队列、检测遮挡、结果与取消，完成契约和单图集成测试。
3. 有限视频：逐帧/音频保留/时间戳、失败清理与实测。
4. 流处理：隔离流 worker、丢帧、音频边界、恢复、所有权隔离及横向扩容测试。
5. 综合验证与人工效果报告；生成部署与操作说明，不增加删除、鉴权或识别能力。

## Complexity Tracking

无违反宪章的复杂性豁免。reconciler 用于 ACK 后恢复，发布隔离用于阻止重复输出，子进程用于隔离阻塞并可靠终止；三者均对应已确认需求。

## 本机运行修订

用户在2026-10-18明确要求原生本地运行，由代理直接下载官方模型。已移除Docker专用文件，基础服务由scripts/start-infrastructure.ps1管理，仅回收自己创建的进程。T002以本地持久化和健康自检验收；流worker扩容采用增加本机进程，不要求容器。
