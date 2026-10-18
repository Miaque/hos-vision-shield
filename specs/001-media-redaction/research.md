# Research: 媒体脱敏技术决策

日期：2026-10-18。设计研究已补充首次实施验证；已安装依赖，运行队列与媒体探针；模型权重仍缺失。各项证据与未验证范围见 [validation/gates.md](validation/gates.md)。

## 1. API 与异步边界

**Decision**: FastAPI/HTTPX/redis.asyncio 处理协议及外部 I/O，阻塞媒体工作放受控子进程；UploadFile 分块读取，落盘线程池执行。
**Rationale**: async 函数不会自动把直接调用的同步工具函数移到线程；媒体处理与控制循环需隔离。
**Alternatives considered**: 请求内推理、FastAPI BackgroundTasks 不满足已确认的 Taskiq、恢复及独立扩容需求。
**Sources**: [FastAPI async](https://fastapi.tiangolo.com/async/)、[UploadFile](https://fastapi.tiangolo.com/tutorial/request-files/)。

## 2. Taskiq、持久状态与恢复

**Decision**: RedisStreamBroker，文件/流独立队列与 worker；Redis 保存任务意图；owner 登记后手动 ACK，reconciler 恢复流。
**Rationale**: 长期未 ACK 消息会遇到 idle reclaim，不能以 broker PEL 判断任务是否仍运行。消息完成和业务结束须分开。Redis 采用 AOF always/noeviction，避免额外数据库；不承诺存储卷丢失后的恢复。
**Alternatives considered**: RabbitMQ/NATS 可用于更复杂部署，但当前再加独立状态存储增加维护成本；PubSub 不适合需要恢复的任务。永久不 ACK 的流消息不采用。
**Sources**: [Taskiq CLI/ACK](https://taskiq-python.github.io/guide/cli.html)、[taskiq-redis](https://github.com/taskiq-python/taskiq-redis)、[broker 源码](https://github.com/taskiq-python/taskiq-redis/blob/main/taskiq_redis/redis_broker.py)、[Redis persistence](https://redis.io/docs/latest/management/persistence/)。

实施发现Taskiq0.13.0与taskiq-redis1.2.2因移除compat.model_dump而导入不兼容。
已固定Taskiq0.12.4/taskiq-redis1.2.2，使用broker保存AckableMessage回调供任务显式确认；该版本没有Context.ack，禁止调用不存在的API或修改第三方依赖来绕过。
worker max-async-tasks=1、max-prefetch=0、xread_count=1已通过真实worker探针验证，见[队列证据](validation/queue-probe.md)。
消息有界准入与原子 XACK + XDEL 需要小范围 broker 适配，不能误把队列 MAXLEN 自动裁剪当作安全背压。

## 3. 检测模型

**Decision**: EgoBlur Gen1 人脸和车牌专用权重作为首版基线，仅复用检测结果，不调用模糊 CLI；实心遮挡由本项目完成。
**Rationale**: 同一官方项目提供两类现成模型和 Apache-2.0 许可，避免引入 OCR 或训练任务。首版效果验证用途允许先评估跨场景表现。
**Alternatives considered**: 通用 COCO YOLO 不直接包含 face/license-plate 类；组合其他脸模型与来源不明的车牌权重会增加来源及适配成本。
**Sources**: [EgoBlur 官方仓库](https://github.com/facebookresearch/EgoBlur)、[官方模型获取入口](https://www.projectaria.com/tools/egoblur/)、[COCO 类别](https://docs.ultralytics.com/datasets/detect/coco/)。
模型面向 Aria 数据，监控场景及中国车牌效果没有本项目实测依据；不宣称性能或准确率适用。
当前默认模型页未提供可直接下载权重；实施时找到Project Aria官方Hugging Face历史revision `9c0b319ad09d0346e8294d7bf569593d3ab865c0` 的原始Gen1压缩包，按官方LFS SHA-256校验。必须取得两个模型、记录版本/来源/SHA-256/许可并在 Python 3.12 加载成功。失败则技术验证不通过，不能用虚假检测替代，也不默默换模型。

## 4. 视频、音频与 RTSP

**Decision**: PyAV 解复用、解码、重编码视频和 remux AAC；MediaMTX 提供 RTSP 发布/播放，禁用录像。
**Rationale**: 保留 packet/frame 时间戳，有机会正确处理变帧率与音画同步；单源读取避免两路输入起点不一致。
**Alternatives considered**: OpenCV VideoCapture/Writer 不满足音频及原始时间线契约；RGB rawvideo 管道按固定 FPS 重造时间戳不适合丢帧流。
**Sources**: [PyAV basics](https://pyav.basswood.io/docs/stable/cookbook/basics.html)、[时间戳](https://pyav.basswood.io/docs/stable/api/time.html)、[I/O timeout](https://pyav.basswood.io/docs/stable/api/_globals.html)、[MediaMTX RTSP](https://github.com/bluenviron/mediamtx/blob/main/docs/3-publish/07-rtsp-clients.md)。
首个验证切片需证明 H.264 编码器可用、AAC remux 可行、PTS/DTS 合法以及丢帧后不加速。编码器 flush、音频缓存和丢帧为工程设计，不是文档自动保证。

## 5. 恢复时的发布隔离

**Decision**: generation 租约 + MediaMTX 发布回调 + Control API 撤销旧 session，播放端匿名。
**Rationale**: 旧 worker 即使失去 Redis 租约仍可能保有 RTSP socket；必须在媒体输出端隔离，且等待旧连接终止后才开放接管。
**Alternatives considered**: 只用 Redis 锁、只查每帧租约、允许新 publisher 覆盖旧 publisher 都无法排除旧进程抢回输出。
**Sources**: [MediaMTX HTTP 验证载荷](https://mediamtx.org/docs/features/authentication)、[Control API](https://mediamtx.org/docs/features/control-api)、[配置参考](https://mediamtx.org/docs/references/configuration-file)。
上述组合协议是本项目设计推导，非 MediaMTX 的分布式租约保证。特别要验证连接建立中的接管竞态；如果现有回调/控制能力不足以证明隔离，技术验证必须失败并修订设计，不放宽 FR-010。

## 6. 默认限制与依赖边界

**Decision**: Windows本机原生进程/CPU基线，共享本地媒体目录（用户修订）；不建设对象存储、数据库集群、GPU调度或模型训练平台。
**Rationale**: 首版以效果与实测为目标，单实例持久 Redis 和独立 worker 已覆盖当前恢复与扩容范围。
**Alternatives considered**: SQL 状态库、对象存储、GPU共享推理服务可后续按实测瓶颈引入，当前没有对应需求。

默认图片 20 MiB、40 MP，视频 2 GiB、60 分钟、3840×2160，HTTP 连接 5 秒/读空闲 30 秒/总下载 30 分钟，RTSP TCP 连接及读 10 秒；上传接收也有总时限和流量计数。参数可由部署配置调整，并在验证报告记录。阈值是本项目保守起点，不是用户业务上限承诺。
无待用户决策项；模型/版本/媒体接管验证为实现前置门槛，不是已经通过的测试。

## 用户运行方式修订

2026-10-18用户要求直接本机运行并由代理下载官方权重。移除Docker/WSL2依赖，不再把容器配置作为技术门槛；本机Redis/MediaMTX健康及持久化参数已由start-infrastructure.ps1 -Check实测。采用官方组织固定模型版本，下载不使用第三方镜像。
