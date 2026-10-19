---
description: "媒体人脸与车牌脱敏的可执行任务清单"
---

# Tasks: 图片、视频与视频流人脸车牌脱敏

**Input**: `specs/001-media-redaction/` 下的设计文档。
**Prerequisites**: [plan.md](plan.md)、[spec.md](spec.md)、[research.md](research.md)、[data-model.md](data-model.md)、[contracts/rest-api.md](contracts/rest-api.md)、[quickstart.md](quickstart.md)。
**Tests**: 依据宪章第五原则及规格验收场景生成针对性测试。先写失败用例，再实现；真实模型、真实队列和媒体验证不能用 mock 替代。测试环境缺失必须报告阻塞，不能将跳过记作通过。
**Organization**: 先完成公共基础，再按 US1 图片、US2 视频、US3 视频流交付。已完成项按实际验证标记，阻塞及局部结果见 [validation/gates.md](validation/gates.md)。

## Format: `[ID] [P?] [Story] Description`

- `[P]` 表示在满足注明的共同前置条件后，可与同批任务并行；不自动授权启动额外代理。
- `[US1]`、`[US2]`、`[US3]` 对应规格中的用户故事。
- 路径均相对仓库根目录；不为未来任务提前建立空模块，不顺带修改未涉及代码。

## Path Conventions

应用位于 `src/vision_shield/`；测试位于 `tests/`；部署文件位于根目录及 `deploy/`；验证证据位于本功能的 `validation/`。
模型权重、原始媒体和含凭据配置不提交 Git；只提交来源、哈希、样本清单和不含敏感数据的报告。

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 建立可运行的验证环境，不提前实现业务功能。

- [X] T001 在 pyproject.toml、uv.lock 和 .python-version 固定 Python 3.12 及计划依赖的兼容版本，加入 python-multipart 和指定测试依赖；以安装和导入自检验证，不改已有项目入口以外的无关配置。
- [X] T002 按用户修订，在 scripts/start-infrastructure.ps1、deploy/redis.conf、deploy/mediamtx.yml 建立原生Windows本地运行环境，Redis启用AOF always/noeviction并保留本地数据，MediaMTX启用RTSP/控制接口并禁用录像；-Check实际验证启动、健康与资源回收，不依赖Docker/WSL2。
- [X] T003 在 tests/unit/test_config.py 先覆盖无效容量和超时配置，再在 src/vision_shield/config.py 与 .env.example 定义计划中的来源白名单、模型/媒体路径、资源上限及时间配置；环境配置加载失败要明确报告，不添加用户鉴权选项。
- [X] T004 在 tests/conftest.py 和 tests/fixtures/manifest.json 建立可复现测试夹具入口，记录授权来源/哈希和人工标注范围，准备 JPEG/PNG、目标/无目标、AAC/无音轨、VFR/B帧视频及受控 HTTP/RTSP 测试源；原始样本放受忽略目录，更新 .gitignore 仅排除这些运行产物。

**Checkpoint**: Python、Redis、MediaMTX 与测试源能启动；后续任务需要的服务地址和样本来源可追溯。

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 先证实关键技术可行，再实现公共任务与执行基础。

### 技术验证门槛（T001–T004 后）

- [X] T005 [P] 在 tests/integration/test_model_probe.py 验证官方 EgoBlur Gen1 两个权重的 Python 3.12 加载及单图输出，记录来源、许可、版本和 SHA-256 到 models/manifest.json，并将真实运行证据写入 specs/001-media-redaction/validation/model-probe.md；缺任一权重不得以假检测通过。
- [X] T006 [P] 在 tests/integration/test_taskiq_probe.py 验证锁定版本的手动 ACK、并发/预取限制及重复投递，验证 Lua 原子 XACK+XDEL 和限长入队不会遗留已确认 entry 或裁掉未确认消息，证据写入 specs/001-media-redaction/validation/queue-probe.md。
- [X] T007 [P] 在 tests/integration/test_media_probe.py 验证 PyAV H.264 编码、AAC remux、无音轨、PTS/DTS换算和丢帧时间线，使用 ffprobe 比较输入输出并记录到 specs/001-media-redaction/validation/media-probe.md；不得只以“能播放”判定通过。
- [X] T008 [P] 在 tests/integration/test_publisher_probe.py 及其专用 tests/fixtures/publisher_probe.py 验证 MediaMTX 回调/session撤销，覆盖已批准尚未建立的发布连接、旧worker失去Redis但仍连媒体服务、旧代次重连及停止竞争；记录具体API调用与证据到 specs/001-media-redaction/validation/publisher-probe.md。
- [X] T009 汇总 T005–T008 到 specs/001-media-redaction/validation/gates.md，明确每项通过/失败及证据；仅全部通过后继续公共业务实现。若发布接管不能证明可靠，修订 specs/001-media-redaction/plan.md、research.md 和受影响契约，再重跑对应验证，不降低 FR-010 或把404/暂时空路径当作会话已消失。

### 公共基础（T009 通过后）

- [X] T010 在 tests/integration/test_task_store.py 先覆盖原子准入、任务与待投递标记一致性、并发claim、旧generation提交、重复终态释放及取消竞争，明确每池100个未结束任务的计数语义。
- [X] T011 在 src/vision_shield/jobs/store.py 实现任务记录、索引、Redis时间租约、generation CAS及状态转换，运行 T010；历史记录无TTL，终态只释放一次容量，旧owner不得覆盖结果或删除新租约。
- [X] T012 在 src/vision_shield/jobs/broker.py 落实 T006 验证的双 RedisStreamBroker、任务ID消息、手动ACK及原子确认删除/物理限长入队；复用探针测试验证每队列最多100条且不丢未确认消息，不实现通用broker框架。
- [X] T013 在 tests/integration/test_reconcile.py 先覆盖投递前后崩溃与重复消息，再在 src/vision_shield/jobs/reconcile.py 实现有租约的单一逻辑协调器、待投递扫描和owner失效处理；文件标记worker_lost，流按持久意图重排，终态或停止意图不重启。
- [X] T014 在 tests/integration/test_runner_lifecycle.py 先覆盖取消、父进程退出、管道EOF、子进程卡住及租约失效，验证终止后不再读写且事件循环仍可响应。
- [X] T015 在 src/vision_shield/media/runner.py 实现受控媒体子进程、管道传参、异步等待、停止信号、限时kill/wait及父进程退出监测，运行 T014；源凭据不得出现在命令行，活动资源释放与文件保留分别处理。
- [X] T016 在 src/vision_shield/api/schemas.py 和 src/vision_shield/main.py 建立契约类型、统一错误响应、日志脱敏、异步生命周期与健康检查，在 tests/unit/test_api_base.py 验证无用户鉴权、错误不泄露凭据及依赖不可用时503；不让API加载模型或执行同步媒体处理。

**Checkpoint**: 四项真实技术验证通过，公共任务状态、队列、执行生命周期有失败路径证据后才能开始用户故事。

---

## Phase 3: User Story 1 - 图片脱敏 (Priority: P1) — MVP

**Goal**: 上传或远程URL提交图片，异步查询、取消并下载人脸和车牌均被实心遮挡的结果。
**Independent Test**: 两种输入方式分别提交 JPEG/PNG，获取真实检测结果；损坏、无目标、取消和失败可区分；原始文件保留，未完成产物不可下载。

### Tests for User Story 1

- [X] T017 [P] [US1] 在 tests/integration/test_image_api.py 编写失败用例覆盖上传/URL创建、202/Location、任务查询、结果下载、取消幂等、未知ID、容量和完整错误结构；使用 contracts/rest-api.md 作为验收依据。
- [X] T018 [P] [US1] 在 tests/unit/test_redaction.py 和 tests/unit/test_source.py 编写失败用例覆盖双类矩形、边缘/重叠、方向处理及实心像素覆盖，URL白名单、重定向、DNS重绑定、路径穿越、大小/超时限制；目标框单元测试与真实检测效果分开。

### Implementation for User Story 1

- [X] T019 [P] [US1] 在 src/vision_shield/detection/egoblur.py 复用 T005 已验证的双模型加载和前后处理，输出统一像素坐标框；两个模型缺失任一则失败，转换规则和模型限制添加必要注释，不引入识别或模型插件系统。
- [X] T020 [P] [US1] 在 src/vision_shield/redaction.py 实现共享矩形裁剪和实心黑色遮挡，运行 T018 的遮挡用例；不实现模糊、马赛克或目标类别选择。
- [X] T021 [P] [US1] 在 src/vision_shield/media/source.py 实现有界上传落盘及HTTPX流式下载、白名单和实际连接地址校验、重定向限制、Host/TLS保留、文件名隔离和限时/限量校验，运行 T018 的来源用例；阻塞文件I/O移出事件循环。
- [X] T022 [US1] 在 src/vision_shield/media/images.py 组合 T019–T021，实现JPEG/PNG真实格式校验、方向归一、解压像素上限、双类检测及遮挡，输出同格式但不复制EXIF/缩略图；先写代次独立结果，失败不返回原始图片。
- [X] T023 [US1] 在 src/vision_shield/jobs/service.py 和 src/vision_shield/jobs/tasks.py 接入图片任务，完成准入/持久意图、owner登记后ACK、心跳、子进程执行、结果原子提交和取消；运行T010/T013/T014相关回归，验证旧执行者不能发布结果。
- [X] T024 [US1] 在 src/vision_shield/api/routes.py 接通文件创建、上传、查询、取消和结果下载端点并在 src/vision_shield/main.py 注册，运行 T017；API只暴露业务ID和相对结果URL，先完整接收上传再返回202，不加入列表或删除接口。
- [X] T025 [US1] 在 tests/e2e/test_images.py 用真实模型与队列执行两种输入方式及无目标/损坏/取消路径，记录结果、原文件保留及负载期间查询响应到 specs/001-media-redaction/validation/images.md，明确未检测到不等于零漏检。

**Checkpoint**: US1 可独立演示，是建议的最小可交付版本；尚未实现的视频/流不得对外宣称可用。

---

## Phase 4: User Story 2 - 完整视频文件 (Priority: P2)

**Goal**: 完整处理MP4视频、保留AAC音频，不因负载主动丢帧，并支持查询、下载及取消。
**Independent Test**: 用带AAC及无音轨MP4验证帧数、顺序、时长和取消；视频结果不依赖RTSP流任务。

### Tests for User Story 2

- [X] T026 [US2] 在 tests/integration/test_video_pipeline.py 先覆盖 H.264/AAC、无音轨、VFR/B帧、旋转、编码flush、无效时间戳、损坏输入及取消，采用 T007 的ffprobe检查方法验证逐帧处理和原AAC包保留。

### Implementation for User Story 2

- [X] T027 [US2] 在 src/vision_shield/media/video.py 实现单源音视频demux、逐帧双类检测/遮挡、H.264编码和AAC remux、时基换算与flush；复用共享检测/遮挡，运行T026，不用固定FPS重造时间线或把整个视频读入内存。
- [X] T028 [US2] 在 src/vision_shield/jobs/tasks.py、src/vision_shield/media/runner.py 和 src/vision_shield/api/routes.py 接入video分派与限制探测，复用文件提交/下载/取消契约；超时、磁盘不足和失败不得暴露半成品，原文件与生成文件保留。
- [X] T029 [US2] 在 tests/e2e/test_videos.py 执行上传和URL视频全链路、过载不丢帧、取消与worker_lost场景，证据写入 specs/001-media-redaction/validation/videos.md，并运行图片回归验证共享代码未退化。

**Checkpoint**: 图片与有限视频均可独立运行；音频和时间线已验证，不需要流分发服务才能完成文件任务。

---

## Phase 5: User Story 3 - 持续视频流 (Priority: P2)

**Goal**: worker持续处理RTSP，返回可播放的脱敏流，支持有界丢帧、故障恢复、停止和扩容。
**Independent Test**: 由其他播放器读取输出，模拟断流、worker退出、旧owner继续连媒体服务及停止竞争，核对无重复有效输出。

### Tests for User Story 3

- [X] T030 [P] [US3] 在 tests/integration/test_stream_api.py 先覆盖流创建、ready与地址区别、查询状态、停止幂等和非终态停止意图；测试不得把接收停止请求当作已释放资源。
- [X] T031 [P] [US3] 在 tests/integration/test_stream_pipeline.py 先覆盖2帧缓冲、丢旧帧、AAC缓存上限、PTS保留、退避取消、发布失败及不支持格式，确保未经处理帧输出数量为0。

### Implementation for User Story 3

- [X] T032 [US3] 在 src/vision_shield/media/publisher.py 和 src/vision_shield/api/internal.py 落实 T008 已证实的MediaMTX所有权回调/撤销协议，注册内部回调，保证接管串行、旧代次重连拒绝、会话确认撤销后开放新发布；媒体控制不可用时不得虚报完成，不新增用户鉴权。
- [X] T033 [US3] 在 src/vision_shield/media/streams.py 实现单源读取、独立有界最新帧队列、串行编码/mux、AAC时长/字节双上限、RTSP输出和有界退避；复用 T027 时间戳规则与公共检测逻辑，超限重建会话，不录制原流。
- [X] T034 [US3] 在 src/vision_shield/jobs/tasks.py、src/vision_shield/jobs/reconcile.py 和 src/vision_shield/jobs/store.py 完成流池注册、持久运行意图、续租及worker崩溃恢复，停止优先于重排，终态只释放一次容量；复用T013故障测试并增加流状态断言。
- [X] T035 [US3] 在 src/vision_shield/api/routes.py 接通流创建、状态及停止，稳定播放路径只用于匿名读取，运行T030/T031；失败流和已停止流不自动重启，ready仅在脱敏输出确认后变为true。
- [X] T036 [US3] 在 scripts/start-application.ps1 和 deploy/mediamtx.yml 接通本机api/reconciler/file-worker/stream-worker进程、内部发布检查、禁止覆盖publisher及关闭录像，配置单进程单任务和共享本地目录，验证增加stream-worker进程不会占用文件池容量。
- [X] T037 [US3] 在 tests/e2e/test_stream_recovery.py 将T008探针转为实际服务故障回归，断流、worker重启、停止竞争各3次，并覆盖旧连接存活/连接建立窗口/队列重投；记录播放、session、音画同步及扩容证据到 specs/001-media-redaction/validation/streams.md。

**Checkpoint**: 三个故事均可独立验收，停止不会被恢复反转，旧发布者无法继续有效输出。

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: 完成跨场景验证和交付文档，不增加新功能。

- [X] T038 在 tests/integration/test_resource_limits.py 补齐队列/磁盘满、Redis不可用、媒体控制不可用、父进程退出、来源限制及日志敏感数据检查，运行三类故事相关回归；报告活动资源上界与退出行为，不删除用户要求保留的文件。
- [X] T039 [P] 在 tests/e2e/test_measurements.py 实现与SC-006匹配的可重复测量入口并生成 specs/001-media-redaction/validation/performance.md，记录固定环境/样本/参数下的处理速度、流延迟/丢帧、CPU/内存及查询/停止时延；不虚设生产吞吐承诺。
- [ ] T040 [P] 按 tests/fixtures/manifest.json 的固定画面范围人工复核双类目标、共存、无目标、遮挡/小目标，写入 specs/001-media-redaction/validation/detection-quality.md，分别记录目标总数、漏检、误检和对照产物；没有人工结果不得标记此任务完成。
- [X] T041 更新 README.md、specs/001-media-redaction/quickstart.md 和 specs/001-media-redaction/validation/acceptance.md，实际执行启动与验证指南，汇总FR-001–020及SC-001–007证据，复核模块依赖/必要注释/无鉴权与文件保留边界；明确真实样本、故障测试与未验证范围，不以测试桩或跳过冒充完成。

---

## Dependencies & Execution Order

### Phase Dependencies

```text
T001 → T002 → T003 → T004
                     ↓
              T005 / T006 / T007 / T008（可并行）
                     ↓
                   T009
                     ↓
                T010 … T016
                     ↓
              US1 T017 … T025
                     ↓
              US2 T026 … T029
                     ↓
              US3 T030 … T037
                     ↓
              T038 → T039 / T040 → T041
```

默认顺序执行；图中的并行批次只能在共同前置条件完成后开始。T009为实际技术门槛，不是文档签字任务。
T008未通过不得以任务分解完整为理由继续声称流实现可行。

### User Story Dependencies

- US1依赖公共基础，交付完整图片MVP。
- US2复用US1的输入、任务生命周期、检测和遮挡；验收可单独运行，不依赖流任务。
- US3复用公共基础及US2的时间线处理经验，默认在US2之后实施；验收不要求同时运行文件任务。
- 共享 jobs/tasks.py、api/routes.py、runner.py 的改动串行合并，不把三个故事标成可无条件并行。

### Within Each User Story

先写并确认相关失败测试，再实现；每项实现任务都需运行对应测试。技术探针应被复用或转成正式回归，不维护两套重复实现。
未完成真实技术验证或人工复核时保留未勾选状态，并记录具体阻塞；不得自动跳过。

### Parallel Opportunities

- T005–T008：共同环境就绪后，四份独立技术探针与报告。
- US1：T017/T018可并行；之后T019/T020/T021可并行，T022等待三者完成。
- US2：主要复用共享代码，T026→T027→T028→T029串行；T026中的纯视频与AAC样本验证可由同一测试任务参数化执行，不并行改同一文件。
- US3：T030/T031测试定义可并行；发布、媒体循环与调度集成按顺序落地。
- T039/T040：综合回归通过后，性能报告与人工效果复核使用各自固定样本副本，并避免测量过程中争抢同一运行资源。

## Parallel Example: User Story 1

T017检查图片HTTP契约，T018检查遮挡与来源边界；两者输出不同测试文件。完成后T019检测器、T020遮挡和T021输入处理独立实现，T022统一集成。

## Parallel Example: User Story 2

不安排并行源码改动：T027与T028依赖同一媒体执行链。T026用独立样本参数覆盖有音轨与无音轨，之后T029统一端到端验收。

## Parallel Example: User Story 3

T030仅编写流接口测试，T031仅编写媒体背压/时间线测试；两者完成后由T032–T035依次实现并验证。

## Implementation Strategy

### MVP First (User Story 1 Only)

先完成T001–T016，随后完成US1并运行T025；此时可以演示图片上传/URL→任务→双类实心遮挡→下载。
不能因为图片可用而把视频或流的未完成任务勾选。

### Incremental Delivery

图片MVP→完整文件视频→持续流→综合实测与人工复核。每次只修改本任务及直接调用方，保留既有未跟踪文件和无关修改。
不在本阶段实现删除、用户鉴权、OCR、模型训练、管理页面或GPU调度。

### Coverage Map

| 规格要求 | 主要任务 |
|---|---|
| FR-001–005 | T017–T025、T028 |
| FR-006 | T026–T029 |
| FR-007–012 | T008、T030–T037 |
| FR-013–014 | T021–T025、T028–T029、T033、T038 |
| FR-015–016 | T003、T016–T018、T021、T032、T038 |
| FR-017–019 | T006、T010–T016、T023、T034、T038–T039 |
| FR-020 | T005、T007、T022、T026–T029、T031–T033 |
| SC-001–003 | T018、T025–T029 |
| SC-004–005 | T006、T008、T031、T037–T038 |
| SC-006–007 | T039–T041 |

## Notes

- 原首版41项；另有下方机器人补充任务。原Setup 4项、Foundational 12项、US1 9项、US2 4项、US3 8项、综合交付4项。
- T001–T039和T041已有首版验证；T040人工质量复核未完成。历史阶段：640完整回归69项、1280与超宽流82项、单独ONNX CPU107项通过；最终效果优先组合完整回归115项通过，范围与性能见validation/quality-optimization.md，各阶段结果不直接相加。
- 前置和后置hooks：未配置 .specify/extensions.yml，均跳过。

## 巡逻机器人模糊素材补充

- [X] T042 读取用户demo素材并记录哈希、格式、尺寸；比较640/1280真实双模型检测，保留原始文件，明确误检与速度代价。
- [X] T043 为检测分辨率、遮挡边缘和短时漏检补偿先建立失败测试，落实到共享检测/帧处理及配置；图片不混入视频时序状态，流重连不沿用状态。
- [X] T044 在固定图片与视频抽样上保存调整前后结果，运行图片/视频/RTSP回归，并记录CPU代价和未完成人工验收的范围。
- [X] T045 实接用户提供的真实RTSP，验证API→worker→MediaMTX→独立客户端播放，复现并修复起始无PTS帧重连，保存截图、吞吐及漏遮证据并停止测试任务；效果不合格的事实见validation/real-rtsp.md，不勾选T040人工质量验收。

## 开发环境CPU优化

- [X] T046 比较轻量CPU检测模型，记录官方来源、固定哈希与许可；以检测专用ONNX实现共享后端，保留EgoBlur对照，不引入OCR或GPU要求。
- [X] T047 为分块覆盖、坐标还原、边缘外扩及异常输出建立回归；固定21份机器人素材记录31个画面与原模型的耗时/遮挡对照，运行完整测试（107项通过）。
- [X] T048 在当前.env下复测用户真实RTSP的独立播放、音轨、吞吐和停止，保存少量截图与验证报告，清理本次应用进程；质量人工验收仍单独保留。

## 效果优先补充

- [X] T049 针对上一轮确认漏遮与大块误遮建立先失败的真实样本回归，接入EgoBlur全图与高阈值YuNet/HyperLPR补充组合，不实现CUDA、不改变REST接口。
- [X] T050 在同一31个机器人画面与固定绿牌截图复核改善及新增问题，运行组合后端完整回归与媒体播放/停止验证，记录CPU代价和未完成质量验收范围（115项通过、真实RTSP五帧与AAC解码、停止和进程清理完成）。
- [X] T051 复现轮胎、背部头颈及夜间背景误遮，为不确定候选增加同类框交叉佐证与局部复核；保留强主模型结果，回查绿牌/侧脸等真实目标并输出调整前后的同帧对比（九张真实样本、76项单元及1项CPU模型测试通过，2秒/6fps预览完成；未重测真实RTSP，质量验收仍待完成）。
- [X] T052 修正夜间大轮胎被两路中等分数人脸候选相互确认的问题，按候选尺寸收紧佐证、保留小侧脸灵敏度；最终16张图片与26帧完整片段实测，轮胎指定区域误遮12帧降为0帧、真实车牌保留，91项机制测试及6项真实模型回归通过，可用13张固定素材区域通过；历史绿牌截图缺失，未重测其余四段视频及真实RTSP，T040仍未完成。
