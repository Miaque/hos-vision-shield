# Tasks: YOLO 人脸与车牌 CPU 效果对比

**Input**: 本目录 spec.md、plan.md、research.md、data-model.md、contracts/evaluation.md、quickstart.md。

**Worktree**: `C:/Users/zhanjh/.codex/worktrees/yolo-comparison/vision-shield`，分支 `feature/yolo-comparison`。以下路径均相对此 worktree；不得在主工作区实现。

**Tests**: spec FR-015 与宪章 V 要求针对性测试。新逻辑先建立失败测试，再实现；真实模型、媒体运行与人工复核分别记录。

**Format**: `- [ ] T编号 [P可选] [US编号] 描述与文件路径`。所有项目前均未执行；[P] 仅表示前置完成后可与指定不同文件任务并行，不授权并行跑 CPU 基准。

## Phase 1: Setup

**Goal**: 保留工作区隔离，准备可复现环境，不重新初始化项目。

- [x] T001 核对分支、现有改动及 main 基线 `0664be4ae9edb6d0eac7bb2635d5563dad9d7956`，检查 Python 3.12 和 uv.lock，按锁定依赖准备本 worktree 环境；将实际解释器、依赖版本及源码位置写入 data/validation/yolo-setup/environment.json，不修改主工作区。
- [x] T002 根据 research.md 的两个固定来源与 SHA-256 创建 models/yolo-manifest.json，记录角色、版本及许可来源，在 models/NOTICE.md 补来源说明；确认 .gitignore 已覆盖 ONNX、.part 和 data/，仅在确有遗漏时补规则。

## Phase 2: Foundational — 模型真实兼容性门槛

**Goal**: 先确认两份真实权重可运行，避免围绕猜测输出格式实现。

- [ ] T003 在 scripts/download-yolo-models.ps1 实现固定 manifest 下载、已有文件哈希复用、超时和 .part 校验后原子替换；提供可在临时目录执行的小型自检，验证哈希错误或下载失败不会覆盖已有有效文件。
- [ ] T004 执行 scripts/download-yolo-models.ps1，并在 tests/integration/test_yolo_model_probe.py 建立真实 ONNX CPU forward 冒烟；检查实际输入尺寸、输出布局、类别、内置 NMS、颜色/归一化约定，将证据补入 models/yolo-manifest.json 与 specs/003-yolo-cpu-comparison/research.md。不得把上游示例当作实际文件契约。
- [ ] T005 在本 worktree models/ 准备基线四份权重，按 models/manifest.json 与 models/cpu-manifest.json 校验并写入 data/validation/yolo-setup/baseline-models.json；若 T004 不兼容，先记录具体错误并更新 plan.md/research.md 的最小替代方案，再重新通过真实冒烟，不自动静默换权重。

**Checkpoint**: 两个 YOLO 权重真实 CPU forward 通过，基线模型完整。未满足前不得开始正式效果对比。

## Phase 3: US1 — 相同脱敏能力 (P1，首个可运行增量)

**Independent Test**: 使用 YOLO 处理含两类目标及无目标输入，输出原图坐标实心遮挡；配置与 worker 能启动，错误/取消不会把未处理媒体当成功。

### Tests first

- [ ] T006 [P] [US1] 在 tests/unit/test_yolo_detection.py 编写实际模型契约对应的失败测试：RGB/uint8 输入、非方图 letterbox 逆变换、分块偏移、边缘裁剪、阈值、NMS、扩框一次、空目标、非有限/非法输出以及任一模型异常不返回部分成功。
- [ ] T007 [P] [US1] 在 tests/unit/test_detector_selection.py、tests/unit/test_inference_device.py、tests/unit/test_worker_models.py 增加 yolo 显式选择、cuda 拒绝、两份指定 ONNX 缺失时提前失败及旧 quality 默认不变的失败测试。

### Implementation

- [x] T008 [US1] 在 src/vision_shield/detection/yolo.py 实现最小双模型 Detector：串行 OpenCV CPU 推理，复用 cpu.Detector._views，按 T004 契约预处理/解码，输出源坐标扩框；固定首轮阈值和可用 NMS，保持跨视图遮挡并集，输入与模型输出验证不得省略；使 T006 通过。
- [x] T009 [US1] 在 src/vision_shield/config.py、src/vision_shield/detection/__init__.py、src/vision_shield/jobs/worker.py 接入 yolo，复用 face_model_path/plate_model_path，保持 quality 默认与其他后端行为；使 T007 通过，不增加 API 字段或队列表。
- [ ] T010 [US1] 在 tests/integration/test_yolo_model_probe.py 补实际 Detector 冒烟，在 tests/integration/test_video_pipeline.py、tests/integration/test_stream_pipeline.py、tests/integration/test_runner_lifecycle.py 补最少必要覆盖并执行：图像/视频/流复用、200ms 保持、无效媒体、异常与取消释放；记录真实模型与替身测试边界于 specs/003-yolo-cpu-comparison/validation.md。
- [ ] T011 [US1] 更新 .env.example、models/README.md、README.md 的 yolo CPU 启动示例及失败说明，配置使用经验证模型路径；执行相关单元及媒体回归，将命令、结果和跳过项写入 specs/003-yolo-cpu-comparison/validation.md。

**Checkpoint**: US1 可运行且旧后端回归无新增失败。该增量只证明接入完成，不证明 YOLO 效果更好。

## Phase 4: US2 — 同批媒体公平对比 (P1)

**Independent Test**: prepare 固定清单；同机两组完整处理 21 份素材并产生 42 份可读输出，样本与时刻可对应；任何失败明确列出。

### Tests first

- [ ] T012 [P] [US2] 在 tests/unit/test_yolo_comparison.py 编写 prepare/run 失败测试：中文路径、输出不得覆盖输入、16图5视频清单校验、变帧率每5秒加末帧去重、帧数元数据为0、哈希变更、错误源码导入、断点只重跑未完成整份素材及失败返回非零。
- [ ] T013 [P] [US2] 在 tests/integration/test_yolo_comparison_media.py 用小型合成媒体建立完整性失败测试：短截输出、PTS/尺寸/音轨异常、原始黑像素不算遮挡证据，以及记录遮挡历史不改变生产处理像素。

### Implementation and actual evaluation

- [x] T014 [US2] 在 scripts/compare-yolo.py 实现 prepare 子命令：冻结源码/配置与输入哈希，实际解码计数和记录时间戳，保存每5秒首个达到时刻的帧及末帧、全部图片，写 run.json/manifest.json/annotations.json（初始 pending）；用增量读取限制内存。
- [ ] T015 [US2] 在 scripts/compare-yolo.py 实现 run：从固定提交导出基线源码快照，以明确导入路径的独立子进程选择版本；显式 CPU 配置，隔离 .env 及环境覆盖，核对实际模块文件、依赖和源码指纹（包括未跟踪候选源码），两组顺序运行，不并行抢 CPU。
- [ ] T016 [US2] 在 scripts/compare-yolo.py 复用 process_image/process_video 完整处理，模型在同一版本批次内驻留；原子写状态与产物，初始化和单次完整处理分别计时；记录完整性及实际遮挡并集（含历史保持），采样取自完整视频流程，取消/失败不能标 complete；通过 T012、T013。
- [ ] T017 [US2] 在 scripts/compare-yolo.py 实现 report 的本地三栏视觉结果：原始、基线、候选图片及完整视频链接/播放器和同帧对照，明确失败/待复核状态；转义文件名和文本，不加载外部脚本或上传媒体。
- [ ] T018 [US2] 执行 prepare，将 C:/Users/zhanjh/Downloads/demo 的固定清单写入 data/validation/yolo-cpu-001/manifest.json；在查看检测结果前逐样本准备原图目标与争议区域标注至 annotations.json，代理建议一律 pending，记录采样与运行参数冻结点。
- [ ] T019 [US2] 顺序执行 baseline、candidate 全量 CPU 运行，生成 data/validation/yolo-cpu-001/baseline/ 和 candidate/ 的全部媒体及完整性记录；不把历史片段或前三帧当全长结果，中断则继续未完成素材，42份完整产物未齐前保持 incomplete。
- [ ] T020 [US2] 生成并用浏览器检查 data/validation/yolo-cpu-001/index.html：全部素材可打开、视频可播放、采样时刻对齐、中文文件名显示正确、失败项可见；记录实际检查结果到 specs/003-yolo-cpu-comparison/validation.md。

## Phase 5: US3 — 可复核质量结论 (P2)

**Independent Test**: 固定标注和遮挡证据可重算分类结果；已知合成用例的 full/partial/missed 和误遮并集计数正确；未确认标注不能进入正式统计。

### Tests first

- [ ] T021 [P] [US3] 在 tests/unit/test_yolo_comparison.py 增加计数与报告失败测试：部分/完全漏遮、重叠框不重复计误遮、零分母=null、unknown 误遮、不确定区域、图片与视频分组、pending/disputed 不进正式分母、真实复核信息必填及新增漏遮可定位。

### Implementation and review

- [ ] T022 [US3] 在 scripts/compare-yolo.py 完成 report 质量汇总：按 data-model.md 读取标注及实际遮挡证据，计算 full/partial/missed、连通误遮区域与分类漏遮率；保留 unknown、过大遮挡案例及未复核占比，生成 summary.json、timings.csv 并更新 HTML；通过 T021，不更改输入证据或人工状态。
- [ ] T023 [P] [US3] 在 specs/003-yolo-cpu-comparison/quickstart.md 明确 annotations.json 编辑与复核步骤、目标/争议区域规则和重新生成报告命令；给出与 data-model.md 一致的小型示例，说明原图先标注、结果后复核以及正式统计边界。
- [ ] T024 [US3] 将 data/validation/yolo-cpu-001/index.html 和 annotations.json 交给用户或实际人工复核者核对所有固定样本，记录真实 reviewer/reviewed_at 和结论；代理只能准备与解释，不冒充人工确认。没有人工回复时保持本项未完成，继续所有不依赖确认的文档/测试，并交付暂定报告。
- [ ] T025 [US3] 根据 data/validation/yolo-cpu-001/annotations.json 已有复核状态重新生成报告，将分人脸/车牌的质量变化、全部已确认新增漏遮、误遮/过大遮挡案例、耗时与证据限制写入 specs/003-yolo-cpu-comparison/comparison.md；人工未齐只写暂定结果，正式验收不得标完成。

## Phase 6: Cross-cutting verification

- [ ] T026 按 specs/003-yolo-cpu-comparison/quickstart.md 执行必要回归与真实模型验证，汇总 specs/003-yolo-cpu-comparison/validation.md 的通过/失败/跳过及 SC-001–006 对应证据；不重复耗时全量运行，除非模型、代码、素材变更或证据失效。
- [ ] T027 检查整个特性 diff 与 specs/003-yolo-cpu-comparison/spec.md 的范围一致性：原始媒体/权重未跟踪、无 GPU/训练/新 API 或主工作区改动；修复本次引入问题并更新 tasks.md 完成状态，未运行和人工待办保持未勾选，不自动合并 main。

## Dependencies & Execution Order

```text
T001 → T002 → T003 → T004 → T005
  → US1: (T006 || T007) → T008 → T009 → T010 → T011
  → US2: (T012 || T013) → T014 → T015 → T016 → T017 → T018 → T019 → T020
  → US3: (T021 || T023) → T022 → T024 → T025
  → T026 → T027
```

T023 的编号按文档展示，执行时可在 US3 开始与 T021 同步。T024 尚待人工时可先执行 T025 的暂定报告和 T026/T027 的独立验证，但不得勾选正式复核或整体完成。

US2 的小型 CLI/媒体测试可独立使用替身数据验证，但真实比较依赖 US1；US3 汇总逻辑可用合成记录独立测试，正式结论依赖 US2 真实证据。共享 scripts/compare-yolo.py 的修改必须顺序进行。

## Parallel Examples

- US1：T006 的检测数学测试与 T007 的配置/worker 测试位于不同文件，前置完成后可并行。
- US2：T012 的清单/状态测试与 T013 的合成媒体完整性测试位于不同文件，可并行；T019 两个 CPU 运行必须串行。
- US3：T021 的汇总测试与 T023 的人工复核指南位于不同文件，可并行；T022 修改脚本须在 T021 后进行。

这里只列可并行关系，不要求启动额外代理。

## Requirement Coverage

| 规格 | 任务 |
| --- | --- |
| FR-001、003、011 基线与公平条件 | T001、T005、T015、T018、T019 |
| FR-002 模型范围 | T002–T004、T008 |
| FR-004、015 兼容与失败保护 | T006–T011、T026 |
| FR-005、006 素材及完整处理 | T012–T016、T018–T020 |
| FR-007、008 固定样本与标注 | T014、T018、T023、T024 |
| FR-009、010 分类质量规则 | T021、T022、T024 |
| FR-012、013 可追溯报告与计时 | T015–T017、T020、T022、T025 |
| FR-014 比较结论 | T025、T026 |

## Implementation Strategy

首个可运行增量是 Phase 1–3：模型真实冒烟、检测器接入及兼容回归；这不是用户最终目标，仍需完成 US2 全量 CPU 对比和 US3 复核报告。各阶段验证后继续后续已授权任务，不为例行 checkpoint 再索要批准。只有真实人工复核不能由代理代签；缺回复时交付可查看结果并明确待办。

## Notes

共 27 项：Setup/Foundational 5 项，US1 6 项，US2 9 项，US3 5 项，跨阶段验证 2 项。无 `.specify/extensions.yml`，before_tasks/after_tasks hooks 均无需执行。任务生成不代表任何实现或验收已完成。

## 保存点（2026-10-19）

本次用户要求保存、提交并推送，实施在此保存进度。详见 progress.md；未完整满足的任务保持未勾选，尤其全量比较和人工复核未执行。T008 执行器已由实际兼容性证据改为 ONNX Runtime CPU；T009 worker 已有路径检查可直接复用，无需改动。
