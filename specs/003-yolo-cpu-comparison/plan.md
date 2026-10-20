# Implementation Plan: YOLO CPU 脱敏效果对比

**Branch**: `feature/yolo-comparison` | **Date**: 2026-10-19 | **Spec**: [spec.md](spec.md)

**Input**: `specs/003-yolo-cpu-comparison/spec.md`

## Summary

新增可显式选择的 yolo 检测器，使用两个专用 ONNX 权重和现有 OpenCV CPU 推理，复用图片、视频、流与遮挡流程。固定 main 0664be4 的 quality CPU 为基线，在同机完整处理 demo 的 16 张图片、5 段视频，交付本地对比结果、固定抽样复核与逐素材耗时。默认 quality 不变，不训练、不做 GPU 对比。

## Technical Context

**Language/Version**: Python >=3.12,<3.13；Windows PowerShell 本地验证。

**Primary Dependencies**: 现有 OpenCV、NumPy、Pillow、PyAV；基线继续使用现有 Torch/Torchvision；Web 与队列依赖不变。首选不新增运行依赖。

**Storage**: 本地 models/ 权重、data/validation/ 下按 run_id 分目录的结果；原 demo 只读；不改 Redis 数据模型。

**Testing**: pytest 单元及现有媒体管线集成测试；真实 ONNX 冒烟；21 份媒体完整运行；人工复核单独记录。

**Target Platform**: 本机 CPU；OpenCV 2 线程，基线 Torch 4 线程，单素材单进程顺序执行。

**Project Type**: 现有异步媒体脱敏服务，加一个离线对比入口。

**Performance Goals**: 本轮无速度提升门槛；记录单次逐素材处理与初始化耗时，优先完整呈现质量变化。

**Constraints**: 不改主工作区、不自动合并；不静默回退、不截短视频；不共享并发访问同一个 DNN 网络；流处理保持有界缓冲及取消语义。

**Scale/Scope**: 16 张 JPEG、5 段约 60–62 秒 MP4，合计约 304 秒。帧数元数据为 0，必须解码计数，不当作空视频。

## Constitution Check

| 原则 | 设计前 | 设计后及验证要求 |
| --- | --- | --- |
| I 媒体隐私与契约 | 通过 | 仅人脸/车牌检测替换，无 OCR、新路由或任务字段；错误不返回原媒体成功结果 |
| II 异步与有界执行 | 通过 | 模型继续在现有媒体子进程运行；离线评估顺序执行；不改队列、取消、背压 |
| III 职责与共享流程 | 通过 | 新检测器实现现有 detect(rgb) 协议；复用 process_image/process_video/FrameRedactor |
| IV 最小实现 | 通过 | 无训练平台、插件层或新服务；配置、启动检查、模型交付与必要评估脚本为完整变更范围 |
| V 可验证正确性 | 通过 | 模型输出/坐标单测、真实权重冒烟、媒体回归、完整性与人工复核分开验收 |

以上为设计合规，不是测试通过。没有宪章例外。真实模型兼容性为首个实施门槛；未通过不得开始正式对比。

## Project Structure

### Documentation (this feature)

```text
specs/003-yolo-cpu-comparison/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/evaluation.md
└── checklists/requirements.md
```

### Source Code (planned changes)

```text
src/vision_shield/
├── config.py                     # yolo 选项；禁止 yolo + cuda
├── detection/__init__.py          # 显式选择，不改其他后端
├── detection/yolo.py              # 两类检测、坐标与扩框
└── jobs/worker.py                 # 只校验当前后端所需模型
scripts/
├── download-yolo-models.ps1       # 固定来源及哈希、原子下载
└── compare-yolo.py               # prepare/run/report 三个固定阶段
models/
├── yolo-manifest.json            # 权重来源、类别、许可、I/O
├── NOTICE.md
└── README.md
 tests/unit/                     # 配置/选择/worker、YOLO解码、评估汇总
 tests/integration/              # 真实权重冒烟，复用媒体生命周期测试
.env.example / README.md / .gitignore
```

**Structure Decision**: 不改 API、任务存储及媒体编码结构。复用 `face_model_path`、`plate_model_path`，yolo 由配置显式指向两份 ONNX，不再增加重复模型路径配置。扩框使用现有 0.15 配置；候选车牌横向 0.30、纵向 0.15，人脸四向 0.15。基线原有各模型扩框保留，报告说明来源差异，不修改 main 来强行统一。

## Implementation Design

1. **固定模型并验证契约**：采用 research.md 指定的 YOLOv8n-face 和 YOLOv11s-plate ONNX。下载后核实 SHA-256、实际输入输出、类别及是否内置 NMS；用当前 cv2.dnn 在 CPU 上加载并跑非敏感合成帧。不能从文件名猜张量结构；不支持的算子或结构须显式失败，记录原因并修订研究决定后再继续，不自动添加 fallback。
2. **最小检测接入**：`detect(rgb)` 接收非空 uint8 RGB，输出原图 xyxy。每个媒体子进程加载两份网络一次并串行运行；使用 OpenCV CPU backend/target。复用 `cpu.Detector._views` 的现有静态分块布局，无须加载 CPU 模型；全图与重叠分块共同覆盖小目标。每个视图按模型固定尺寸等比缩放、letterbox，记录实际缩放比例与 pad 以反投影；不把 RGB 再误交换为 BGR。按验证的张量布局解析，拒绝非有限数、非法分数/维度，裁剪越界框。
3. **固定后处理**：首轮类别阈值 0.25、外部 NMS IoU 0.45；仅在未内置 NMS 时执行外部 NMS。内置 NMS 的阈值限制须记录，不能声称可调到其导出下限以下。NMS 在每视图、每类内完成；跨视图保留遮挡并集，不因去重缩小遮挡。不额外加追踪或投票。图片反馈修复后，YOLO分数低于0.6的候选须经同类模型局部复核，IoU至少0.2；取最长边3倍、至少640源像素，人脸失败后再试至少480源像素，网络输入仍640。通过时保留原框与原扩框，推理/输出异常继续使整帧失败；规则与源码在新run前冻结。
4. **保持媒体行为**：复用 `redact`、`FrameRedactor` 及现有 process_image/process_video。视频逐帧处理，200ms 遮挡保持原样；评估视频帧来自完整输出而非孤立重跑。模型加载/推理错误沿既有异常链传播；worker 的模型检查对 yolo 仅检查两份指定 ONNX。
5. **固定对比输入**：prepare 阶段计算全部输入 SHA-256，解码视频确定真实帧数/时间戳，每 5 秒取首个达到该相对时刻的帧，加末帧，按帧索引去重；所有图片入样。先保存清单再运行两组。抽样只决定复核范围，不决定处理范围。
6. **隔离运行**：基线从固定提交导出到评估目录内的只读源码快照，候选使用新 worktree。相同解释器及锁定依赖，通过独立子进程和明确源码路径运行；记录实际导入路径/源码指纹，防止 editable install 错导入另一版本。两组依次运行，同次运行各媒体完成后写结果；中断只允许重新执行未完成的整个素材，不拼接半段视频。
7. **报告与复核**：本地静态 HTML 三栏图片及视频，同一采样帧可并排查看；JSON 保存可重算记录。预编码遮挡区域判断采用现有 redactor 的实际框并集与历史保持规则，最终解码输出用于人工视觉复核；不以黑像素判定遮挡，因为原画面也可能是黑色。只在评估脚本内记录所需遮挡证据，不新增生产回调或输出字段。
8. **完整性与计时**：输出临时文件成功关闭并核对后才标 complete。分别核对帧数、尺寸、PTS/time_base 对应及音轨存在/时长；图片核对方向、尺寸。初始化另计；单次处理含解码、检测、遮挡、编码，不含报告/人工复核/输出核对。保持一进程模型驻留的批处理口径，不冒充 API E2E。

## Validation and delivery order

1. 模型下载完整性及 cv2 冒烟 → 验证实际模型契约；失败则先修订方案。
2. 坐标、输入输出、无目标及选择/worker 测试 → 实现 yolo；旧后端回归通过。
3. prepare 与计数/汇总测试 → 固定清单，标注原图与采样帧；记录 pending/human_confirmed。
4. 两组完整运行 → 42 份产物及完整性证据，全部失败项显式记录。
5. 人工复核与报告 → 分类型漏遮/误遮、过大遮挡案例及耗时；未复核时只给暂定结论。

下一步用 speckit-tasks 拆分执行任务。本计划不包含已运行结果；许可信息与模型兼容性边界见 research.md。

## Complexity Tracking

无宪章违反项；不引入新数据库、模型服务、训练流程或通用评估框架。

## 实施修订

固定人脸 ONNX 的 NonZero 算子在 OpenCV 4.14.0 中实际加载失败，YOLO 执行器改为 ONNX Runtime CPUExecutionProvider（2 个推理线程，1 个 inter-op 线程，串行双模型）。其余媒体、基线与权重选择不变。以下实施以这一修订覆盖此前 OpenCV DNN 运行方案；实际输入/输出将在模型 manifest 记录。

## 当前图片优先增量（2026-10-20）

用户已取消本轮全量视频：停止 yolo-cpu-003 的活动worker，不自动开始其candidate。新run yolo-images-001冻结16张JPEG，使用新增的prepare --images-only；后续run/report仍只消费冻结清单，无须运行时过滤或修改清单。两组顺序处理，图片成功门槛32份。视频任务保留为后续计划，不作为本轮未完成项追跑。
