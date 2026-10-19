# Quickstart: 实施后的 CPU 验证流程

本指南定义后续实现应支持的验证步骤。当前仅完成设计，下面新增的下载脚本、比较脚本及测试尚未实现；不能将命令列出视为已经运行通过。

## 前提

- 在 `C:/Users/zhanjh/.codex/worktrees/yolo-comparison/vision-shield` 的 feature/yolo-comparison 分支执行。
- Python 3.12 与项目锁定依赖可用；原始目录包含 16 张图片、5 段 MP4。
- 基线四份模型按现有 manifest 准备到本 worktree 的 models/，可从本机主工作区复制并验证哈希，不修改主工作区。
- 下载并校验两份候选 ONNX，记录许可、实际 I/O 和 NMS；真实 CPU 冒烟通过后才执行完整对比。
- 可用磁盘足以保存两套全长视频、采样画面及源码快照；不要以缩短视频替代磁盘或耗时问题的明确报告。

## 安装与模型（脚本待实现）

```powershell
Set-Location C:/Users/zhanjh/.codex/worktrees/yolo-comparison/vision-shield
uv sync --frozen
./scripts/download-yolo-models.ps1
uv run pytest tests/unit/test_yolo_detection.py tests/unit/test_detector_selection.py tests/unit/test_inference_device.py tests/unit/test_worker_models.py tests/unit/test_yolo_comparison.py -q
uv run pytest tests/integration/test_yolo_model_probe.py -q
```

预期：校验固定哈希，CPU 真正 forward 成功；非空 RGB 校验、坐标逆变换、边界/NMS/扩框、空检测与异常传播通过。若 ONNX 契约或算子不兼容，停止全量运行，记录失败并更新方案；不自动下载其他权重替代。

## 固定输入并完整运行（CLI 待实现）

```powershell
uv run python scripts/compare-yolo.py prepare --source C:/Users/zhanjh/Downloads/demo --run-dir data/validation/yolo-cpu-001
uv run python scripts/compare-yolo.py run --run-dir data/validation/yolo-cpu-001 --variant baseline
uv run python scripts/compare-yolo.py run --run-dir data/validation/yolo-cpu-001 --variant candidate
uv run python scripts/compare-yolo.py report --run-dir data/validation/yolo-cpu-001
```

运行脚本应显式构造 Settings(_env_file=None)，避免继承主工作区 .env 中的设备或模型路径；记录实际导入路径，确认 baseline 确实来自 0664be4 快照。两组顺序执行，固定 CPU，禁止自动 GPU。

预期：42 份完整媒体产物；两组每段视频解码帧数、时序及尺寸对应，音轨存在/时长保持；初始化和处理耗时分开。未完成项显式列出。不得用历史 104 帧片段报告代替本轮结果。

## 人工复核

打开 run 目录下 index.html，对全部图片及每段每 5 秒加末帧的固定样本逐项复核。先核对原始目标标注，再对比两组遮挡；目标按 full/partial/missed 记录，误遮及遮挡过大单列，无法判定的样本标 disputed。复核人和时间必须真实，代理预标注保留 pending。

更新 annotations.json 后重跑 report，无须重跑模型。只有人工确认样本进入正式分类漏遮率；待复核、争议和 unknown 误遮数量必须可见。若缺少人工复核，只能交付暂定视觉对比，不宣称完整质量验收通过。

## 兼容性回归

```powershell
uv run pytest tests/unit tests/integration/test_video_pipeline.py tests/integration/test_stream_pipeline.py tests/integration/test_runner_lifecycle.py -q
```

验证现有错误、取消和媒体资源回收，新增 yolo 配置与 worker 缺失模型检查。需要 Redis/发布器等环境的测试按其原有前提执行；跳过和缺依赖必须报告，不算通过。

通过本地文件对比不等于真实 RTSP、生产部署或 GPU 验收。不设 YOLO 必须更好或更快的通过条件；最终报告应能清楚看出改善、退化、取舍与证据不足。
