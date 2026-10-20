# CPU 对比与人工复核

**当前只执行图片。** 用户已明确要求“先处理图片，视频不要全量运行”；使用本文末尾的“图片优先”四条命令。下方包含视频的流程是历史计划，本轮不要执行或恢复 yolo-cpu-002/003。

在独立 worktree 执行。当前续作目录为 `C:/Users/zhanjh/.codex/worktrees/yolo-cpu-evaluation/hos-vision-shield`；原始素材已迁至 `E:/workspace/python/hos-vision-shield/tests/test_assets`，共 16 张 JPEG、5 段 MP4。原始文件只读，权重和运行产物不提交 Git。

## 环境与模型

```powershell
uv sync --frozen
./scripts/download-yolo-models.ps1
uv run --no-sync pytest tests/unit/test_yolo_download.py tests/unit/test_yolo_detection.py tests/integration/test_yolo_model_probe.py -q
```

还需按原有 manifest 准备 `face.jit`、`license_plate.jit`、`yunet_face.onnx`、`hyperlpr_plate.onnx`。prepare 会校验全部六份模型。YOLO 固定 640 输入，quality 最长边 1280；这两项不能混写为相同网络输入尺寸。模型来源和许可差异见 models/README.md。

## 历史全量流程（本轮不执行）

```powershell
uv run --no-sync python scripts/compare-yolo.py prepare --source E:/workspace/python/hos-vision-shield/tests/test_assets --run-dir data/validation/yolo-cpu-003
# 本批原视频有两处视频和一处音频重复时钟，用户已授权生成无损修正副本：
uv run --no-sync python scripts/repair-comparison-video.py data/validation/yolo-cpu-003
```

prepare 解码整段素材，以实际 PTS 从起点每 5 秒及末帧固定抽样；先查看 `samples/*-original.png` 并编辑原始目标标注，再查看模型结果。初始 `targets=[]`、`review_status=pending` 只是标注壳，不是“没有目标”的真值。

修复仅修改两处视频 PTS/DTS 各 1/90000 秒及一处 AAC PTS/DTS 1/8000 秒，逐流核验压缩包、逐帧核验解码像素和帧数；相邻包时长由封装器按新 DTS 推导。原始视频的失败记录保留，修正副本单列；禁止把补充结果算作原始素材成功。repair.json 是修复证明，缺少它的 part 文件不能使用。

在首次完整运行前完成代码、脚本和参数修改；prepare 冻结源码（含未跟踪 Python 文件）、工具、依赖锁、参数、模型、清单及样本定位。之后改动这些内容须创建新 run。修改目标标注和真实人工复核字段可直接重新生成报告。

```powershell
uv run --no-sync python scripts/compare-yolo.py run --run-dir data/validation/yolo-cpu-003 --variant baseline
# 上一步因原始重复时钟视频返回非零是显式失败，先核对逐项记录，再执行另一组。
uv run --no-sync python scripts/compare-yolo.py run --run-dir data/validation/yolo-cpu-003 --variant candidate
uv run --no-sync python scripts/compare-yolo.py report --run-dir data/validation/yolo-cpu-003
```

两组必须顺序执行，不同时争用 CPU。baseline 来自固定 `0664be4` 源码快照；candidate 来自当前 worktree。子进程明确验证导入位置并以 `Settings(_env_file=None)` 隔离 .env/环境覆盖。模型在每组驻留；完整处理耗时包含解码、检测、实心遮挡、编码，初始化另计，完整性检查、采样 PNG 写盘和报告生成均不计入处理时间。计时为单次辅助证据，不是业务请求端到端延迟。

中断后重复同一 run 命令，仅跳过哈希仍正确的 complete 整份产物；失败素材重跑整份，不续接部分视频。总状态只有两组全齐才为 processed。本批保留原始失败时总状态仍为 incomplete，报告分别列原始 42 份及补充 2 份的进度。

## 标注与复核

annotations.json 每个样本的 `id/media_id/frame_index/pts/time_base/width/height/kind` 都是冻结定位字段，不可编辑。仅编辑 targets、review_status、reviewer、reviewed_at 和 oversized_cases。坐标使用已纠正方向的原图像素，均为有限数；多边形至少 3 点，x 在 `[0,width)`、y 在 `[0,height)` 内。

```json
{
  "id": "00-000000", "media_id": "00", "frame_index": 0,
  "kind": "image", "width": 1920, "height": 1080,
  "targets": [
    {"id": "face-1", "kind": "face", "polygon": [[100,100],[120,100],[120,125],[100,125]], "uncertain": false},
    {"id": "plate-1", "kind": "plate", "polygon": [[300,300],[330,300],[330,310],[300,310]], "uncertain": true}
  ],
  "review_status": "pending"
}
```

先基于原画面圈出可见敏感区域，类别仅 face/plate，不抄写身份或车牌字符。模糊且无法判定的区域标 `uncertain=true`，不要默认为无目标。随后在 index.html 对照两组及固定样本；同一样本内目标 id 唯一。代理只能给 pending 建议。

实际人工完成后，使用真实 reviewer、带时区 ISO 时间（例如 `2026-10-20T10:00:00+08:00`）并改为 human_confirmed；存在争议可标 disputed。不得由代理代签。过大遮挡可用 `oversized_cases: [{"variant":"candidate","note":"此目标周围遮挡过大"}]` 单列。复核时间示例不是已发生的确认。

重跑 report 即可更新 summary.json、timings.csv 和 HTML，不运行模型、不改变输入或人工状态。正式计数仅使用 human_confirmed 且有两组证据的样本；遮挡并集覆盖敏感区域 100%/部分/0 分别为 full/partial/missed；零分母为 null。误遮先合并连通区域，不归属于任何目标的区域计一次；uncertain 区域不计误遮，基线不暴露类别，所以误遮保留 unknown。图片与视频样本分组，候选新增漏遮可由样本链接定位。没有人工确认时只能交付暂定视觉报告。

## 回归

```powershell
uv run --no-sync pytest tests/unit tests/integration/test_yolo_comparison_media.py tests/integration/test_yolo_model_probe.py tests/integration/test_video_pipeline.py tests/integration/test_stream_pipeline.py tests/integration/test_runner_lifecycle.py -q
```

需要本机 FFmpeg，真实模型冒烟另需六份已校验模型；外部 Redis/RTSP 测试仍按原有前提单独执行。通过本地文件比较不等于真实 RTSP、服务器部署或 GPU 验收。详细命令、结果与未运行项写入 validation.md。

## 图片优先（当前执行范围）

最新指令是先完成图片、视频不要全量运行。使用独立目录冻结16张图片；与前面的全量视频计划分开，不运行repair或任何视频run：

```powershell
uv run --no-sync python scripts/compare-yolo.py prepare --source E:/workspace/python/hos-vision-shield/tests/test_assets --run-dir data/validation/yolo-images-001 --images-only
uv run --no-sync python scripts/compare-yolo.py run --run-dir data/validation/yolo-images-001 --variant baseline
uv run --no-sync python scripts/compare-yolo.py run --run-dir data/validation/yolo-images-001 --variant candidate
uv run --no-sync python scripts/compare-yolo.py report --run-dir data/validation/yolo-images-001
```

本轮期望16×2=32份图片；run的scope=images，清单不含视频，报告明确声明视频未纳入。不要对yolo-cpu-002/003调用run；其中中止的视频不续跑，不计为通过。
