# Contracts: YOLO 与本地对比

以下 CLI 是待实现契约，不是当前已经存在的命令。

## 服务与检测器

- 公共 HTTP 请求/响应、任务状态、错误语义与流停止行为保持现有契约，无新增路由。
- 配置新增 `DETECTION_BACKEND=yolo`；要求 `INFERENCE_DEVICE=cpu`，cuda 必须校验失败，不能静默切回 CPU。
- 复用 `FACE_MODEL_PATH`、`PLATE_MODEL_PATH` 指向两份经 manifest 校验的 ONNX；默认 backend 仍为 quality。
- `detect(rgb: np.ndarray) -> list[tuple[float,float,float,float]]`：输入非空 RGB uint8 HWC；输出原图 xyxy 遮挡框，有限、裁剪到边界、扩框仅一次；不改变源像素。
- 任一模型加载/推理/输出校验失败都传播异常，不返回另一模型的部分结果或空列表伪装成功；无目标时正常返回空列表。

## CLI

当前用户范围为仅图片：prepare增加`--images-only`，恰好冻结16张JPEG、scope=images，不打开视频。该标志只用于prepare，run/report始终消费冻结清单；本轮完成门槛为32份图片。下文默认全量视频命令保留为历史计划，本轮不执行。

工作目录为本 worktree。`scripts/compare-yolo.py` 提供三个子命令：

```powershell
python scripts/compare-yolo.py prepare --source C:/Users/zhanjh/Downloads/demo --run-dir data/validation/yolo-cpu-001
python scripts/compare-yolo.py run --run-dir data/validation/yolo-cpu-001 --variant baseline
python scripts/compare-yolo.py run --run-dir data/validation/yolo-cpu-001 --variant candidate
python scripts/compare-yolo.py report --run-dir data/validation/yolo-cpu-001
```

- prepare：拒绝覆盖已有 run；要求 16 张图片、5 段视频，冻结哈希、原始时序、固定抽样、基线提交及运行参数；不启动模型。
- run：核对输入及配置未变，再运行指定版本。自动准备固定提交源码快照并验证实际导入位置；candidate 记录提交加 dirty diff 指纹。只处理未完成的整个素材；已有 complete 结果先校验其指纹。逐项写入状态，失败返回非零退出码。
- report：只读运行证据与标注，生成 index.html、summary.json、timings.csv；缺失结果明确列出。标注 pending 时仍可生成视觉报告，但标题注明“待人工复核”，不得标质量通过。
- 输出目录必须独立于输入目录；不覆盖输入；路径包含空格和中文也能处理；HTML 对文件名和错误文本进行转义。
- report以校验过哈希的originals/只读副本展示原始媒体，全部链接相对run目录，既可从本地打开也可通过本机HTTP预览；不使用跨盘file URL。

## 输出目录

```text
run-dir/
├── run.json
├── manifest.json
├── annotations.json
├── baseline/                 # 16图+5视频及逐素材记录
├── candidate/
├── samples/                  # 固定帧及遮挡证据
├── index.html
├── summary.json
└── timings.csv
```

大媒体及本机源码快照留在 data/ 忽略目录。权重来源与下载哈希进入 models/yolo-manifest.json，报告只引用本地路径，不加载外部脚本或上传素材。

## 错误与复核契约

模型缺失/不兼容、输入哈希变化、输出不完整必须返回明确错误，不静默换模型、改设备或缩短视频。部分结果可查看但总状态为 incomplete。

人工只编辑 annotations.json 的目标、争议项及复核字段；保留 sample_id 和原始帧定位。report 验证分类计数、范围和分母；unknown 误遮不硬分入人脸或车牌。数据字段与状态见 [data-model.md](../data-model.md)。
