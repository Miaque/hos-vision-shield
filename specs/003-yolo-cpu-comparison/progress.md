# 实施进度保存点

日期：2026-10-19。用户要求保存当前任务进度、提交并推送；本文件是续作入口，不是完成报告。

## 工作位置与已确认范围

- worktree：C:/Users/zhanjh/.codex/worktrees/yolo-comparison/vision-shield
- 分支：feature/yolo-comparison；基线固定 0664be4ae9edb6d0eac7bb2635d5563dad9d7956 的 quality CPU。
- 输入：C:/Users/zhanjh/Downloads/demo，16 张图片、5 段完整视频。
- 同时检测人脸与车牌，只比较 CPU、不训练；优先漏遮挡，兼顾误遮挡，耗时辅助。

## 已保存内容

1. 规格、设计、研究、契约、验证指南、27 项任务清单，以及 GLOSSARY 脱敏术语。
2. yolo 后端接入 Settings/load_detector，保持默认 quality，拒绝 yolo+cuda；复用现有 worker 两条模型路径检查，无须改 worker。
3. YOLO 检测器：固定640 RGB letterbox、源坐标反投影、已有分块布局、两种真实输出解码、扩框及异常校验。
4. 下载脚本和模型 manifest；两份真实权重已下载并校验 SHA-256，基线四模型已复制到 worktree 并校验。
5. 离线比较脚本初版：prepare/run/report、固定样本、基线源码快照、来源哈希、完整性检查、临时结果、人工状态分离及本地报告。已有小型单元测试，尚未完成端到端验证。

## 实际模型发现

- OpenCV 4.14.0 加载 yolov8n-face.onnx 失败：NonZero 算子不支持。改用新增 onnxruntime 1.31.0（uv.lock），仅 CPUExecutionProvider，双模型串行，2个 intra-op线程/1个 inter-op线程。
- 人脸输入 [1,3,640,640]，输出 [1,300,21]，含关键点且内置 NMS；车牌输出 [1,5,8400]，需要外部 NMS。真实 CPU forward 已成功。
- 人脸仓库标记 GPL-3.0，但权重 metadata 声明 AGPL-3.0；manifest 保留双方声明，尚未核定部署许可。不能只引用仓库许可。

## 素材问题与用户授权

“16-32-32疑似狗实际人.mp4”共900帧，第84和160帧有重复时间戳；其余四段分别915、1550、1524、1511帧，时间戳递增。

用户明确同意保留原视频失败记录，并额外生成不丢帧、只修正重复时间戳的副本，供两组使用同一副本补充比较。原始媒体不得改写。

scripts/repair-comparison-video.py 是未完成的修复草稿，首次 remux 被防护检查中止：`Not the authorized duplicate PTS/DTS case`。不得将 data/validation/yolo-cpu-001/repaired-input.mp4 现有部分产物视为成功；尚无 repair.json，manifest 尚未加入副本。续作先定位实际触发的 packet，确认 PyAV 时间基转换及包突变影响，再修复并验证两处时钟之外全部解码像素、帧数及音频 payload 一致。

## 本地产物（不提交/不上传）

- data/validation/yolo-setup/：环境、基线模型校验和保存点测试日志。
- data/validation/yolo-cpu-001/：prepare 已完成的 manifest.json、run.json、annotations.json 和原始采样 PNG；未运行 baseline/candidate。
- annotations.json 目前 targets 为空、review_status=pending，仅为标注壳，不能当作无目标真值。
- data/validation/yolo-preflight-aborted/：第一次因重复时间戳中止的准备产物，保留诊断。
- models/ 权重及 .venv 均本地保留。远程仅包含源码、manifest、文档及测试，不含运行产物。

## 验证状态与续作顺序

- 初次检测选择/坐标/设备/worker 定向测试16项通过；随后比较单测与真实模型探针组合13项通过。
- 曾启动较广回归（tests/unit + 视频/流/runner）；会话句柄已失效，未取得最终结果，不计为通过。本保存点重新执行并落盘定向测试，结果见 validation.md。
- 尚未完成：下载失败保护自检、完整测试覆盖、文档启动示例、时间戳修复、合成媒体比较测试、浏览器核验、两组全量运行和人工标注复核。
- 比较脚本待核对：总 run 状态不能因单组成功就表示两组完成；输出音轨检查目前仅存在性，还须验证时长/时序；计时目前包含采样PNG写盘，报告已披露，正式比较前应统一明确；report 还需验证原视频 HTML/播放和标注字段完整校验。
- detector 保存了 max_side 但网络固定640；需明确该配置在 YOLO 全图策略中的实际意义，不把配置值1280误报为网络输入1280。

## 主干合并（2026-10-20）

- 在主工作区 `E:/workspace/python/hos-vision-shield` 的 `feature/yolo-comparison` 合并 main（27716e1）。保留 YOLO 检测器、比较工具和既有任务状态；模型来源说明保留双方内容。
- 主干 worker 新增强制启动预检，因此补齐 yolo-manifest.json 的两模型身份校验和 ORT CPUExecutionProvider 检查；YOLO 仍仅支持 CPU，默认 quality 不变。
- 按本分支现有 pyproject.toml/uv.lock 重新生成 GPU 部署锁，补入 onnxruntime 及其依赖，共 70 项；根声明与根锁未改。
- 本次回归为 153 passed、1 skipped，详见 validation.md。当前主工作区没有两份 YOLO 权重，未重跑真实 YOLO 模型或媒体对比，也未构建 Docker 镜像；上述全量对比及人工复核待办不变。
- 继续顺序：修复副本及测试 → 补齐上述小缺口与README/.env示例 → 原图标注准备 → 冻结最终源码/脚本指纹 → 两组顺序全量运行 → 浏览器查看 → 人工复核/暂定报告。
- 首次完整运行前不要再修改冻结检测源码；如必须修改，创建新 run，避免混用旧记录。不要拿原本的42产物门槛掩盖原视频失败；补充副本应明确单列。

## 恢复命令

```powershell
Set-Location C:/Users/zhanjh/.codex/worktrees/yolo-comparison/vision-shield
uv sync --frozen
# 先修复并验证修复脚本，再运行：
uv run python scripts/repair-comparison-video.py data/validation/yolo-cpu-001
uv run python scripts/compare-yolo.py run --run-dir data/validation/yolo-cpu-001 --variant baseline
uv run python scripts/compare-yolo.py run --run-dir data/validation/yolo-cpu-001 --variant candidate
uv run python scripts/compare-yolo.py report --run-dir data/validation/yolo-cpu-001
```

原始失败记录会使 run 返回非零，须检查逐素材结果后再继续另一组；不得隐藏失败退出码。没有人工确认时，只能报告暂定视觉差异。当前没有生产部署、GPU或真实RTSP验收结果。
