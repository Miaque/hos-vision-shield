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

## 本轮续作位置（2026-10-20）

- 旧 worktree 已不存在；现位于 `C:/Users/zhanjh/.codex/worktrees/yolo-cpu-evaluation/hos-vision-shield`，分支 `feature/yolo-cpu-evaluation`，从主工作区已合并main的 bad26b6 创建，主工作区未修改。
- 用户确认当前原始素材路径为 `E:/workspace/python/hos-vision-shield/tests/test_assets`（仍16图5视频）。基线四模型已从主工作区复制并校验；YOLO两模型按固定清单重新下载并通过CPU实测。
- 已定位并修复中止原因：原始视频除两处重复视频PTS/DTS外，还有AAC包46016/8000秒处重复；用户在本会话追加授权该音频包时钟增加1/8000秒。修正只针对原始SHA `db44dd...0d67ca`，逐流全部压缩包、900帧像素、其余PTS/DTS均校验相等。repair.json 已生成。
- 比较工具已补：总状态须两组全齐、音频内容及包时序/时长校验、失败/取消钩子恢复、采样PNG移出处理计时、样本定位冻结、工具/配置指纹、完整标注检查、unknown误遮及新增漏遮定位、相对原始媒体预览。README、.env.example、模型说明及复核指南已更新。
- 回归197 passed、1 skipped；修正相对链接后16项定向通过。最新代码提交 a9f5ad2；后续只可修改文档和人工标注，不修改冻结检测/比较源码。
- 正式run：`data/validation/yolo-cpu-003`。原始21份+补充修正1份，98个固定样本。已在查看两组结果前看原图、准备47个代理建议并保存 annotations-before-results.json，全部pending，未构成完整目标真值。空列表不能解释为无目标。
- `yolo-cpu-002` 因浏览器不能访问跨盘file原始链接而主动中止，保留部分产物/运行状态和原因，不并入003；原始预览现为原始文件的哈希校验副本及相对链接。
- baseline已启动，候选仍待baseline完成后串行执行；原始01视频预期失败，另有21修正副本。每组应有21份成功、1份原始失败；总结果原始40/42，补充2/2，不能以合计42掩盖失败。运行未结束，不得写成上述结果已达到。
- 运行句柄失效时查 `data/validation/yolo-cpu-003/baseline/*.json` / `candidate/*.json`、各组log、execution.json及实际进程，不以旧句柄失败推断任务结束。CLI用同一run可跳过哈希一致的complete；有未预期失败先诊断，不开始另一组。
- 本地浏览器预览服务为127.0.0.1:18767（现有Starlette StaticFiles/Uvicorn，支持Range）；普通http.server预览不支持Range，曾造成视频读取等待。已核对22个区块、16原图及6个原始/修正视频元数据，首段实际播放解码5帧。完整输出尚待浏览器检查；后台标签页节电AbortError单列为环境限制。
- 未做人工签名/正式质量统计；未执行GPU、服务器Docker、生产或真实RTSP验收。本续作未推送。

## 最新指令覆盖：仅图片（2026-10-20）

用户明确要求“先处理图片，视频不要全量运行”。已停止yolo-cpu-003活动视频worker，串行驱动因中止退出，不会自动启动视频candidate；002/003的run.json均记录范围变更原因。**不要恢复这两个全量视频run。**

当前仅处理 `data/validation/yolo-images-001`：16张JPEG、16个固定样本、31个原图代理目标建议（全部pending）。候选检测/比较源码冻结于6bc1cce，新增prepare --images-only，测试证明不会打开视频。两组完成门槛32份图片；视频不在本轮交付范围。baseline图片已启动，candidate须在baseline完成后串行执行；最终状态、耗时和浏览器证据待结束后补充。

## 图片交付结果（2026-10-20）

- 当前图片阶段已完成：`data/validation/yolo-images-001/index.html`，16图×2=32份complete；run.status=processed，baseline/candidate退出码均0。
- 候选检测/比较源码冻结于6bc1cce；图片报告、summary.json、timings.csv、image-verification.json、原始标注副本及6个放大案例均已生成。报告/标注属于data忽略目录，不提交或上传。
- baseline处理111.954秒、初始化4.274秒；candidate处理32.755秒、初始化0.431秒。仅为本机单次16图批次，处理耗时不含PNG写盘、完整性检查和报告。
- 浏览器16个区块、96张图片全部加载，无video；明确显示32/32和待人工复核。暂定视觉取舍见comparison.md：有裁切人脸补遮，也有右边缘人脸漏遮风险；保持默认quality。
- 最终回归198 passed、1 skipped（28.14秒），日志data/validation/yolo-setup/images-final-tests.txt。
- 16样本/31个代理建议仍pending，完整标注和人工质量复核未完成，不能写正式准确率。用户已要求本轮不跑全量视频，002/003均已停止；不得自动恢复它们。本轮没有后台自动提醒任务，没有推送。
