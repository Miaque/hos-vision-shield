# Data Model: CPU 对比记录

全部实体为本地文件记录，不新增数据库表；媒体存于 Git 忽略目录，标注不包含身份或车牌文字。

## Run（一次对比）

- run_id、created_at、baseline_commit、candidate_commit、candidate_diff_sha256、实际源码路径与指纹。
- environment：Python/依赖版本、CPU、线程数、device=cpu；models：角色、来源/revision、sha256、许可、实际输入输出与预后处理参数。
- manifest_sha256、annotation_revision；status：prepared → running → processed → reviewed；任一素材失败时为 incomplete，可在输入和配置不变时重跑该完整素材。
- processed 仅代表媒体完整处理；reviewed 需要指定样本全部有人工确认或明确争议状态，不等于模型更好。

## Media（21 份输入）

- media_id=内容 SHA-256；源文件名/路径、kind=image|video、尺寸、图像方向。
- 视频：解码帧数、逐帧序号与有理数 PTS/time_base、时长、音轨信息；不能信任本批素材值为 0 的帧数元数据。
- 原媒体只读；run 下每份媒体关联 baseline 和 candidate 两条 Result。

## Sample（质量样本）

- sample_id、media_id、frame_index（图片为 0）、PTS/time_base、原图尺寸与抽样原因。
- 全部图片；每段视频从起始时刻每隔 5 秒选首个达到该时刻的帧，补末帧并去重；变帧率不按固定 FPS 推算帧号。
- targets：target_id、kind=face|plate、需要遮挡的可见区域、uncertain 标记；不记录字符内容。
- review_status=pending|human_confirmed|disputed、reviewer、reviewed_at；代理可写初始建议，但不能自动标 human_confirmed。
- 先基于原画面标注目标，再查看两组遮挡，减少结果引导；区域允许多边形以避免矩形空白影响覆盖判断。

## Result（每素材每版本）

- variant、status=pending|running|complete|failed|cancelled、output_path、error_code。
- init_seconds、processing_seconds、timing_scope、measurement_count=1、processed_frames、output_integrity。
- 成功必须存在完整可解码产物并通过完整性核对；part 文件不算成功；失败信息不泄露敏感 URL。
- 抽样 evidence：实际遮挡区域及对应输出帧。视频包含既有短时保持，不能只用当帧新检测框。

## Judgment（每样本每版本）

- 每个确定目标的 outcome=full|partial|missed；可见敏感区域遮挡并集覆盖 100% 为 full，覆盖为 0 为 missed，其余为 partial；争议区域单列，不纳入分母。
- false_mask_regions：与任何真实目标均不对应的遮挡连通区域，跨框重叠先求并集再计数，避免分块重复框膨胀误遮数；无法归属于 face/plate 的历史遮挡标 unknown，不强行猜类。
- oversized_cases：与真实目标对应但范围过大的遮挡案例，单列人工说明，不混算 false_mask。
- 分类漏遮率=(partial+missed)/(full+partial+missed)，零分母为 null。
- 真实目标可按 face/plate 分组；误遮按可追溯类别分组并保留 unknown 组，不能修改基线推理来制造类别信息。
- 图片与视频样本分开汇总；正式统计只用 human_confirmed，pending 与 disputed 数量及占比必须展示。

## 校验不变量

SHA-256 格式有效；坐标为有限数且位于规范化图像内；目标类别合法；同一 sample_id/variant/target_id 不重复；计数之和等于可判定目标数。报告生成不得更改清单、配置或人工复核状态。人工修改标注生成新 revision，可重新汇总而不重跑模型。
