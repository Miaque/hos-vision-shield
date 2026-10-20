# 图片 CPU 对比：暂定结果（2026-10-20）

16张图片的两组CPU处理已完成，32/32份输出均完整、哈希有效；本轮没有执行全量视频。YOLO本批次处理更快，但画面中既有补遮改善，也有新增漏遮风险，不能据此替换默认quality或声称总体准确率更高。

## 查看结果

- [16组三栏图片报告](../../data/validation/yolo-images-001/index.html)，原始 / baseline / candidate；每张完整图片可单独打开。
- [图片完整性与计时](../../data/validation/yolo-images-001/image-verification.json)、[逐图耗时CSV](../../data/validation/yolo-images-001/timings.csv)。
- [待人工复核的原图标注](../../data/validation/yolo-images-001/annotations.json)，16样本、31个代理目标建议，全部pending；[结果生成前的标注副本](../../data/validation/yolo-images-001/annotations-before-results.json)。
- 本机预览：`http://127.0.0.1:18767/yolo-images-001/index.html`，只监听回环地址；预览关闭后仍可从本地打开HTML，所有媒体均用报告目录内的相对路径。

## 本轮条件

输入为用户确认的 `E:/workspace/python/hos-vision-shield/tests/test_assets` 中16张JPEG。`prepare --images-only`不打开视频；清单16份，scope=images。独立worktree为 `C:/Users/zhanjh/.codex/worktrees/yolo-cpu-evaluation/hos-vision-shield`，分支 `feature/yolo-cpu-evaluation`。

baseline固定 `0664be4ae9edb6d0eac7bb2635d5563dad9d7956` 的quality CPU源码快照，candidate冻结于 `6bc1cce`。同一解释器、锁定依赖、CPU及模型，顺序运行。quality采用1280最长边配置和既有分块/局部确认；YOLO每个全图/分块固定640输入，双模型ORT CPU串行。padding配置均为0.15，不更改基线后处理。

## 值得优先看的画面

以下仅为代理对实际输出的视觉观察，未由人工签字确认，不计为正式质量结论。

| 样本 | 观察与取舍 | 放大对比（左原始、中基线、右YOLO） |
| --- | --- | --- |
| 01 / `1aacfe36…jpg` | 两辆白车的牌照遮挡均存在，候选对远处车牌覆盖更宽；需原尺寸核对基线边缘。 | [车牌局部](../../data/validation/yolo-images-001/cases/01.png) |
| 06 / `20261014_145251_607523_visual_events.jpg` | 候选增加若干侧脸/背侧区域遮挡；前景白衣人物的遮挡扩展到颈肩，需核对过大遮挡代价。 | [人群局部](../../data/validation/yolo-images-001/cases/06.png) |
| 07 / `226dbd04…jpg` | 图像上边缘只露出部分脸的近处人物，候选新增遮挡，基线未覆盖该处；后方两人两组均有遮挡。 | [裁切人脸](../../data/validation/yolo-images-001/cases/07.png) |
| 11 / `adbf33e1…jpg` | 最右边缘人物的脸在基线有遮挡，候选未遮挡，是优先人工复核的新增漏遮风险；后排人物遮挡边缘也需检查。 | [边缘人脸风险](../../data/validation/yolo-images-001/cases/11.png) |
| 10 / `8dcb4e9c…jpg`、12 / `c4d37ae3…jpg` | 两组都有可见遮挡，但大小与边缘不同；建议多边形是粗略初标，不能直接把其partial标签当成确证漏遮。 | [低头人脸](../../data/validation/yolo-images-001/cases/10.png)、[侧脸](../../data/validation/yolo-images-001/cases/12.png) |

[代理建议区域的覆盖定位](../../data/validation/yolo-images-001/provisional-target-coverage.json)只用于定位上述画面，包含不完整/有争议的建议区域，既不是准确率，也不是已确认新增漏遮统计。没有在annotations.json中冒充human_confirmed。

## 本次耗时

| CPU版本 | 初始化 | 16图累计处理 | 每图平均 |
| --- | ---: | ---: | ---: |
| 固定quality基线 | 4.274秒 | 111.954秒 | 6.997秒 |
| YOLO候选 | 0.431秒 | 32.755秒 | 2.047秒 |

本次累计处理时间比约3.42，候选耗时约为基线的29.3%。这是单次、同机批次辅助测量，未建立稳定性能分布。处理计时包含解码/检测/遮挡/编码及少量证据记录，不包含初始化、输出完整性验证、采样PNG写盘或报告生成，也不是HTTP请求端到端延迟。

## 证据边界

正式人工确认样本为0，待复核16，未复核占比100%；正式人脸/车牌漏遮率分母为0，均为null。正式误遮计数中的0仅表示没有正式样本，不能解释为没有误遮。空targets仍是pending标注壳，不是无目标真值；31个建议不保证目标数量完整。

浏览器已验证16个素材区块、96/96个原图/结果/固定样本图片加载，三栏布局，图片范围提示、32/32状态和中文名称；video元素为0。输出哈希、尺寸/方向和固定采样证据均由工具核验。相关回归198 passed、1 skipped，详见validation.md。

用户已明确要求不跑全量视频，原有002/003视频worker和串行驱动已停止；那些中止产物不并入本轮。原始重复时钟视频与修正副本的诊断证据保留，未宣称视频完成。未执行生产、服务器Docker、GPU或真实RTSP验收；YOLO权重许可差异也仍待未来部署时核定。本轮不更改默认quality。
