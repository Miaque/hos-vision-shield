# 保存点验证（2026-10-19）

- 命令：`.venv/Scripts/python.exe -m pytest tests/unit/test_yolo_detection.py tests/unit/test_yolo_comparison.py tests/unit/test_detector_selection.py tests/unit/test_inference_device.py tests/unit/test_worker_models.py tests/integration/test_yolo_model_probe.py -q`
- 结果：21 passed in 9.31s；日志 data/validation/yolo-setup/checkpoint-tests.txt。
- compileall：YOLO 检测器、比较脚本和时间戳修复草稿语法检查通过。
- git diff --check：通过。
- 两份实际 ONNX 权重哈希及 CPU forward 已验证。OpenCV NonZero 不兼容由 ONNX Runtime CPU 执行器解决。
- 广泛回归未取得最终输出，不计通过；全量媒体比较、修复副本像素一致性和浏览器检查未完成。
- 人工标注未完成，不能宣称准确率或效果提升。详细待办见 progress.md。

## 主干合并验证（2026-10-20）

- 合并 main 的模型来源说明冲突时保留 EgoBlur/YuNet/HyperLPR 与 YOLO 的原有来源、许可及 metadata 差异声明。
- 先确认 GPU 锁一致性测试因缺少 onnxruntime 失败，以及 YOLO 启动预检的两项用例失败；补齐模型清单/CPU provider 兼容和 GPU 锁后，相关 35 项测试通过。
- `uv sync --locked` 安装本分支已锁定的 onnxruntime 1.31.0、flatbuffers 与 protobuf；未下载权重，未安装 Docker。根 pyproject.toml/uv.lock 与合并前 YOLO 分支一致。
- `uv run --no-sync pytest tests/unit tests/integration/test_runner_lifecycle.py -q`：153 passed、1 skipped，14.72 秒。跳过项为 POSIX 冷加载父 EOF，Windows 结果不替代 Linux 实测。
- GPU 部署锁重新生成 70 项，安装 dry-run 与共同版本检查通过；没有实际在本机安装 Linux CUDA 依赖。
- 主工作区没有 YOLO 的两份真实权重，未执行 test_yolo_model_probe.py；没有重跑媒体全量比较、人工效果复核或服务器 Docker 验收，不以单元测试代替。

## 续作验证（2026-10-20）

- 当前 worktree：`C:/Users/zhanjh/.codex/worktrees/yolo-cpu-evaluation/hos-vision-shield`，分支 `feature/yolo-cpu-evaluation`；主工作区未改。锁定环境通过 `uv sync --frozen` 重新准备，两份 YOLO 权重按固定 manifest 重新下载，四份基线权重从主工作区只读复制并由 prepare 核验。
- 原素材现位于 `E:/workspace/python/hos-vision-shield/tests/test_assets`，用户在本会话确认。实际解码帧数仍为 900、915、1550、1524、1511。
- 修复草稿中止的根因：除第84、160帧外，AAC第46个包也重复 DTS=PTS=46016、time_base=1/8000。用户明确追加授权该包增加1刻度；不是 PyAV 时间基换算造成的误判。新修复以有理时间比较，拒绝倒退或 PTS/DTS 不一致，只允许指定原始 SHA-256 和2个视频+1个音频重复时钟。
- 实际修复验证：900帧逐帧 RGB 相等；视频900包和AAC470包逐流压缩字节相等，逐包 PTS/DTS 仅授权三处变化；其余时钟相等。封装器按新 DTS 重排跨流包顺序并推导相邻 duration，因此校验不能按全局 demux 顺序配对。repair.json 只在验证成功后写入，部分文件不加入 manifest。
- 时间基与 mux 约定参考 [PyAV 官方时间文档](https://pyav.org/docs/stable/api/time.html)，结论以本机逐包验证为准。
- `uv run --no-sync pytest tests/unit tests/integration/test_yolo_comparison_media.py tests/integration/test_yolo_model_probe.py tests/integration/test_video_pipeline.py tests/integration/test_stream_pipeline.py tests/integration/test_runner_lifecycle.py -q`：**197 passed、1 skipped，25.94秒**；日志 `data/validation/yolo-setup/resumed-tests.txt`。唯一跳过仍为 Windows 下 POSIX 父 EOF 专用用例。
- 覆盖：下载失败/错哈希不覆盖旧模型、双模型真实CPU forward与图片/24帧合成视频/共享流帧函数、RGB letterbox逆映射/分块偏移/扩框一次/异常传播、变帧率实际解码及末帧、完整性/音频内容与时序、保持框证据不改变生产像素、失败/取消恢复钩子、黑色原像素不作为遮挡证据、重复/非法标注、未知误遮并集、零分母、争议及新增漏遮定位、整组状态、源码冻结、恢复跳过已完成产物及输出哈希变更拒绝。
- 故意反转总状态的完成条件，单项回归如预期失败；恢复文件后通过。没有使用绿色测试代替条件生效的证据。
- 浏览器工具禁止直接打开 file URL；原稿跨盘原始媒体链接在 HTTP 预览中不可用。已改为报告目录内哈希校验的只读原始副本和相对链接；相关16项定向测试通过。保留 yolo-cpu-002 部分运行与中止原因，创建独立 yolo-cpu-003，不混入旧产物。
- 本地预览使用现有 Starlette StaticFiles/Uvicorn（仅127.0.0.1:18767，支持Range）。普通 `python -m http.server` 不支持视频Range，不能据此判断MP4不可播放。浏览器已确认22个素材区块、16张原图加载成功、6个原始/修正视频元数据可读，首段实际播放已解码5帧；全部正式输出的浏览器核验仍待全量运行结束。后台标签页会被浏览器节电策略暂停，不能把 AbortError 算成媒体损坏。
- 当前候选检测/工具源码冻结于 a9f5ad2，正式目录 `data/validation/yolo-cpu-003`。98个原始固定样本的初看及47个代理目标建议先于两组输出，全部 pending；它们不构成完整真值，也没有人工 reviewer 代签。
- 尚未完成：两组全量运行及正式输出浏览器检查、完整原图标注/人工效果复核、正式分类质量结论。未执行GPU、生产部署、真实RTSP接入或服务器Docker验收。

## 图片范围最终验证（2026-10-20）

- 用户最新要求：先处理图片，视频不要全量运行。已停止003的活动worker及其串行驱动；没有启动视频candidate，没有设置后台自动化，也不再恢复002/003。
- 新增 `prepare --images-only` 只冻结16张JPEG。失败测试先证实参数不存在，再完成实现；26项比较/合成媒体用例通过，新用例使用损坏MP4占位证明图片prepare不打开视频。
- 正式图片run为 `data/validation/yolo-images-001`，候选冻结于6bc1cce，基线仍为0664be4。baseline/candidate顺序运行，各16份complete，二者CLI退出码均0，run.status=processed，summary为32/32。全部输出SHA-256匹配，各图尺寸、方向与证据数量通过完整性检查。
- baseline初始化4.273826秒，16图累计处理111.953541秒；candidate初始化0.431042秒，累计处理32.755061秒。PNG输出/复核与初始化不计入处理耗时；不把这次单样本计时当稳定延迟保证。
- 原图建议按内容SHA从两组结果生成前的副本迁移：16个样本、31个待确认目标建议。全部pending，无实际人工签字；正式分类漏遮率均为null。另生成provisional-target-coverage.json仅作待复核区域定位，不是正式质量统计。
- 浏览器实际核验 `http://127.0.0.1:18767/yolo-images-001/index.html`：16个section，96/96个img均有有效naturalWidth，无失败图片；video数量0；三栏宽度584.328px×3；标题含待人工复核、状态32/32、范围明确16图。报告源码不加载外部脚本或媒体；宿主浏览器的杀软扩展注入与favicon请求不属于报告代码。
- 实际局部对照见run目录cases/：候选补遮图上边缘裁切脸、部分车牌覆盖更宽；也有右边缘人脸基线遮挡而候选未遮挡的风险，并有更大的颈肩遮挡。详情comparison.md，不宣称YOLO整体胜出。
- 最终回归命令同上节（tests/unit + comparison_media + model_probe + video_pipeline + stream_pipeline + runner_lifecycle）：**198 passed、1 skipped，28.14秒**，日志 `data/validation/yolo-setup/images-final-tests.txt`。Windows下POSIX父EOF用例跳过，未降级为通过。
- 生产/GPU/真实RTSP/Docker未测；全量视频按用户指令取消本轮执行，人工完整目标与质量复核未完成。图片交付已完成，不把这些边界算作图片32份产物失败。
