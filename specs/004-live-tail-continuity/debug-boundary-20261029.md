# DEBUG 诊断边界检查与修复

## 检查结论

检查基线为 `main / 92b7463`。此前并非完全满足要求：API、模型/逐帧计时和资源采样已有 DEBUG 门禁，但详细流连接历史、异常栈、FFmpeg 来源打开时的日志捕获及连接结束诊断日志仍有无条件路径；本地 profiling 脚本还写死 `debug=True`。

本轮只修复诊断边界，不开发 P2，不改变检测、复核、遮罩、时钟修复、P0 算法选择或 P1 发布节奏。正常任务进度、安全错误码、必要错误告警、输出健康、重连和停止控制属于业务功能，不能为关闭调试而禁用。

## 实际入口与关闭行为

| 路径 | DEBUG=false 行为 |
| --- | --- |
| `/v1/diagnostics/resources`、任务 diagnostics | 404，在读取诊断数据前拒绝 |
| 上传的全部 7 种 profiling 请求头 | 422 diagnostics_disabled，不创建任务或媒体文件 |
| 子进程中的残留 profiling 参数 | 忽略，不建立 host/CUDA/ORT profiler 或采样文件 |
| 模型分项计时、逐帧读取/解码/处理时刻、VCL 哈希 | 不采集，StageTimes 及详细计时不启用 |
| StreamMetrics | 不创建诊断 attempt ID 或读取采样时钟；仍保留重复 PTS、编码积压上限等保护 |
| worker CPU/GPU/显存资源采样 | 不启动采样协程，不调用 nvidia-smi |
| FFmpeg 来源打开协议日志 Capture | 不启用；错误仍返回安全类型、阶段和 errno |
| 异常调用栈 | 不调用 extract_tb、不发送 failure_at；模型故障分类所需的业务判断仍执行 |
| 详细连接历史、media_failure、连接结束诊断日志 | 不采集或持久化，不输出详细栈；正常重连不受影响 |
| worker 收到旧/异常的调试事件 | 不存 timings、profile、编解码信息、检测计数或 stats.stream，仅保留普通进度 |
| `scripts/profile-local-clip.py` | 从进程环境读取 DEBUG；未启用时，在读取原片或创建输出目录前报错退出 |

DEBUG=true 时，既有诊断保持可用；host/CUDA/ORT profiling 还需要显式请求相应模式，不因开关开启而自动对所有任务进行高开销追踪。API 与全部 worker 都必须应用同一配置并重启/重新创建容器。关闭不会删除此前已经生成的诊断文件或历史记录；它们仍被 API 的 DEBUG 门禁隐藏。

## 验证与限制

新增回归先复现了未开启 DEBUG 仍执行本地 profiling、生产仍保存连接历史/日志及采集异常栈的问题，修复后通过。覆盖所有 profiling 模式、DEBUG true/false、普通/常驻进程、生产重连、停止时输出进程关闭顺序及编码安全保护。原有详细诊断测试改为显式开启 DEBUG，并保留全部断言。

最终完整单元检查：886 通过、96 项既有条件用例跳过、9 项既有 Torch 警告；日志为本地证据目录的 `debug-boundary-unit-final-20261029.log`。文件任务和实时任务都验证 DEBUG=false 不存子进程传回的详细调试字段，DEBUG=true 继续保留这些字段；没有以删除业务断言或跳过失败用例换取通过。

本轮使用本地测试，未修改服务器配置、重启服务或执行新的 GPU 采样；源端部署需要更新修复版本才具有新增门禁。P0/P1 仍会在生产模式生效，P2 保持暂停。

## 人脸长尾原片

未脱敏原片为：`C:/Users/zhanjh/.codex/visualizations/2026/10/28/01a18121-3312-7923-8d87-5418064fa5b0/g3-original-diagnostic.mp4`。

SHA256：`5b60318c58c8530502bda305f89de1888417b2f701a40dc949bcae4a233d562c`。文件 1,902,798 字节，440 帧、960×268、约 29.15 秒，无音频，平均约 15.09 FPS。原慢调用位于约 22.086 秒、第 334 帧。模拟实时流时保留原 PTS，不使用补帧或改帧率来宣称达到 25–30 FPS。
