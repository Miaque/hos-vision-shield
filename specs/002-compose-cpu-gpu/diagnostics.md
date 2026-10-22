# Worker 资源观测与后续优化

## 本轮已实现的接口

`GET /v1/diagnostics/resources` 返回当前有有效租约的文件、流 worker，以及各 worker 最近完成的资源采样。采样在 worker 端执行，不将 API 容器资源当作视频处理资源；不改变既有任务响应和处理流程。没有新增依赖。

```powershell
Invoke-RestMethod http://localhost:18000/v1/diagnostics/resources | ConvertTo-Json -Depth 8
```

响应顶层是 `sampled_at_unix` 和 `workers`。每个 worker 包含 `worker_id`、`pool`、`age_seconds`、`stale`、`unavailable_reason`、`snapshot`。

| snapshot 字段 | 口径 |
| --- | --- |
| `sampled_at_unix` | Redis 服务器时间，Unix 秒；表示本次采样完成并准备上报的时间 |
| `sample_window_seconds` | 两次 CPU 计数读取之间的实际时间；首样本为 null |
| `cpu.percent` | 当前 worker 的 cgroup v2 累计 CPU 时间增量 / 采样时长 × 100；包含媒体子进程；100 表示使用一个逻辑核，可以超过 100 |
| `cpu.limit_cores` | 当前 cgroup 可见的 CPU quota 与有效 cpuset 的较小值；不是 CPU 利用率分母，不能证明不可见祖先没有更严格的限制 |
| `memory.used_bytes` / `limit_bytes` | worker cgroup 的内存占用及限制，字节；包含子进程及 cgroup 计入的缓存，不等同于 RSS；无限制时 limit 为 null |
| `gpu.devices` | 只查询 `GPU_DEVICE_ID` 指定的一张显卡，包含 UUID、名称、整卡 GPU 利用率、整卡显存占用/总量（字节）、功耗（瓦） |
| `*.unavailable_reason` | 无法采集的原因；缺失指标返回 null，不能解释为使用率 0 |

采样约每秒一次，GPU 查询有一秒超时，通过已有 Redis 保存 60 秒；采样结束和查询时间也计入实际周期。超过 5 秒的快照标为 `stale=true`。只返回租约有效的 worker，旧 worker 的快照不会作为当前实例展示；旧版 worker 或尚未上报时返回 `no_sample`。Redis 不可用时接口沿用 503，监测上报重试不阻断媒体任务。

CPU/内存目前支持 Linux cgroup v2。Windows、cgroup v1 或不可读取时返回 `cgroup_v2_unavailable`，不回退到宿主机总 CPU。GPU 需要容器可执行 `nvidia-smi` 并设置 `GPU_DEVICE_ID`，现有 GPU Compose 配置已经传入该值；命令缺失、超时、查询失败有不同原因码。CPU worker 不查询显卡，返回 `cpu_inference`。

GPU 指标明确为 `scope=device`，并非项目进程独占用量；同一显卡有其他服务时无法据此归因。文件和流 worker 若共享同一 UUID，会展示同一张卡，不得把两行显存或利用率相加。本轮不提供容器 PID 无法可靠对应到宿主 PID 的进程 GPU 利用率。

## 任务处理阶段计时

`GET /v1/tasks/{id}` 对新版 worker 处理的视频和流任务附带 `Server-Timing` 响应头；请求参数、JSON响应和原3个stats字段不变，图片及旧任务没有该头。使用[W3C格式](https://www.w3.org/TR/server-timing/)，`dur`单位为毫秒；这里是**当前媒体会话的累计处理耗时**，不是查询接口本身的耗时。

| 指标 | 口径 |
| --- | --- |
| `task-frame` | RGB转换、检测、跟踪和遮罩，以及构建输出帧的累计墙钟时间 |
| `task-inference` | 全图检测和实际局部复核的累计墙钟时间，是frame的子集；包含模型预处理/后处理，不是纯GPU算子时间 |
| `task-encode` | 视频编码及相应封装写入；文件还包括最终编码器冲刷及关闭容器 |

不要将3项相加：inference已包含在frame内。以`dur/processed_frames`估计每帧耗时；`elapsed_seconds*1000-frame-encode`还含解码、输入等待、事件上报和其他开销，流输入不足帧时等待也会占用该差值。模型加载、排队仍不计入elapsed；模型加载继续仅记录内部日志。重连创建新媒体会话后重新累计。

文件最终stats现在在MP4关闭后上报，计入此前遗漏的编码器冲刷/封装收尾，不能把计时口径变化称作处理提速。阶段数值随stats在原任务所有权/代次校验下原子更新，不返回源地址、凭据或模型路径。

## 部署

在服务器仓库更新源码，等当前任务完成后，在原项目根目录执行：

```bash
dc=(docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.gpu.yaml)
"${dc[@]}" build api file-worker
"${dc[@]}" up -d --no-deps --no-build --force-recreate api file-worker stream-worker
curl --fail http://localhost:18000/v1/diagnostics/resources
```

最后一行使用实际 API 映射端口；18000 是当前用户访问入口。API 属于 CPU 镜像，两个 GPU worker 共用 cu126 镜像，因此构建两个镜像、重启上述三个服务。本次没有变更依赖声明或锁文件；同一 builder 保留旧缓存时可复用依赖层。重新创建 worker 会中断在途任务，不能把它当作热加载。CPU 部署只使用 compose.yaml。

本轮验证：156 项单元测试通过，覆盖资源计数增量、CPU 配额、GPU 选择/缺失/超时、不支持的计数、陈旧及坏快照、Redis 上报重试、worker 退出回收。Windows 本机 GTX 1080 的真实 `nvidia-smi` 采集通过。尚未构建 Docker 镜像或在服务器验证 Linux cgroup 和 RTX 6000 Ada；接口上线后还需用同一视频记录空闲及处理期间的连续样本。

## 当前证据与后续顺序

服务端任务 `0c3657ab-f075-4a9e-ba53-80bf1516bd18`：源视频 1920×536、20 fps、2450 帧，处理 654.083 秒，平均 3.746 fps，无丢帧。源文件 40,520,742 字节，结果 53,045,149 字节，增加约 30.9%；视频码率约 2.65 → 3.46 Mbps。源与结果全部帧的尺寸和 PTS 已核对一致。视频与验证材料位于 Git 忽略目录 `data/validation/server-face-video-20261020/`。

用户标记 00:02–00:03 遮挡消失后重现，体积要求为“优先画质，体积尽量接近原片”。本地对原片 0–3.5 秒共 71 帧运行当前 CUDA 检测链，记录两路全图候选、最终框、保留框及场景变化量；证据位于 `data/validation/motion-dropout-20261020/`。它是本机模型重放，不能冒充服务器内部逐帧诊断。

1. **先修连续漏遮。** 现有 `FrameRedactor` 只在原位置保留最多两组真实框 200 毫秒，未做移动跟踪。原片中间门口人物在 2.0 秒补检分数约 0.303、2.05 秒约 0.451，2.65–2.95 秒出现超过 0.6 的补检候选并形成遮挡；3.0 秒又降至约 0.533，只剩历史补偿，3.2 秒历史已过期。2–3 秒缩略图变化量最大约 3.07，远低于清空阈值 20，没有该段因场景变化清空的证据。下一步使用此前已确认目标的运动信息跟随框，并对漏检目标做局部重新检测；设定失踪与跟踪失败边界。不能只放宽全局阈值或延长旧框寿命。验收要覆盖用户标记的整段及前后连续帧、原有误遮负例，并分别记录漏遮与额外遮挡。
2. **再控制编码体积。** 当前文件编码调用 libx264，未显式指定质量或码率策略，源视频的压缩参数不会自动沿用。用相同的已遮挡帧比较编码参数，按画质优先选择显式质量参数及合理码率预算，保持源分辨率、帧率和时间戳；不能把某个 CRF 或 maxrate 宣称为文件大小保证。NVENC 需要单独比较速度、画质与体积，启用硬件编码本身不保证文件更小。参数口径参见 [FFmpeg libx264 文档](https://www.ffmpeg.org/ffmpeg-codecs.html#libx264_002c-libx264rgb)。
3. **根据服务器分项计时提速。** 现在 EgoBlur 使用 CUDA，YuNet/HyperLPR 仍在 CPU；此尺寸每帧有全图及四个分块、共十次 CPU 补检模型调用，局部复核另计。这是待测的瓶颈候选。资源接口上线后补测解码/RGB 转换、两路全图检测、局部复核、遮挡和编码，区分并行重叠耗时。若补检成为关键路径，优先验证相同 ONNX 权重的 GPU 执行与分块批处理，先保留 FP32、预处理和阈值，再评估低精度。后端迁移必须通过既有质量集合及本次移动目标连续帧验证；没有这些数据不能承诺达到实时 20 fps。

上述三项处理行为尚未修改，本轮落地的是资源观测接口。GPU 利用率只是诊断数据，提速以相同素材的完成时长和质量验收为准。
