# P0 A：局部人脸诊断候选

## 变更

模型、裁剪、缩放、阈值和复核工作量保持原样。新增内容仅在独立的显式 profiling 进程启用：

- `host`：墙钟计时，包含复核请求/父请求、帧、目标框、首次 ROI 缩放前后尺寸、模型输入 shape/dtype、会话/provider，以及首次观察的局部 shape 和局部 shape 切换。
- `local-face`：对实际 NumPy 局部 ORT 调用打开 CUDA 追踪，仅保存前面遇到的至多三次 >500ms 调用；其余 trace 不落盘。槽位 64/424/768 与真实调用序号分别记录。
- 请求关联号不是持久轨迹 ID；现有轨迹没有这样的 ID，不伪造跨帧身份。
- 记录下载、NumPy、resize、既有准备/后处理及 ORT 阶段；下载墙钟含同步。完整 trace 用于区分 CUDA 计算、同步、分配和传输，不能用 host time 直接称 GPU 计算时间。
- 各组补充 P99/max 与 >500ms/>1000ms 计数；摘要最多保留八条最长局部调用。
- 快速调用也要先跟踪才能知道它是否慢。因此此模式会扰动耗时，只用于单独采样实验。普通 `host` 不开设备 profiler，普通业务路径不安装这些观察器。

CLI 不连接 Redis/API，不创建业务任务：

```powershell
python scripts/profile-local-clip.py <固定原始MP4> <新证据目录> --models <已有模型目录> --mode host
python scripts/profile-local-clip.py <同一MP4> <另一个新证据目录> --models <同一模型目录> --mode local-face
```

API 沿用视频上传的 `X-Detection-Profile`，新增 `local-face` 枚举。仍仅 DEBUG 可用、仅视频、仅 quality_yolo/CUDA，任务目录和 trace 大小保护不变。

## 本机验证（2026-10-28）

固定原始短片为 440 帧、960×268、无音频。SHA256：`5b60318c58c8530502bda305f89de1888417b2f701a40dc949bcae4a233d562c`。硬件为 GTX 1080，不是 RTX 6000 Ada。

- 原始 `host` 与新增 `host`、新增 `local-face` 三份输出 MP4 编码字节完全一致；完整 440 帧、尺寸及递增 PTS 校验通过。
- forward 次数均为：主人脸 1174、主车牌 152、补人脸 925、补车牌 452，没有通过减少模型工作量改善指标。
- 925 次补人脸调用中的 45 次走当前目标 NumPy/动态 ORT 局部路径，其余主要为 Tensor/固定执行路径。
- 新 `host` 的 45 次局部 ORT：P50 3.217ms，P95 36.230ms，P99 42.081ms，max 44.438ms，>500ms 与 >1000ms 均为 0。
- `local-face` 追踪同样 45 次实际局部调用，没有保留 >500ms trace，明确返回 `partial`，不当作捕获 RTX 长尾成功。
- 真实 CUDA 校准把保留阈值在测试进程内临时设为 0，已验证采到实际 NumPy 输入、CUDA provider、HtoD 与相关 kernel；该测试不证明秒级问题已复现。
- 849 项 unit 通过、96 项既有 GPU 用例按原条件跳过；另显式 CUDA 校准 1 项通过，静音/AAC/VFR/推理失败关闭等真实媒体 4 项通过。

两轮诊断运行时间存在较大波动。当前没有算法优化，不能把时间差当吞吐收益，也不能据本机无秒级样本宣布 RTX 已修复。

## 当前决策与后续边界

**不预选 P0 B 的技术方案。** 现有 RTX 证据能定位同步局部复核长尾，不能区分动态 ORT 内部的最终机制。本机没有复现该秒级长尾，故先交付能捕获慢局部调用的诊断候选。

下一步需要将这份诊断候选在 RTX 上作为独立实验运行同一短片，不能替换生产服务。应使用临时 GPU 容器、固定源码/原片、只读模型和独立证据目录，分开运行 host 与 local-face。额外诊断负载会短时共享 GPU，其 FPS 不算正式验收。

此实验需要新的服务器执行授权：现有唯一远程部署命令不能用于部署候选分支或启动临时诊断容器，禁止借修改原部署脚本实现。未授权前不执行新 SSH/SCP/Docker 操作。

P0 B 尚未修改；P1/P2 按用户规定等待 P0 独立验证，P3 及 RTX 30 分钟双流验收尚未执行。没有合并 main 或改服务器配置/部署。

## 2026-10-29 后续授权

以上是 10 月 28 日交付时的边界。10 月 29 日用户要求合入当前 main（已完成，`d6b1eb5`），随后明确授权部署相关指令及通过 `http://127.0.0.1:18000` 继续服务测试。本轮改为使用现有部署脚本部署本分支，再由业务上传 API 运行隔离文件诊断；不启动临时容器、不执行其他服务器探测命令。部署会重建应用容器并让已有实时任务恢复，诊断文件任务与已有流共享 GPU，须分别报告。

## ORT 节点采样

`local-face` 独立诊断额外建立一份相同模型、会话选项和 CUDA provider 选项的局部动态会话，只开启 ORT 原生 profiling。最多记录前 64 次局部调用，结束后检查 32 MiB 文件预算，按实际 `run` 序号把采样调用关联到 ORT 的 `model_run`，每次最多返回 8 个最慢算子。超过 64 次后出现的 CUDA 样本可能没有节点详情；`host` 和普通处理不启用此机制。

这是算子的主机墙钟时间，包含准备、提交和等待，不能当作纯 GPU 时间。API 既有 trace 仍为 CUDA 事件，节点摘要位于 `device_samples[].ort_top_nodes`。依据 [ORT profiling 文档](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)，profiling 必须在创建会话前启用，因此诊断会话保持新建；不把额外采样实验的 FPS 当性能收益。
