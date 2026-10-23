# P0 测量工具的本地验证

日期：2026-10-23。硬件为 **GTX 1080 8GB**，不是 RTX 6000 Ada。
本报告证明诊断工具能够观察固定版本，不构成路线筛选、双路速度或绝对质量通过。
本轮没有修改 `src/vision_shield`、main、旧实验或部署脚本。

## 实际检查

- `tests/unit`：349 passed，28.73 秒。
- `tests/integration/test_video_profile.py`：4 passed，5.50 秒。
  真实 FFmpeg/PyAV 静音、AAC、VFR 样例中，包装前后 MP4 逐字节一致；
  所有帧的有理数 PTS/尺寸、AAC 载荷/PTS/DTS 一致，编码包含最终 drain。
  注入检测失败后没有调用编码器，不放行原始失败帧。
- 汇总器改为增量累计，避免逐帧对全部历史样本重复求和；最后 11 项计时/编码定向检查
  通过，5.40 秒。正在运行的完整人脸诊断保留启动时工具快照
  `full-face-host-harness.py`（SHA256 `a5b9b07b352b994424c52e660aec0cb6a46ec28a18344d8c0897bee0bcb4cd88`），
  仍属仪表诊断，不据其 FPS 判断提速。
- GPU 短片：原人脸视频 8 秒处开始，12 帧、1920×536、无音频，
  FFmpeg libx264 CRF18 生成独立工具检查素材，**不是完整原视频验收**。
  短片 SHA256 `b3dfceb22fff9c0900aa4c45566505f1fda14566a0ac821972a5130bf4bfd5e5`。
- B 普通处理、两次 host 诊断、四组 device 诊断全部 12 帧、零丢帧，
  所有源有理数 PTS/尺寸保持，输出 SHA256 都是
  `2f9a55e6f2ecea1b136ca8facd56dac53170b26aaf927bc3ff2038328dd7eb6d`。
  实际 decoder `h264_nvdec`、encoder `h264_nvenc`，四份文件权重哈希符合清单；
  两个 EgoBlur 参数 FP32、detail 为 CUDAExecutionProvider，B 未导入 YOLO。
- A 使用精确检出 `581c175`、interval3/flow，在相同短片跑通并保留调用记录。
  本机 A 车牌参数为 FP32；Ada 才会选择 FP16，所以不能据此推导服务器 A/B 差距。
- 该短片人物距离远，不能凭文件相同证明清晰正脸/车牌的绝对质量；
  非空遮罩的真实 CPU 编码检查已通过，真实 GPU 中大目标及完整视频质量继续验证。

## 四模型真实调用与计时样例

取 `host-codecs-face/report.json`。全部是主机包含时间，含排队/隐式等待；
前十次之后样本仅 2–3 个，不足以声称稳定分位已收敛。

| 模型 | 实际调用数 | host P50 ms | host P95 ms | 首次 ms |
| --- | ---: | ---: | ---: | ---: |
| EgoBlur face | 13 | 129.457 | 225.917 | 344.123 |
| EgoBlur plate | 12 | 117.503 | 123.976 | 126.921 |
| YuNet | 12 | 39.289 | 95.520 | 161.585 |
| HyperLPR detection | 13 | 38.390 | 108.668 | 209.885 |

12 帧不等于每个模型只有 12 次：局部复核确实增加了主脸/补车牌调用。
JSONL 将逐次调用定位到源帧及用途；不再用检测轮数代替 forward 次数。

四组独立设备运行各采第 2/6 次真实调用，所有 trace 重算得到关联 CUDA 内核：

| 模型 | 两次内核数 | GPU 活跃并集 ms | 同步 API 主机时间 ms |
| --- | --- | --- | --- |
| EgoBlur face | 2110 / 2110 | 113.959 / 111.161 | 55.900 / 51.112 |
| EgoBlur plate | 2109 / 2087 | 111.591 / 111.177 | 62.218 / 63.156 |
| YuNet | 150 / 150 | 4.189 / 4.425 | 0.005 / 0.006 |
| HyperLPR | 434 / 434 | 3.107 / 3.788 | 0.036 / 0.038 |

这是两个样本的诊断区间，不是 GPU 稳态 P50/P95，更不是 RTX 成绩。
例如 YuNet 主机前向 21.288/37.732 ms，而关联内核与复制活跃并集只有 4.189/4.425 ms；
不能把全部 host 时间归为卷积计算，也不能仅用传输字节少推定没有等待。
EgoBlur trace 存在多个内部 CUDA stream，内核耗时之和与活跃并集不同，不直接相加。

Kineto 在部分工作线程采样时打印 `External init callback must run in same thread as registerClient`；
进程返回 0，八份 trace 均含关联内核且输出不变。保留此现象，服务器采样仍需独立复核。

## 失败记录及修复

首次 host 短片在生产事件构造时遇到 `candidate-detail` 重复关键字：
诊断新增的同名累计字段与原候选计时合并冲突。修复为 `snapshot()` 仅返回生产 `STAGES`，
额外诊断阶段只进入独立报告；补测试防止字段再次冲突，真实 GPU 重跑后通过。
失败目录 `host-face` 保留，未覆盖失败输出或把它计作完成。

## 证据与待办

本地大文件在忽略目录 `data/validation/route-b-p0-local/`，包括八份原始 trace、
`calls.jsonl`、各次 `report.json`、`pilot-integrity.json` 和 MP4。
本轮已启动 B 原始人脸完整文件的 host 诊断，完成后单独补记录；不把运行中任务视为完成。

仍须完成两条路线的 RTX 6000 统一 P0、三份原视频质量基线、资源与队列观测、
最终候选/固定对照至少三次双路整片，以及双路 30 分钟实时验收。
API 创建到完成的 FPS 在 CLI 报告为 null，不能用进程启动 FPS 冒充。
