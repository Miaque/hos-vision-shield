# 固定 A 的独立诊断包装

固定算法参考仍为 `581c175a0fad21efe10a0caf08a01e37b25fd3ad`，普通吞吐使用
`exp-20261023-a-baseline`（`940e368`）。本分支仅移入 B 已验证的显式诊断观察器和
API/任务传递，不修改检测器、降频/光流调度、遮罩、视频处理器、Compose、权重及依赖。
普通请求不安装观察器；诊断 FPS 不作吞吐验收。

服务器以 `quality_yolo` / CUDA 运行，两路上传都明确请求
`X-Detection-Interval: 3`、`X-Tracking-Mode: flow`，不依赖默认参数。
该配置仍逐帧运行人脸/细节候选；主车牌有周期及复核调度，不能称为四模型严格每三帧一次。
Ada 的主车牌使用固定参考已有的完整图 FP16；本机 GTX 1080 保持原 FP32。

显式 `X-Detection-Profile: host` 汇总真实 forward 次数、全图/分块/局部/跟踪复核分布、
host 累计及 P50/P95、首次/十次后分布。`forward-primary-face`、`forward-primary-plate`、
`forward-detail-face`、`forward-detail-plate` 分别在该模型第 64/424/768 次真实前向采集 CUDA trace，
每份不超过 32 MiB，不补跑模型。样本同时保存源帧、局部/全图视图和源区域等上下文；
主 YOLO 的 provider、实际 session 输入类型与权重哈希单独报告，PyTorch 模型列表的人脸项为 null，不伪报其 dtype。
其他参数精度、设备、权重和核心执行源码 SHA256 均随报告保存。

通过原任务 diagnostics 接口取得报告及 `?sample=64` 等原始 trace；路径限制在该任务 attempt。
未满三个采样点标为 partial，预算超出或诊断失败中止任务，不发布原帧或伪成功结果。
模型加载段不包含预先导入的推理依赖，完整冷启动仍应从任务创建至完成边界观察。
嵌套和并发 host 时间不可直接相加。

本地验证：136 项定向单元/真实编解码检查及 56 项降频/光流/稳定性检查通过；非空 12 帧 GPU 短片普通/host/不足采样点的
指定模型模式逐字节一致（SHA256 `d4bf200ceeb45455ce91e56c995a0c63189420cbf5e3b45305bc74360192894d`），
全部 PTS/尺寸保持。真实 YOLO 第 2 次局部前向另外采到 359 个关联内核、GPU busy 2.894 ms，
样本上下文为第 1 帧、320×320 局部确认，输出仍逐字节一致；这是 GTX 组件证据，不是 Ada 吞吐。
证据位于本 worktree `data/validation/a-child-profile-pilot-v3/`、`a-device-pilot/`。

首个本地 harness 关闭控制管道导致子进程正常执行撤销保护，已修正 harness 保持管道；
随后 host 元数据发现 CUDA helper 尚未导入，已在读取实际源码指纹前加载该无推理副作用模块。
两次失败记录保留，未计为完成；修复后重新执行以上真实 GPU 验证。

尚未取得本包装的 RTX 单/双路诊断，也未完成固定 A 的三次普通双路质量/速度和实时验收。
部署须等正在进行的 B 测试结束，再用已授权的同一 tag 部署命令顺序切换，不并发混测。

## 共用 CUDA 归因修复

B 的真实 RTX 第三份主人脸 trace 暴露 CUPTI 在重建线程后沿用旧 tid，已将共用汇总器
更新为按父范围/CPU 算子的 External id 关联；没有该字段的 ORT 调用保持线程及时间限制。
回归同时验证不混入并行模型、父范围外算子或清理事件，保留无关联内核即失败的保护。
A 的检测和跟踪逻辑继续固定；此变更仅修正独立诊断统计。
