# 待批准：默认交织与100ms的同步音视频对照

目的：阶段一线上观察存在“直接RTP时间戳递增，但PyAV接收端PTS回退”和PotPlayer缓冲。通过相同采集方式的默认/候选对照，确定是否是100ms新增退化，不修改任何新业务代码，不启动阶段二。

当前HTTP核对：17模块匹配`1c9be543c30fed7a2f4263b85c447ba079d438f8`，输出选项为100ms，目标`27c39007…`与背景`f78fb79c…`均在运行。完整现场证据见`stage1-acceptance.md`和`evidence/current-runtime-verification.json`。

以下两次部署均须用户明确授权本计划后执行；尚未执行：

1. 部署固定默认配置`77f98a96b2de803740aded5fa28dfab90a3cf932`，tag=`diag-20261029-interleave-a-restored`，同步采集原流/脱敏流的RTP视频、AAC AU、RTCP及PyAV解码约180秒。每组记录连接身份、内容哈希和负载；错误与重连单列。

```powershell
ssh -T gpu-server "cd /data/hos-vision-shield && sudo -n /bin/bash /data/hos-vision-shield/scripts/deploy-server.sh --tag diag-20261029-interleave-a-restored"
```

2. 核对第一步运行源码/参数/任务状态后，部署回同一个已验证候选`1c9be543c30fed7a2f4263b85c447ba079d438f8`，tag=`fix-20261029-rtsp-interleave-100ms-01`，以同样方式同步采集约180秒。

```powershell
ssh -T gpu-server "cd /data/hos-vision-shield && sudo -n /bin/bash /data/hos-vision-shield/scripts/deploy-server.sh --tag fix-20261029-rtsp-interleave-100ms-01"
```

两次均会重建API/worker并短时中断现有流。第一步暂时恢复原有较严重交织停顿，第二步是计划内恢复100ms候选；目标及背景任务保留原ID自动恢复。不会停止任务、改变权限、修改外部源、关闭音频、重写标签或合并main。

每次覆盖前重新HTTP核对当前指纹、实际选项及运行/排队任务。若实际运行任务范围发生变化、版本不符或出现未预期错误，先报告并停止后续部署，不增加其他命令或自动扩大回滚范围。用户批准本计划只覆盖上述两条固定命令及顺序，不构成后续部署授权。

观察成功条件：两组均获得同类输入/输出时钟和内容匹配证据，能够判断异常是否只在候选出现；不能仅以没有异常或音频包数相近宣称音画同步正确。若仍无法区分，报告不足，不立刻修改其他交织上限或架构。
