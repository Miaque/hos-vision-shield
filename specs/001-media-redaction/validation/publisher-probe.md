# T008 发布接管局部验证

环境：Windows、MediaMTX1.21.2原生程序，独立HTTP回调服务器和临时RTSP/控制端口。
命令：`uv run pytest tests/integration/test_publisher_probe.py -q`，结果2 passed。

已验证：

1. ANNOUNCE已完成、SETUP/RECORD未开始时，回调载荷中的session ID可由Control API撤销；旧socket随后关闭，旧generation重连返回401，新generation可ANNOUNCE。
2. HTTP校验已经决定批准、尚未发送回复时，可按已记录session ID撤销。释放回调后可能收到在途ANNOUNCE 200，但随后连接必须EOF，旧端无法进入后续SETUP/RECORD。

原始断言将“收到ANNOUNCE 200”等同于“撤销失败”，首轮1 passed/1 failed；修正探针为检查连接最终关闭，
没有将任何仍能发布的连接视为通过，也没有以当前path为空或404作为成功依据。

这两项只证明连接建立窗口中的API行为。尚未验证：运行中持续RTP流、真正worker与Redis网络分区、
持久租约/代次协调、停止和恢复组合各3次、Linux容器环境。当时T008保持未完成；下方补充后续原语验证。不能以技术探针宣称应用FR-010已实现。

## 本机后续验证

新增真实FFmpeg持续RTSP发布及独立Redis控制租约，共3次：停止续租后租约过期但旧媒体进程仍连接，撤销旧session后旧进程退出，旧代次重连401，新代次可发布，主动撤销后新代次也不能重连。探针共5项通过，T008媒体端原语门槛通过。
这里验证的是媒体会话撤销、代次校验及租约丢失时的数据面隔离；完整应用worker/reconciler的真实故障组合仍由T037验证，不将探针通过表述为应用已完成。Linux容器已按用户要求移出范围。
