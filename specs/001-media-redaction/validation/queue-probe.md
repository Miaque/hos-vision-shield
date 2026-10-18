# T006 队列探针

环境：Windows、Python3.12.14、Redis7.4.3独立临时进程、Taskiq0.12.4、taskiq-redis1.2.2。
命令：配置TEST_REDIS_SERVER后 `uv run pytest tests/integration/test_taskiq_probe.py -q`。
结果：2 passed。

验证内容：

- 原子限长入队拒绝超限，未确认消息仍在PEL，不被裁剪。
- 原子XACK+XDEL同时清除PEL与物理entry。
- 真实Taskiq worker在协程仍运行时提前确认消息，单活动任务、预取0及重复逻辑任务去重均通过。

兼容性发现：Taskiq0.13.0移除了taskiq.compat.model_dump，taskiq-redis1.2.2在导入调度模块时失败。
已固定0.12.4/1.2.2，全部依赖导入通过。0.12.4没有Context.ack，因此探针在broker.listen保存消息的ACK回调，
由任务显式触发；执行完成时再次自动ACK为幂等操作。该适配已迁入src/vision_shield/jobs/broker.py，探针继承正式broker，避免重复实现。
不能宣称0.13的Context.ack已验证，也未修改或猴子补丁第三方依赖。

本机Windows原生进程验证；不证明Redis持久化文件丢失后的恢复能力。
