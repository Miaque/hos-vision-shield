# 实现与前置验证

2026-10-18，Windows本机原生进程；不依赖Docker/WSL2。四项技术门槛均通过并迁入应用，没有用跳过或假模型替代。

| 门槛 | 当前证据 |
|---|---|
| T001 依赖 | Python3.12.14，uv sync/uv lock --check，57个锁定依赖 |
| T002 本机运行 | 基础脚本-Check校验Redis健康、AOF always/noeviction及MediaMTX；应用脚本-Check -StreamWorkers 2校验API及两个worker池，结束后仅回收自身进程 |
| T003 配置 | 16项容量、超时、白名单与服务地址校验 |
| T004 样本 | 官方EgoBlur人脸/OpenCV车牌及哈希；合成JPEG/PNG、AAC/无音轨/VFR/B帧/旋转视频、HTTP/RTSP源 |
| T005 模型 | model-probe.md，两个官方权重真实CPU推理 |
| T006 队列 | queue-probe.md，真实Taskiq/Redis，应用直接复用探针broker |
| T007 媒体 | media-probe.md，逐帧PTS、AAC包相等与丢帧时间线 |
| T008 发布隔离 | publisher-probe.md，回调建立窗口、撤销session、旧代次拒绝、活连接失租 |
| T009 业务门槛 | 全部通过后实现公共基础与三个业务链路 |

完整应用验证见acceptance.md。历史缺模型及首次AAC缓存失败已定位修复，不将失败运行记作通过。
人工质量复核仍需人工完成，不能从模型加载、黑色像素检查或合成媒体推导准确率。
