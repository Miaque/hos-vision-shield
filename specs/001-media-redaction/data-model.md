# Data Model

## Task（Redis hash，task:{UUID}）

| 字段 | 类型/约束 | 用途 |
|---|---|---|
| id | UUID，由服务生成 | 公共任务标识，不等于 Taskiq message ID |
| kind | image / video / stream | 决定处理池及状态集合 |
| state | 下述状态枚举 | 当前已确认状态 |
| desired_state | run / cancelled / stopped | 持久用户意图；文件不能设 stopped，流不能设 cancelled |
| created_at / updated_at | 内部 UTC ISO 8601；API 输出 UTC+8 `%Y-%m-%d %H:%M:%S` | 可追溯时间 |
| source_kind | upload / url / rtsp | 输入方式 |
| source_ref | 内部路径或地址，不返回客户端 | URL凭据仅保留内部，日志必须脱敏 |
| result_ref | 可空，任务内相对路径 | 仅成功文件可下载 |
| playback_path | 流为 redacted/{id}，其他为空 | 稳定逻辑播放路径 |
| ready | bool | 已确认可播放或可下载，不等于有地址 |
| owner_id / generation | 可空标识 / 单调递增整数 | 执行者与所有权代次 |
| lease_until | Redis服务时间毫秒 | 心跳有效性，禁止各worker独立时钟判断 |
| dispatch_pending | bool | 持久待投递标记 |
| error | 可空 code/message | 无堆栈、无源凭据 |
| stats | 有界计数对象 | 处理帧数、丢帧数、处理时长，不保存每帧历史 |
| publish_allowed / session_id | 内部字段 | 流发布接管及撤销 |

任务 ID 唯一；Redis原子脚本负责创建准入、claim、续租、停止与条件提交。状态不设自动TTL；容量不足拒绝新任务，不驱逐旧任务。
用户提交两次相同来源会创建两个任务；队列重投同一 ID 才按幂等处理，不按 URL 擅自合并。

## 文件与结果

每个任务独立目录：data/{id}/source.ext、attempt-{generation}/、result.ext。
输入名由服务生成，用户文件名不参与路径解析。写入先使用 .part，完成后改名；未完成产物不能通过结果接口读取。
文件名和任务对应关系存Redis；多机worker与API须访问同一共享卷。原文件、结果和失败残留暂不自动删除。
只读模型目录包含 face.jit、license_plate.jit 和来源、版本、SHA-256记录；仅部署方提供，不能由请求指定任意模型路径。

## 文件状态转换

- queued → processing → succeeded / failed。
- queued + cancelled 意图 → cancelled；processing + cancelled 意图在退出并回收资源后 → cancelled。
- 取消处理中保持 processing，但 desired_state=cancelled；不得提前声称资源已释放。
- 有效owner失去租约 → failed，error=worker_lost，不自动从中间视频续跑。
- succeeded/failed/cancelled 为终态；重复取消已取消任务返回现有状态，取消其他终态返回409。
- 结果发布先准备独立结果文件，再以 owner+generation+desired_state 条件提交；失败的提交不能改变可下载路径。

## 流状态转换

- queued → connecting → running；ready 仅在发布端就绪且产生处理后数据时为true。
- connecting/running → recovering（来源超时、发布失败、owner丢失），成功重连 → running。
- 任何非终态 + stopped 意图 → 清除发布资格、撤销会话、终止执行 → stopped。
- 停止收敛前保留原状态，desired_state=stopped、ready=false；不能确认旧连接结束时不得返回stopped。
- 不支持的媒体或配置不可恢复 → failed；暂时源不可达持续有界退避，直到停止。
- failed/stopped 不自动恢复；需要重新提交创建新任务，首版不提供重启接口。

## 调度与所有权不变量

- 待调度任务索引和更新任务的Redis事务一致，避免写任务后未投递永久遗失。
- claim 必须同时检查非终态、run意图及无有效owner，原子增加generation并登记租约。
- 旧generation的心跳、结果、状态变更一律无效；异常释放不能删除新owner的租约。
- 先登记所有权再ACK；进程崩溃后reconciler根据意图恢复流。重复消息只有一个claim成功。
- 媒体侧单发布者及撤销协议详见plan.md；Redis CAS本身不保证RTSP断开。
- 队列项通过 Lua 原子 XACK + XDEL 确认并删除；未确认项不得被长度裁剪。每个任务最多一个未完成投递意图，控制重复投递的物理积压。
- 文件终态/流终态（停止或失败）时释放未结束任务容量计数，但保留历史记录。计数修复应由索引校验完成，不靠盲目减一。
