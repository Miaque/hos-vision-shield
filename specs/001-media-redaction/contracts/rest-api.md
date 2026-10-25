# REST API Contract — v1

首版内部网络，无登录、API Key、用户或角色；错误格式统一，JSON字段使用snake_case。
任务接口为HTTP；播放为RTSP。上传完成并持久接收后才返回202，202不保证结果已生成。

## Endpoints

| 方法与路径 | 请求 | 成功返回 |
|---|---|---|
| POST /v1/file-tasks | JSON FileUrlInput | 202 Task，Location=/v1/tasks/{id} |
| POST /v1/file-tasks/uploads | multipart UploadInput | 202 Task，同上 |
| POST /v1/stream-tasks | JSON StreamInput | 202 Task，同上 |
| GET /v1/tasks/{id} | UUID | 200 Task |
| PATCH /v1/tasks/{id} | JSON DesiredStateInput | 202 Task；已达到相同终态则200 |
| GET /v1/tasks/{id}/result | UUID，仅成功文件任务 | 200 image/jpeg、image/png或video/mp4，附件下载 |
| GET /health/live | 无 | 200 {status: alive} |
| GET /health/ready | 无 | 200 {status: ready}；依赖不可用503 |

不提供列表、删除、重启、模型选择或目标类型选择接口。
POST创建不承诺幂等：超时后盲目重试可能创建新任务；拿到ID后必须查询现有任务。队列重投同一ID则幂等。
PATCH重复设置同一意图幂等，不允许重新设run，不能以停止接口隐含删除文件。

## Request Schemas

- **FileUrlInput**：kind必填枚举image/video；url必填HTTP或HTTPS URL，最长4096字符；禁止其他字段。
- **UploadInput**：kind必填枚举image/video；file必填且恰好一个文件；不接受URL字段或服务器路径。
- **StreamInput**：url必填RTSP或RTMP URL，最长4096字符；可选publish_url指定外部RTSP/RTMP输出；可选playback_url为完整HTTP/HTTPS/RTSP/RTMP播放地址，最长4096字符，不由服务端访问；可选max_bitrate为正整数bit/s实时编码上限；省略max_bitrate时不设置码率上限；允许摄像头URL凭据但不得回显。RTMP输入沿用来源允许列表及DNS固定，tcUrl保留原厂域名。
- **DesiredStateInput**：desired_state必填cancelled/stopped；文件只允许cancelled，流只允许stopped；其他值422。

kind声明必须与实际解码格式一致。图片JPEG/PNG；文件MP4/H.264；流H.264；音频仅AAC或无音轨。
按部署限制检查字节数、像素、时长与允许目标；HTTP重定向至多3次，每次重新验证，禁止降级到非HTTP(S)。
API可提前拒绝协议或已知格式错误；需探测后才能发现的错误表现为任务failed，不追改此前202。

## Task Response

所有字段均出现，可空项为null：

| 字段 | 类型 |
|---|---|
| id | UUID string |
| kind | image / video / stream |
| state | queued / processing / succeeded / failed / cancelled（文件）；queued / connecting / running / recovering / stopped / failed（流） |
| desired_state | run / cancelled / stopped |
| ready | boolean |
| created_at / updated_at | UTC+8 字符串，格式 `%Y-%m-%d %H:%M:%S`，例如 `2026-10-22 09:00:00` |
| result_url | string或null；仅成功文件任务非空 |
| playback_url | URL或null；流优先使用请求的playback_url，否则沿用完整publish_url，默认输出使用公开RTSP地址；ready=false时不可承诺可播放 |
| error | null或{code: string, message: string} |
| stats | {processed_frames: nonnegative integer, dropped_frames: nonnegative integer, elapsed_seconds: nonnegative number} |

流stats描述当前媒体会话，重连后重置，不代表跨故障累计的所有源帧；elapsed_seconds不含模型加载及排队。

所有任务创建、详情、列表及停止接口的日期字符串均使用上述 UTC+8 格式，不含 `T`、时区后缀或小数秒。历史 UTC 记录在响应时转换，内部存储仍为 UTC。诊断接口顶层及 worker 快照的 `sampled_at_unix` 保持 Unix 秒数；耗时、年龄、采样窗口等数值字段保持秒数语义。当前接口没有日期时间入参。

新版视频/流任务的`GET /v1/tasks/{id}`可附带`Server-Timing`响应头，JSON和stats字段保持不变。`task-frame`、`task-inference`、`task-encode`为当前媒体会话累计毫秒，inference是frame的子集；旧任务/图片可不返回。组合后端还可提供两项重叠的候选调用计时`task-candidate-primary`/`task-candidate-detail`，同样为累计毫秒，不可相加。文件最终elapsed包括编码冲刷及关闭MP4。详细口径见[任务阶段计时](../../002-compose-cpu-gpu/diagnostics.md#任务处理阶段计时)。

实际视频编码事件已上报的任务还可附带`X-Video-Encoder: libx264`或`h264_nvenc`；旧任务和未上报路径不返回，不改变请求参数或JSON。该头不是硬件编解码全链路的证明。

文件任务还可附带`X-Video-Decoder: h264_nvdec`或`h264`，表示实际硬件帧或原软件解码路径；未开始处理及旧任务可省略，不改变请求或JSON。兼容格式的CUDA文件使用NVDEC，其他格式保留原解码方式；选中硬解后失败不重试CPU，详见[诊断说明](../../002-compose-cpu-gpu/diagnostics.md#任务处理阶段计时)。
上述诊断响应头仅在 `DEBUG=true` 时返回；默认生产环境不采样，也不返回旧任务的 `stats.stream` 诊断样本。原始来源、绝对磁盘路径、worker标识不进入响应。外部推流未指定playback_url时，返回的播放地址包含publish_url原有的凭据及查询参数；不应将其当作匿名播放地址。没有进度百分比，避免长流或未知时长的伪进度。

```json
{
  "id": "e5176342-68c1-4c5c-a43e-39c5d220cc8a",
  "kind": "stream",
  "state": "queued",
  "desired_state": "run",
  "ready": false,
  "created_at": "2026-10-18 16:00:00",
  "updated_at": "2026-10-18 16:00:00",
  "result_url": null,
  "playback_url": "rtsp://media.internal:8554/redacted/e5176342-68c1-4c5c-a43e-39c5d220cc8a",
  "error": null,
  "stats": {"processed_frames": 0, "dropped_frames": 0, "elapsed_seconds": 0}
}
```

## Errors

统一响应：{"error":{"code":"capacity_exceeded","message":"Task capacity is full"}}。
不暴露内部异常、源码路径、源地址或凭据；参数校验错误也转换为同一外层结构。

| HTTP | code示例 | 场景 |
|---|---|---|
| 404 | task_not_found | ID不存在 |
| 409 | result_not_ready / invalid_transition | 非成功文件取结果、流取文件、终态不允许操作 |
| 413 | media_too_large | 已知或读取后超出大小上限 |
| 408 | request_timeout | 请求接收超过时间限制 |
| 415 | unsupported_media | API已能确定不支持的媒体 |
| 422 | invalid_input / source_not_allowed | 结构、协议、目标或类型错误 |
| 429 | capacity_exceeded | 未结束任务容量上限；带Retry-After |
| 503 | dependency_unavailable | 状态存储、可用worker或其他必需依赖不可用 |
| 500 | internal_error | 未预期错误，日志按task_id定位 |

后台错误码包括source_unreachable、source_not_allowed、media_too_large、storage_full、worker_lost、processing_failed。解码、模型或编码异常首版统一为processing_failed，日志仅记录异常类型和代码位置，不回显异常文本中的来源凭据。
流暂时中断时state=recovering并携带简短原因；恢复后error=null。

## Internal Media Contract

仅服务网络访问 POST /internal/media/publish-check，接收MediaMTX官方HTTP验证载荷。
发布回调核验task/generation/owner/lease及publish_allowed，成功登记session并返回204，否则非2xx；读流无需凭据。
该检查只用于任务所有权隔离，不是新增用户权限管理。Control API也只在服务网络开放。
停止接口仅写意图；stopped状态必须在会话撤销和资源释放确认后出现，控制依赖不可用时不得虚报完成。

## 每任务遮挡样式

三个创建入口均支持可选 `redaction_mode`，仅允许 `solid`（默认，不透明深灰色矩形 RGB 64/64/64）或 `mosaic`（马赛克）。URL文件和RTSP创建使用JSON字段；上传使用同名multipart文本字段。非法值返回422，上传重复字段同样拒绝。值持久化到任务记录并随响应返回，流恢复/重连保持原值；历史缺少字段的任务使用solid。不提供环境变量控制或运行中切换。
