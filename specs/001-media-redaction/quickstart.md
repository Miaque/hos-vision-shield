# 本机运行与验证

按用户要求使用Windows本机原生进程，不要求Docker/WSL2。业务接口和三个媒体处理链路均已实现。

## 环境

Python3.12、uv、Redis7.4.3、MediaMTX1.21.2和FFmpeg/ffprobe。两个模型固定官方来源和revision，见models/README.md。
配置项、端口及本机二进制位置见仓库README和.env.example。

```powershell
uv sync --frozen
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\scripts\start-infrastructure.ps1 -RedisExecutable 'H:\software\Redis-7.4.3\redis-server.exe' -MediaMtxExecutable 'H:\workspace\python\vision-shield\.tools\mediamtx\mediamtx.exe' -Check
$env:TEST_REDIS_SERVER='H:\software\Redis-7.4.3\redis-server.exe'
uv run pytest -q
```

去掉-Check保持基础服务运行；第二个终端执行 `.\scripts\start-application.ps1`，然后访问 http://127.0.0.1:8000/docs 。两个终端分别保留，Ctrl+C停止各自创建的进程。

本机基础服务与应用脚本的-Check已实际运行，通过Redis持久化配置、MediaMTX控制接口和两个worker池的健康检查；应用脚本也以-StreamWorkers 2验证启动。

图片上传示例：

```powershell
curl.exe -F 'kind=image' -F 'file=@samples/official-demo.jpg' http://127.0.0.1:8000/v1/file-tasks/uploads
```

使用返回的id查询 `/v1/tasks/{id}`；成功后GET `/v1/tasks/{id}/result`。有限视频同一上传端点，kind=video。URL文件使用POST `/v1/file-tasks`，JSON形如 `{"kind":"image","url":"http://授权地址/image.jpg"}`。

RTSP使用POST `/v1/stream-tasks`，JSON形如 `{"url":"rtsp://授权地址/live"}`。查询ready=true后播放playback_url。PATCH `/v1/tasks/{id}`，文件设 `{"desired_state":"cancelled"}`，流设 `{"desired_state":"stopped"}`；202是请求已保存，终态是清理完成。

远程来源必须在.env的SOURCE_ALLOWLIST中显式授权（默认拒绝全部远程输入）；内网域名还需授权其解析IP/CIDR。修改.env后重启应用。默认仅监听本机，内网接入的监听及防火墙配置见README。

## 用户场景验收

- 图片：上传和远程地址分别提交JPEG/PNG，查询任务并下载，与保留原图对照；无目标、坏文件和取消必须可区分。
- 有限视频：带AAC、无音轨、B帧、VFR、旋转；核对完整帧顺序、时间戳、音轨与取消结果，不因负载主动丢帧。
- RTSP：其他系统播放输出；过载丢未处理旧帧但不透传原图，AAC缓冲有界；源中断、worker退出及停止竞争各3次。
- 发布隔离：旧worker失去控制租约但媒体连接仍活着时，撤销旧session再开放新代次；额外覆盖回调已批准但连接尚未建立的窗口。
- 不可用依赖不得触发虚假成功；停止需在会话和活动资源释放后才报告完成。原文件与结果按要求保留。

接口与状态以contracts/rest-api.md和data-model.md为准。
最终需生成性能与人工检测复核报告；没有承诺固定并发数或生产准确率。
