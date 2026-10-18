# 用户.env环境核对（2026-10-18）

只读检查确认.env加载、两个模型存在、Redis可PING、MediaMTX控制接口200；没有在日志中输出密码或完整源地址，也没有清空用户Redis。
Redis版本6.2.21，COMMAND INFO确认XAUTOCLAIM/XACK/XDEL/EVAL/TIME存在。MediaMTX为HTTP发布回调、预期内部路径、TCP RTSP；检查不等于真实摄像头或用户环境完整业务E2E。

进一步在该Redis实例用随机临时前缀实际验证Lua TIME+写入、XREADGROUP/XAUTOCLAIM及应用的XACK+XDEL原子脚本，通过后删除本次创建的两个临时键；没有执行FLUSHDB或操作已有任务。

使用当前.env执行 `./scripts/start-application.ps1 -Check` 返回成功：实际启动API、协调器、文件worker、流worker，/health/ready通过Redis、两个worker心跳和MediaMTX检查，随后脚本回收本次进程。此自检不保持服务运行，也未在用户后端提交媒体任务；完整媒体E2E使用隔离的Redis/MediaMTX与临时端口。

后续按用户给定地址，在该环境实际提交并停止了两次RTSP测试任务，其中修复后的任务完成独立客户端播放和音频解码；运行证据与不合格效果见[真实RTSP实测](real-rtsp.md)。

仅新增DETECTION_MAX_SIDE=1280、DETECTION_BOX_PADDING=0.15，将不足16秒的AUDIO_BUFFER_SECONDS调整为16以容纳本机CPU耗时；组件地址、凭据和来源白名单保留。修改后需重启应用进程。

## Redis服务端持久化待配置

实际读取：appendonly=no、appendfsync=everysec、maxmemory-policy=noeviction。
当前Redis状态不能满足本项目AOF always的持久化前提；未执行服务端配置修改或重启。
在已有实例上应在线开启AOF并确认首次写入完成后再重启，避免仅修改配置就重启而损失现有数据。由实例管理员在Redis控制台执行：

```text
CONFIG SET appendonly yes
CONFIG SET appendfsync always
CONFIG SET maxmemory-policy noeviction
INFO persistence
```

确认aof_enabled=1、aof_rewrite_in_progress=0、aof_last_bgrewrite_status=ok及aof_last_write_status=ok。
有可写配置文件时执行CONFIG REWRITE持久保存；否则在实际服务使用的redis.conf或部署管理配置中写入下列内容，并在确认AOF初始化成功后按服务方式重启/复核：

```text
appendonly yes
appendfsync always
maxmemory-policy noeviction
```

这些属于Redis服务端配置，写在项目.env里不会生效。托管实例由平台控制台修改，配置影响整个实例，需管理员实施。
依据：[Redis持久化及切换到AOF](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/)，[XAUTOCLAIM命令](https://redis.io/docs/latest/commands/xautoclaim/)。
