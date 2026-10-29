# E1 独立候选：取消明显迟到后的计划时间重设

基线：`21bd631d5a3d02de052974b57f73428560944804`，本地独立分支`feature/stream-e1-no-rebase`。本候选不包含模型关键路径观测或优化，没有部署、push、main或已有tag修改。

## 唯一应用变量

只删除`media/stream_pacing.py`中：

```python
rebased = due < checked_at - interval / 2
if rebased:
    due = self.clock()
```

实际due仍为`max(previous_release + interval, previous_actual + interval / 2)`；相邻正常40ms媒体时钟仍保留20ms防突发下限；单次媒体间隔限幅.5s，停止事件仍可打断。E0观测中的`rebased`显式为false；原PTS、音频修复、前视补遮、队列、编解码配置及任务生命周期均未改。

新回归先在基线失败：迟到1秒后的`previous_release`被重设为1.0，预期保留.04；E1通过，后续快帧只等到1.020而非1.040。没有关闭全部节奏等待。

## 政策测试的必要变化

- 原场景切换/2秒停顿用例要求之后每帧至少完整1/15秒；它锁定了旧重设政策。E1只在存在停顿时改为至少1/30秒，仍验证全部40帧、原PTS与前视补遮mask一致；无停顿时完整1/15秒断言保留。
- 原E0等待例子的rebase=1、条件新增等待.04s，改为E1的rebase=0及新增等待0，保留观测开/关决策一致和防突发等待.015s断言。
- 固定旧版本与新版本逐次相同的回归，改为新版本观测开/关逐次相同、请求等待总量不多于旧版本及实际释放不早于due。它不再宣称E1政策与旧政策完全等价。
- 新增等待进行中停止的真实Event/Thread检查，要求停止返回false，不推进未发出帧的媒体时钟。

上述变化只调整明确要求改变的发布政策断言，没有删除PTS、帧序、遮罩、音频或停止保护。

## 真实媒体链验收

新集成测试`tests/integration/test_e1_pacing_media.py`使用同一100帧H264、25FPS、8kHz AAC素材、真实`process_stream`、真实`PlateLookahead`、编码、RTSP/TCP peer及独立解码接收。

1. CPU适配器不伪装GPU能力结论：只替代CUDA输入接口，以真实CPU H264编码及固定ROI执行；素材夹具的前12帧依赖前视补遮或保守补查取得车牌mask，并断言反向补遮计数>0，分别验证连续、双轨PTS重置和第30帧停止。基线/E1每轨包内容、PTS/DTS及网络图片字节一致，独立解码后人脸与车牌区域满足遮罩像素断言。
2. 本机GTX970实际NVDEC→CUDA颜色转换/遮罩→真实PlateLookahead→NVENC→RTSP的连续与PTS重置两组通过，每臂100帧，视频/AAC包字节及时钟完全一致。固定ROI检测夹具，不加载quality模型；这证明媒体链正确性，不代替真实rfm遮挡质量或RTX模型吞吐。
3. RTP时钟与提交包的相对映射一致；RTCP音视频共同sender时间偏差<2ms。两政策均保留既有固定素材teardown少一个最终AAC AU现象，未宣称完美AAC交付。
4. 两臂统一使用`perf_counter`测Pacer释放间隔，避免本机Python的Windows `monotonic`粗粒度tick误报20ms下限。实际Pacer释放最小间隔断言>=19ms，精确20ms政策由确定性时钟回归锁定。应用默认时钟未改；mux返回及RTP到达仍可能因编码/网络批次挤压，不用它们的最小间隔代替Pacer实际释放间隔。

全部6项媒体测试通过：`6 passed in 79.07s`。其中2项显式开启真实GPU，没有按条件跳过。媒体测试整体UTC+8约22:41:12–22:42:31，GPU两项在最后约20秒，与主线quality诊断有并发；不能使用这两项的耗时推导GPU吞吐。

## 独立可复现的调度收益

同一真实4秒H264/AAC素材，使用上述CPU适配器、每帧相同90/10/10ms处理成本循环、同一原PTS、FIFO背压和真实前视/编码/RTSP。只替换固定基线Pacer与E1；首尾有效mux间隔口径：

| 轮次 | 基线秒 / FPS | E1秒 / FPS | 时间减少 |
|---|---:|---:|---:|
| 1 | 5.9831 / 16.55 | 5.3447 / 18.52 | 10.67% |
| 2 | 5.9892 / 16.53 | 5.3509 / 18.50 | 10.66% |

两轮均超过2%测试门槛，FPS约提高11.9%，不是模拟时钟结果。但处理成本为注入，源按FIFO背压读取，不是独立稳定25FPS的真实网络摄像机。这个结果只证明有界调度收益可在真实媒体链复现，**不能外推rfm/RTX收益，也仍未达到25FPS**。源文件SHA256与逐轮值见[local-media-results.json](local-media-results.json)。

## 测试与复跑

本工作树使用共享venv，但必须显式令`PYTHONPATH`指向本工作树`src`，避免误测主线：

```powershell
$env:PYTHONPATH='C:/Users/zhanjh/.codex/worktrees/stream-e1-no-rebase/vision-shield/src'
H:/workspace/python/vision-shield/.venv/Scripts/python.exe -m pytest tests/unit -q
$env:RUN_GPU_TESTS='1'
H:/workspace/python/vision-shield/.venv/Scripts/python.exe -m pytest tests/integration/test_e1_pacing_media.py -q
```

完整unit：**915 passed、96 skipped、9 warnings，164.09s**（新增真实Event停止用例随后单独复核；不把96个条件跳过算通过）。最终Pacer、E0计数、CUDA合同、真实音频时钟及前视复核 **122 passed，27.99s**。没有为既有接收器退出问题调整测试超时或删断言。

本候选已有本地媒体正确性和调度收益证据，但**尚无真实rfm质量模型E1对照、RTX文件对照、实时RTP/长期延迟或播放器缓冲验收**。不得把本地18.5 FPS宣布为实机修复；下一次覆盖部署仍应先提交固定commit、影响和回滚到E0的精确指令并申请授权。
