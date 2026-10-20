# 四模型 CUDA 推理与部署

2026-10-20。`INFERENCE_DEVICE=cuda` 且 `DETECTION_BACKEND=quality` 时，EgoBlur人脸/车牌使用PyTorch CUDA，YuNet/HyperLPR使用ONNX Runtime CUDA。图像缩放、NMS后处理、遮挡及x264编码仍在CPU；这里的“四模型GPU”不代表整条视频管线没有CPU操作。

## 实现与边界

复用原分块、预处理、阈值、HyperLPR解码和跨模型复核。YuNet是全卷积网络，运行时仅将固定输入及中间尺寸元数据改成动态尺寸，保持原权重文件、哈希及像素输入尺寸；按OpenCV FaceDetectorYN的公式解码，并保留整数框NMS语义。GPU与CPU浮点运算可能产生亚像素差异，不能承诺逐字节一致。

CUDA路径要求两路ONNX session都启用CUDAExecutionProvider，禁止CPU算子回退和运行异常后的自动重试降级。预检返回 `detail_providers`，真实GPU测试还解析两路ONNX执行profile，核对实际算子provider。使用FP32，关闭TF32；卷积算法使用HEURISTIC。

新增可选依赖 `cuda`。ONNX Runtime固定1.20.2，与现有CUDA12.6/cuDNN9运行库配套；不盲目使用当前默认CUDA13发行包。CPU镜像不安装该extra。Windows本地安装：

```powershell
uv sync --frozen --extra cuda
```

生成GPU部署锁的共同约束导出及 `uv pip compile` 均需加 `--extra cuda`，其余沿用[原锁生成流程](research.md)。本轮已生成带哈希的 `deploy/requirements-cu126.txt` 并检查根锁与共同版本。

官方依据：[ONNX Runtime CUDA兼容性](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、[OpenCV FaceDetectorYN实现](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_detect.cpp)。

## 本机验证

- 185项单元/视频管线测试通过；3项真实CUDA测试通过，包含两个已标注质量样例与实际算子provider检查。
- 15个可用质量标注样例通过；历史 `rtsp-green.png` 缺失，没有计为通过。
- 四模型CUDA连续重放视频1的0～4秒、视频2的8～12秒和88～91秒，共223帧，帧数与时间戳核对通过。10秒标注侧脸遮挡、手提包原像素保留断言通过；人工检查关键区间未见此前的遮挡消失。源片位于 `tests/test_assets/report/`，文件哈希与先前服务器测试相同。
- GTX1080同图补充检测单独计时，两张1920×536视频帧预热后CPU约0.30秒、CUDA HEURISTIC约0.11～0.15秒。DEFAULT卷积算法约0.69～0.80秒，已排除。此项不包括主检测、局部复核、编解码或队列，不能作为服务器整条视频链的提速倍数。
- 证据目录：`data/validation/known-issues-20261020/`，包括 `cuda-model-tests.log`、`cuda-quality-regression.log`、`cuda-detail-benchmark.json`。

## 服务器部署

将 `vision-shield-gpu-update.tar.gz` 复制到现有服务器项目目录并在那里解压；包内不含 `.env.compose`、模型或测试视频，沿用现有GPU UUID与模型目录。先让正在运行的任务完成，避免重建worker中断任务。

```bash
tar -xzf vision-shield-gpu-update.tar.gz
dc=(docker-compose --env-file .env.compose -p vision-shield -f compose.yaml -f compose.gpu.yaml)
"${dc[@]}" build api file-worker
"${dc[@]}" run --rm --no-deps file-worker python -m vision_shield.detection.preflight
```

预检必须成功，并同时满足：

- `device` 为 `cuda`，`model_devices` 为 `["cuda"]`；
- `detail_providers` 为 `["CUDAExecutionProvider", "CUDAExecutionProvider"]`；
- GPU名称与所选RTX6000 Ada一致，`visible_devices` 为1。

然后替换运行服务：

```bash
"${dc[@]}" up -d --no-deps --force-recreate api file-worker stream-worker
"${dc[@]}" ps
"${dc[@]}" exec -T api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health/ready').read().decode())"
```

健康检查使用容器内部端口，不依赖本机18000转发。API与worker使用同一批源码。此次新增GPU依赖会重建依赖层，保留BuildKit缓存，不加 `--no-cache`；后续纯源码更新继续复用依赖层。

部署完成后通过已有转发地址发起原视频任务，记录完整帧数、耗时、体积、资源采样，并复查视频1的2～3秒及视频2的10秒。服务器端到端验收尚未完成，不把本机结果记作服务器结果。
