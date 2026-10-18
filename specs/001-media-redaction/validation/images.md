# 图片验证

入口：tests/e2e/test_images.py、tests/integration/test_image_api.py、tests/unit/test_redaction.py、tests/unit/test_source.py。
端到端使用实际Taskiq、临时Redis和官方双模型；HTTP边界另用ASGI传输，测试桩只验证已知目标框的遮挡。

覆盖官方人脸JPEG上传/URL、无目标PNG上传/URL、终态下载、原文件保留、损坏输入失败、未成功产物不可下载。
真实人脸结果保留1080×1920尺寸，已知检测区域内部为黑色；EXIF方向被应用，输出不复制EXIF/缩略图。
另验证202/Location、未知ID、容量429、取消幂等、任意磁盘路径隔离；源URL不进入公共响应。

来源边界包括默认拒绝、显式授权loopback、混合DNS地址拒绝、连接批准IP并保留Host/TLS SNI、重定向重新授权、实际字节上限及总超时。
DNS与重定向用确定性替身检查连接目标，实际限量下载和提交使用真实HTTP服务；未完成真实TLS证书/代理组合测试。

对照产物在data/validation/official-demo-redacted.jpg与data/validation/opencv-licenseplate-redacted.jpg，原图在samples。
模型各返回一个框不等于人工标注或没有漏检，质量验收另见detection-quality.md。
