# 检测质量：待人工验收

T040 / SC-007 **未完成**。没有人工复核结果，不计算或宣称准确率。

| 样本 | 原始文件 | 双模型结果 | 人工目标总数/漏检/误检 |
|---|---|---|---|
| 官方EgoBlur演示人脸 | samples/official-demo.jpg | data/validation/official-demo-redacted.jpg | 未复核 |
| OpenCV运动模糊车牌 | samples/opencv-licenseplate.jpg | data/validation/opencv-licenseplate-redacted.jpg | 未复核 |
| 无目标 | 自动生成纯色JPEG/PNG | E2E逐像素检查PNG | 合成输入，不计代表性准确率 |

来源、许可和哈希见tests/fixtures/manifest.json。自动运行两张真实样本各得到一个框，仅证明模型输出，不等于人工核定目标数。
固定结果供用户对照；尚缺人脸与车牌共存、遮挡/小目标，以及代表性真实视频画面的人工标注范围。
完成时逐样本记录类别、目标总数、漏检、误检、复核帧编号/时间段，保留原始与输出；不得将未复核范围纳入统计。

本机采用CPU，当前检测最长边1280、每侧外扩15%，输出仍为原分辨率。用户提供16张巡逻机器人图片与5段视频，局部自动对照见robot-media.md；尚未建立人工穷举标注，T040仍未完成。小目标和模糊目标仍可能漏检；原音频未脱敏，输出不代表音视频完全匿名化。
