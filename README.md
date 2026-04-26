# EmotionDetect — 人脸情绪实时检测 Android App

基于 **MediaPipe Face Landmarker** + 自定义情绪分类器，实时检测人脸 7 种基本情绪。

## ✨ 功能

- 📷 前/后置摄像头实时切换
- 🧠 MediaPipe Face Landmarker（478 个 3D 人脸关键点 + 52 个 Blendshape 系数）
- 😊 7 种情绪实时分类：开心 / 悲伤 / 惊讶 / 愤怒 / 厌恶 / 恐惧 / 中性
- 🎨 半透明人脸网格叠加可视化
- 📊 情绪置信度进度条 + 动画过渡

## 🏗️ 架构

```
app/src/main/java/com/emotiondetect/
├── MainActivity.kt          # CameraX 管理 + 帧处理调度
├── FaceLandmarkerHelper.kt  # MediaPipe LIVE_STREAM 模式封装
├── EmotionClassifier.kt     # Blendshape → 情绪 规则分类器
└── OverlayView.kt           # Canvas 绘制人脸网格 + 情绪标签
```

## 🚀 快速开始

### 要求

- Android Studio Hedgehog (2023.1) 或更高
- Android 7.0 (API 24) 及以上设备
- 有摄像头的真实设备（模拟器不支持摄像头实时流）

### 构建步骤

1. 用 Android Studio 打开项目根目录
2. 等待 Gradle 同步完成（首次会下载 MediaPipe 依赖，约 50MB）
3. 连接真实设备（需开启 USB 调试）
4. 点击 Run，授予摄像头权限即可

> **注意**：模型文件 `face_landmarker.task`（约 3.7MB）已包含在 `assets/` 目录中，无需额外下载。

## 📦 主要依赖

| 库 | 版本 | 用途 |
|----|------|------|
| MediaPipe Tasks Vision | 0.10.14 | 人脸关键点 + Blendshapes |
| CameraX | 1.3.4 | 摄像头预览与帧分析 |
| Material Components | 1.12.0 | UI 组件 |

## 🎯 情绪分类原理

基于 MediaPipe Blendshapes 52 个面部表情系数的加权规则：

| 情绪 | 关键 Blendshape | 颜色 |
|------|----------------|------|
| 😊 开心 | mouthSmileLeft/Right + cheekSquint | 🟢 绿色 |
| 😢 悲伤 | mouthFrownLeft/Right + browInnerUp | 🔵 蓝色 |
| 😮 惊讶 | jawOpen + eyeWideLeft/Right | 🟡 黄色 |
| 😠 愤怒 | browDownLeft/Right + noseSneer | 🔴 红色 |
| 🤢 厌恶 | noseSneerLeft/Right + mouthUpperUp | 🟣 紫色 |
| 😨 恐惧 | eyeWideLeft/Right + browInnerUp | ⬜ 灰色 |
| 😐 中性 | 其他情绪均低 | ⚪ 白色 |

内置 5 帧滑动平均滤波，减少帧间抖动。


## 模型下载
https://github.com/sb-ai-lab/EmotiEffLib/blob/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx
https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion_ferplus/README.md

## logcat过滤日志
package:com.emotiondetect

## 开发日志
模型更新之后，在旧版应用上会有旧缓存，必须卸载之后才能更新