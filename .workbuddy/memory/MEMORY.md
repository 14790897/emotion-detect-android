# MEMORY.md — emotion-detect-android

## 项目概述
Android 人脸情绪实时检测 App，MediaPipe Face Landmarker + CameraX + Kotlin。

## 关键技术决策
- MediaPipe `tasks-vision:0.10.14`，`faceBlendshapes()` 返回 `Optional<List<List<Category>>>`
- 使用 CPU Delegate 确保兼容性（GPU 代理兼容性不稳定）
- 情绪分类：基于 Blendshape 加权规则 + 5 帧滑动均值平滑
- CameraX RGBA_8888 格式输入，前置摄像头需水平镜像

## 项目结构
```
app/src/main/java/com/emotiondetect/
  MainActivity.kt / FaceLandmarkerHelper.kt / EmotionClassifier.kt / OverlayView.kt
app/src/main/assets/face_landmarker.task  (3.76MB, 已下载)
```

## 构建说明
Android Studio 打开根目录，需真实设备（模拟器无摄像头流）
