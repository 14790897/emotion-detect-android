# MEMORY.md — emotion-detect-android

## 项目概述
Android 人脸情绪实时检测 App，MediaPipe Face Landmarker + CameraX + Kotlin。

## 关键技术决策
- MediaPipe `tasks-vision:0.10.14`，`faceBlendshapes()` 返回 `Optional<List<List<Category>>>`
- **双路径情绪推理**：ONNX Runtime（emotion-ferplus-8.onnx，64×64，8类，标签：neutral/happiness/surprise/sadness/anger/disgust/fear/contempt，微软 ONNX Model Zoo 官方发布）为主，Blendshape 规则为降级备选
- 人脸裁切：FaceLandmarkerHelper 在异步结果回调时根据 landmarks 从缓存帧 Bitmap 裁切 64×64 传人 ONNX（避免每帧双检测）
- ONNX 模型加载：`OrtEnvironment.getEnvironment()` + `env.createSession(path, opts)`（ai.onnxruntime 包，构造函数为私有，必须用工厂方法）
- ONNX 模型大小：33 MB，模型需放入 `app/src/main/assets/emotion-ferplus-8.onnx`，运行时自动复制到私有目录
- 使用 CPU Delegate 确保兼容性（GPU 代理兼容性不稳定）
- 情绪分类：基于 Blendshape 加权规则 + 5 帧滑动均值平滑
- CameraX RGBA_8888 格式输入，前置摄像头需水平镜像

## 项目结构
```
app/src/main/java/com/emotiondetect/
  MainActivity.kt / FaceLandmarkerHelper.kt / EmotionClassifier.kt / FerEmotionClassifier.kt / OverlayView.kt
app/src/main/assets/
  face_landmarker.task     (3.76MB, MediaPipe)
  emotion-ferplus-8.onnx   (33MB, 微软 ONNX Model Zoo，8类情绪)
```

## 构建说明
Android Studio 打开根目录，需真实设备（模拟器无摄像头流）
