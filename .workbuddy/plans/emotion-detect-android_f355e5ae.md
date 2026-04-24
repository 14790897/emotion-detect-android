---
name: emotion-detect-android
overview: 从零构建一个 Android 人脸情绪实时检测 App，使用 MediaPipe Face Landmarker（Blendshapes）+ 自定义情绪分类器，支持前/后置摄像头切换，实时绘制人脸网格并展示情绪标签和置信度。
design:
  styleKeywords:
    - Camera
    - Minimal
    - Modern
  fontSystem:
    fontFamily: Roboto
    heading:
      size: 24sp
      weight: 600
    subheading:
      size: 18sp
      weight: 500
    body:
      size: 16sp
      weight: 400
  colorSystem:
    primary:
      - "#6200EE"
      - "#03DAC6"
    background:
      - "#000000"
      - "#FFFFFF"
    text:
      - "#FFFFFF"
      - "#000000"
    functional:
      - "#4CAF50"
      - "#2196F3"
      - "#FFEB3B"
      - "#F44336"
      - "#9C27B0"
      - "#607D8B"
todos:
  - id: init-project
    content: 初始化项目结构（Gradle wrapper、项目配置文件）
    status: completed
  - id: config-dependencies
    content: 配置 build.gradle.kts（MediaPipe + CameraX 依赖）
    status: completed
    dependencies:
      - init-project
  - id: create-manifest
    content: 创建 AndroidManifest.xml（摄像头权限声明）
    status: completed
    dependencies:
      - init-project
  - id: download-model
    content: 下载 face_landmarker.task 模型到 assets 目录
    status: completed
    dependencies:
      - init-project
  - id: implement-face-landmarker-helper
    content: 实现 FaceLandmarkerHelper 封装类
    status: completed
    dependencies:
      - config-dependencies
      - download-model
  - id: implement-emotion-classifier
    content: 实现 EmotionClassifier 情绪分类器
    status: completed
    dependencies:
      - config-dependencies
  - id: implement-overlay-view
    content: 实现 OverlayView 人脸网格绘制
    status: completed
    dependencies:
      - config-dependencies
  - id: implement-main-activity
    content: 实现 MainActivity（CameraX + 实时处理）
    status: completed
    dependencies:
      - implement-face-landmarker-helper
      - implement-emotion-classifier
      - implement-overlay-view
  - id: create-layout
    content: 创建 activity_main.xml 布局文件
    status: completed
    dependencies:
      - implement-overlay-view
  - id: build-test
    content: 编译构建和功能测试
    status: completed
    dependencies:
      - create-layout
      - implement-main-activity
---

## 产品概述

使用 MediaPipe Face Landmarker 实现 Android 人脸情绪检测应用，支持前置/后置摄像头实时检测人脸情绪状态。

## 核心功能

- 实时摄像头预览（前置/后置切换）
- MediaPipe Face Landmarker 人脸检测（478个关键点）
- Blendshapes 表情系数提取（52维向量）
- 情绪分类与实时显示（7种基本情绪）
- 人脸网格可视化叠加

## 情绪分类

基于 MediaPipe Blendshapes 52个系数权重规则映射：

- **开心(Happy)**：mouthSmileLeft + mouthSmileRight 高
- **悲伤(Sad)**：mouthFrownLeft + mouthFrownRight + browInnerUp 高
- **惊讶(Surprised)**：jawOpen + eyeWideLeft + eyeWideRight 高
- **愤怒(Angry)**：browDownLeft + browDownRight + noseSneerLeft 高
- **厌恶(Disgusted)**：noseSneerLeft + noseSneerRight 高
- **恐惧(Fearful)**：eyeWideLeft + eyeWideRight + browInnerUp 高
- **中性(Neutral)**：其他情绪均低时

## 技术架构

- **语言**：Kotlin
- **AI模型**：MediaPipe Tasks Vision（face_landmarker.task，支持 Blendshapes）
- **摄像头**：CameraX 1.3.x
- **运行模式**：LIVE_STREAM（实时流）
- **最低SDK**：API 24（Android 7.0）
- **目标SDK**：API 34

## 技术选型

- **框架**：原生 Android + Kotlin
- **AI库**：MediaPipe Tasks Vision（com.google.mediapipe:tasks-vision:latest.release）
- **摄像头**：CameraX（camera-core/camera-camera2/camera-lifecycle/camera-view 1.3.x）
- **视图绑定**：ViewBinding

## 项目结构

```
emotion-detect-android/
├── app/src/main/
│   ├── assets/face_landmarker.task  # 模型文件
│   ├── java/com/emotiondetect/
│   │   ├── MainActivity.kt          # 主界面 + CameraX
│   │   ├── FaceLandmarkerHelper.kt  # MediaPipe 封装类
│   │   ├── EmotionClassifier.kt     # 情绪分类器
│   │   └── OverlayView.kt           # 人脸网格 + 情绪绘制
│   ├── res/layout/activity_main.xml # 布局文件
│   └── AndroidManifest.xml          # 权限配置
├── build.gradle.kts (项目级)
└── app/build.gradle.kts (应用级)
```

## 核心模块

### FaceLandmarkerHelper

- 初始化 FaceLandmarker，配置 outputFaceBlendshapes=true
- detectAsync() 处理实时帧（Bitmap → MPImage）
- 回调返回检测结果和 Blendshapes 系数

### EmotionClassifier

- 输入：52维 Blendshapes 向量
- 逻辑：基于规则权重计算7种情绪概率
- 输出：最高置信度情绪名称 + 数值

### OverlayView

- Canvas 绘制 478 个人脸关键点
- 人脸上方显示当前情绪标签（带背景色）
- 支持缩放和旋转适配

### MainActivity

- CameraX 生命周期管理
- 前/后置摄像头切换按钮
- 实时帧处理管道

## 依赖配置```kotlin

// MediaPipe
implementation("com.google.mediapipe:tasks-vision:latest.release")

// CameraX
implementation("androidx.camera:camera-core:1.3.4")
implementation("androidx.camera:camera-camera2:1.3.4")
implementation("androidx.camera:camera-lifecycle:1.3.4")
implementation("androidx.camera:camera-view:1.3.4")
```

## 界面设计

采用简洁现代的相机界面风格，底部提供摄像头切换按钮，检测结果以浮动标签形式显示在人脸区域上方。

### 页面结构

1. 全屏相机预览区域（占据全部空间）
2. 人脸网格叠加层（半透明轮廓）
3. 情绪标签（浮动在检测到的人脸上方）
4. 底部工具栏（切换摄像头按钮）

### 情绪标签设计

- 圆角矩形背景
- 根据情绪类型显示不同颜色（开心-绿色、悲伤-蓝色、惊讶-黄色、愤怒-红色、厌恶-紫色、恐惧-灰色、中性-白色）
- 显示情绪名称 + 置信度百分比

### 交互设计

- 点击切换按钮切换前/后置摄像头
- 实时更新情绪标签（每帧刷新）
- 无脸时隐藏情绪标签

# Agent Extensions

本任务无需使用 Agent Extensions。