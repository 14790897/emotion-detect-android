package com.emotiondetect

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import java.util.Locale
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.emotiondetect.databinding.ActivityMainBinding
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * MainActivity — 主界面
 *
 * 职责：
 * 1. 管理摄像头权限
 * 2. 启动 CameraX（前/后置切换）
 * 3. 将帧传给 FaceLandmarkerHelper
 * 4. 接收结果并更新 OverlayView（Blendshape + ONNX 双路径）
 */
class MainActivity : AppCompatActivity(), FaceLandmarkerHelper.LandmarkerListener {

    private lateinit var binding: ActivityMainBinding

    // 摄像头相关
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private var cameraFacing = CameraSelector.LENS_FACING_FRONT
    private var isFrontCamera = true

    // 分析线程
    private lateinit var cameraExecutor: ExecutorService

    // MediaPipe 封装
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper

    // 图像分析
    private var imageAnalyzer: ImageAnalysis? = null
    private var lastInferenceTimeMs: Long = 0
    private var frameCounter = 0
    private var lastFpsTimestamp: Long = 0
    private var processedFrameCount = 0
    private var currentFps = 0.0
    private var currentModelName = "None"

    // ONNX 情绪分类器（emotion-ferplus-8.onnx）
    private val ferEmotionClassifier by lazy { FerEmotionClassifier(this) }

    // 权限请求
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                startCamera()
            } else {
                Toast.makeText(this, R.string.permission_denied, Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 全屏显示
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_FULLSCREEN or
            View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        )

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 初始化 executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 初始化 FaceLandmarkerHelper 和 ONNX 分类器
        cameraExecutor.execute {
            faceLandmarkerHelper = FaceLandmarkerHelper(
                context = this,
                faceLandmarkerHelperListener = this
            )
            ferEmotionClassifier.initialize()
        }

        // 切换摄像头按钮
        binding.btnSwitchCamera.setOnClickListener {
            isFrontCamera = !isFrontCamera
            cameraFacing = if (isFrontCamera) {
                CameraSelector.LENS_FACING_FRONT
            } else {
                CameraSelector.LENS_FACING_BACK
            }
            startCamera()
        }

        // 检查权限
        checkCameraPermission()
    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: run {
            Log.e(TAG, "Camera provider is null")
            return
        }

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(cameraFacing)
            .build()

        // 预览
        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .build()
            .also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

        // 图像分析（用于 MediaPipe）
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor, ::analyzeImage)
            }

        // 解绑旧用例
        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer!!
            )
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    /**
     * 每帧图像分析回调
     */
    private fun analyzeImage(imageProxy: ImageProxy) {
        frameCounter++

        // 根据上一次的推理时间动态决定处理间隔（Skip Interval）
        val skipInterval = when {
            lastInferenceTimeMs < 40 -> 1
            lastInferenceTimeMs < 80 -> 2
            else -> 3
        }

        if (frameCounter % skipInterval != 0) {
            imageProxy.close() // 丢弃该帧，不进行推理
            return
        }

        if (!::faceLandmarkerHelper.isInitialized || faceLandmarkerHelper.isClosed()) {
            imageProxy.close()
            return
        }
        faceLandmarkerHelper.detectLiveStream(
            imageProxy = imageProxy,
            isFrontCamera = isFrontCamera
        )
    }

    // ---------- FaceLandmarkerHelper.LandmarkerListener ----------

    override fun onResults(resultBundle: FaceLandmarkerHelper.ResultBundle) {
        runOnUiThread {
            val result = resultBundle.results.firstOrNull() ?: return@runOnUiThread

            // 更新推理时间
            lastInferenceTimeMs = resultBundle.inferenceTime
            binding.tvInferenceTime.text = getString(
                R.string.inference_time_ms, lastInferenceTimeMs
            )

            // 计算实际处理 FPS
            processedFrameCount++
            val now = SystemClock.elapsedRealtime()
            if (now - lastFpsTimestamp >= 1000) {
                currentFps = processedFrameCount * 1000.0 / (now - lastFpsTimestamp)
                processedFrameCount = 0
                lastFpsTimestamp = now
            }

            // 更新状态显示
            val mode = when {
                lastInferenceTimeMs < 40 -> "全速 (1:1)"
                lastInferenceTimeMs < 80 -> "均衡 (1:2)"
                else -> "节能 (1:3)"
            }
            binding.tvStatus.text = String.format(Locale.getDefault(), "模型: %s | %s | %.1f FPS", currentModelName, mode, currentFps)
            binding.tvStatus.setTextColor(when {
                lastInferenceTimeMs < 40 -> 0xFF4CAF50.toInt() // 绿色
                lastInferenceTimeMs < 80 -> 0xFFFFC107.toInt() // 黄色
                else -> 0xFFF44336.toInt() // 红色
            })

            if (result.faceLandmarks().isNotEmpty()) {
                // 有人脸：分类情绪 + 更新叠加层
                binding.tvNoFace.visibility = View.GONE

                // --- ONNX 推理路径（优先）---
                val ferResult = resultBundle.faceCropBitmap?.let { faceCrop ->
                    if (ferEmotionClassifier.isReady()) {
                        ferEmotionClassifier.classify(faceCrop).also {
                            currentModelName = "ONNX (FER+)"
                        }
                    } else null
                }

                // --- Blendshape 备用路径（ONNX 失败时降级）---
                val blendshapeResult = EmotionClassifier.classify(
                    result.faceBlendshapes().orElse(emptyList())
                ).also {
                    if (ferResult == null) currentModelName = "MediaPipe"
                }

                // 优先使用 ONNX 结果，降级到 Blendshape
                val emotionResult = if (ferResult != null) {
                    ferResult.toEmotionResult()
                } else {
                    blendshapeResult
                }

                binding.overlayView.setResults(
                    result,
                    emotionResult,
                    resultBundle.inputImageWidth,
                    resultBundle.inputImageHeight
                )
            } else {
                // 无人脸
                binding.tvNoFace.visibility = View.VISIBLE
                binding.overlayView.clearResults()
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        runOnUiThread {
            Log.e(TAG, "FaceLandmarker error: $error (code $errorCode)")
            Toast.makeText(this, "检测错误: $error", Toast.LENGTH_SHORT).show()
        }
    }

    // ---------- 生命周期 ----------

    override fun onResume() {
        super.onResume()
        if (::faceLandmarkerHelper.isInitialized && faceLandmarkerHelper.isClosed()) {
            cameraExecutor.execute {
                faceLandmarkerHelper.setupFaceLandmarker()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (::faceLandmarkerHelper.isInitialized) {
            cameraExecutor.execute {
                faceLandmarkerHelper.clearFaceLandmarker()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.execute {
            ferEmotionClassifier.close()
        }
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "MainActivity"
    }

    /**
     * 将 FerEmotionClassifier.Result（8类 ONNX）转换为 EmotionClassifier.EmotionResult（7类）
     * 用于统一 OverlayView 接口
     *
     * ONNX 8类 → App 7类映射：
     * neutral(0) → NEUTRAL, happiness(1) → HAPPY, surprise(2) → SURPRISED,
     * sadness(3) → SAD, anger(4) → ANGRY, disgust(5) → DISGUSTED,
     * fear(6) → FEARFUL, contempt(7) → NEUTRAL
     */
    private fun FerEmotionClassifier.Result.toEmotionResult(): EmotionClassifier.EmotionResult {
        return EmotionClassifier.EmotionResult(
            emotion = when (this.emotion) {
                FerEmotionClassifier.Emotion.NEUTRAL   -> EmotionClassifier.Emotion.NEUTRAL
                FerEmotionClassifier.Emotion.HAPPINESS -> EmotionClassifier.Emotion.HAPPY
                FerEmotionClassifier.Emotion.SURPRISE -> EmotionClassifier.Emotion.SURPRISED
                FerEmotionClassifier.Emotion.SADNESS   -> EmotionClassifier.Emotion.SAD
                FerEmotionClassifier.Emotion.ANGER     -> EmotionClassifier.Emotion.ANGRY
                FerEmotionClassifier.Emotion.DISGUST  -> EmotionClassifier.Emotion.DISGUSTED
                FerEmotionClassifier.Emotion.FEAR      -> EmotionClassifier.Emotion.FEARFUL
                FerEmotionClassifier.Emotion.CONTEMPT -> EmotionClassifier.Emotion.NEUTRAL  // 轻蔑映射到中性
            },
            confidence = this.confidence,
            scores = mapOf(
                EmotionClassifier.Emotion.NEUTRAL   to probScores.getOrElse(0) { 0f },
                EmotionClassifier.Emotion.HAPPY     to probScores.getOrElse(1) { 0f },
                EmotionClassifier.Emotion.SURPRISED to probScores.getOrElse(2) { 0f },
                EmotionClassifier.Emotion.SAD       to probScores.getOrElse(3) { 0f },
                EmotionClassifier.Emotion.ANGRY     to probScores.getOrElse(4) { 0f },
                EmotionClassifier.Emotion.DISGUSTED to probScores.getOrElse(5) { 0f },
                EmotionClassifier.Emotion.FEARFUL   to probScores.getOrElse(6) { 0f }
            )
        )
    }
}
