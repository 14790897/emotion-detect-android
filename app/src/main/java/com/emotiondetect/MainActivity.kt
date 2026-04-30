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
    private var maxNumFaces = 1 // 1: 单人, 5: 多人

    // ONNX 情绪分类器
    private val ferEmotionClassifier by lazy { FerEmotionClassifier(this) }
    private val hseEmotionClassifier by lazy { HseEmotionClassifier(this) }
    private var selectedModelId = 0 // 0: FER+, 1: HSEmotion

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

        // 加载保存的模型设置
        val prefs = getSharedPreferences("settings", MODE_PRIVATE)
        selectedModelId = prefs.getInt("selected_model", 0)
        maxNumFaces = prefs.getInt("max_num_faces", 1)

        // 初始化所有模型
        cameraExecutor.execute {
            Log.d(TAG, "后台线程：开始初始化 FaceLandmarker 和 AI 模型...")
            faceLandmarkerHelper = FaceLandmarkerHelper(
                context = this,
                maxNumFaces = maxNumFaces,
                faceLandmarkerHelperListener = this
            )
            Log.d(TAG, "FaceLandmarkerHelper 初始化指令已发出")
            
            ferEmotionClassifier.initialize()
            Log.d(TAG, "FER+ 初始化完成，状态: ${ferEmotionClassifier.isReady()}")
            
            hseEmotionClassifier.initialize()
            Log.d(TAG, "HSEmotion 初始化完成，状态: ${hseEmotionClassifier.isReady()}")
        }

        // 设置按钮
        binding.btnSettings.setOnClickListener {
            showModelSelectionDialog()
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

            val allFaceLandmarks = result.faceLandmarks()
            val allFaceBlendshapes = result.faceBlendshapes().orElse(emptyList())

            if (allFaceLandmarks.isNotEmpty()) {
                // 有人脸：分类情绪 + 更新叠加层
                binding.tvNoFace.visibility = View.GONE

                val finalEmotionResults = mutableListOf<EmotionClassifier.EmotionResult>()

                for (i in allFaceLandmarks.indices) {
                    val faceCrop = resultBundle.faceCropBitmaps.getOrNull(i)
                    val faceBlendshape = allFaceBlendshapes.getOrNull(i) ?: emptyList()

                    var ferResult: FerEmotionClassifier.Result? = null
                    
                    if (faceCrop != null) {
                        when (selectedModelId) {
                            0 -> {
                                if (ferEmotionClassifier.isReady()) {
                                    ferResult = ferEmotionClassifier.classify(faceCrop)
                                    if (i == 0) currentModelName = "ONNX (FER+)"
                                }
                            }
                            1 -> {
                                if (hseEmotionClassifier.isReady()) {
                                    ferResult = hseEmotionClassifier.classify(faceCrop)
                                    if (i == 0) currentModelName = "HSEmotion (E-Net)"
                                }
                            }
                        }
                    }

                    // 优先使用 ONNX 结果，降级到 Blendshape
                    val emotionResult = if (ferResult != null) {
                        ferResult.toEmotionResult()
                    } else {
                        if (i == 0) currentModelName = "MediaPipe"
                        // 只有单人模式才开启平滑
                        EmotionClassifier.classifySingle(faceBlendshape, useSmoothing = (maxNumFaces == 1))
                    }
                    finalEmotionResults.add(emotionResult)
                }

                binding.overlayView.setResults(
                    result,
                    finalEmotionResults,
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

    private fun showModelSelectionDialog() {
        val models = arrayOf("FER+ (经典, 64x64)", "HSEmotion (现代, 224x224)")
        val detectionModes = arrayOf("单人检测", "多人检测 (最多5人)")
        
        val dialogView = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(60, 40, 60, 10)
            
            addView(android.widget.TextView(this@MainActivity).apply {
                text = "识别模型"
                textSize = 16f
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(0, 20, 0, 10)
            })
            
            val modelSpinner = android.widget.Spinner(this@MainActivity)
            modelSpinner.adapter = android.widget.ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, models)
            modelSpinner.setSelection(selectedModelId)
            addView(modelSpinner)
            
            addView(android.widget.TextView(this@MainActivity).apply {
                text = "检测模式"
                textSize = 16f
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(0, 40, 0, 10)
            })
            
            val modeSpinner = android.widget.Spinner(this@MainActivity)
            modeSpinner.adapter = android.widget.ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, detectionModes)
            modeSpinner.setSelection(if (maxNumFaces > 1) 1 else 0)
            addView(modeSpinner)
        }

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("设置")
            .setView(dialogView)
            .setPositiveButton("保存") { _, _ ->
                val newModelId = (dialogView.getChildAt(1) as android.widget.Spinner).selectedItemPosition
                val newModeId = (dialogView.getChildAt(3) as android.widget.Spinner).selectedItemPosition
                val newMaxFaces = if (newModeId == 1) 5 else 1
                
                var changed = false
                if (newModelId != selectedModelId) {
                    selectedModelId = newModelId
                    changed = true
                }
                if (newMaxFaces != maxNumFaces) {
                    maxNumFaces = newMaxFaces
                    changed = true
                    // 如果检测模式改变，需要重新初始化 FaceLandmarker
                    cameraExecutor.execute {
                        faceLandmarkerHelper.clearFaceLandmarker()
                        faceLandmarkerHelper.maxNumFaces = maxNumFaces
                        faceLandmarkerHelper.setupFaceLandmarker()
                    }
                }
                
                if (changed) {
                    getSharedPreferences("settings", MODE_PRIVATE).edit().apply {
                        putInt("selected_model", selectedModelId)
                        putInt("max_num_faces", maxNumFaces)
                    }.apply()
                    Toast.makeText(this, "设置已更新", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.execute {
            ferEmotionClassifier.close()
            hseEmotionClassifier.close()
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
