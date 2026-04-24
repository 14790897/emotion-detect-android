package com.emotiondetect

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Toast
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
import com.google.mediapipe.tasks.components.containers.Category
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
 * 4. 接收结果并更新 OverlayView
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

        // 初始化 FaceLandmarkerHelper（在后台线程）
        cameraExecutor.execute {
            faceLandmarkerHelper = FaceLandmarkerHelper(
                context = this,
                faceLandmarkerHelperListener = this
            )
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
        val imageAnalyzer = ImageAnalysis.Builder()
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
                imageAnalyzer
            )
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    /**
     * 每帧图像分析回调
     */
    private fun analyzeImage(imageProxy: ImageProxy) {
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
            binding.tvInferenceTime.text = getString(
                R.string.inference_time_ms, resultBundle.inferenceTime
            )

            if (result.faceLandmarks().isNotEmpty()) {
                // 有人脸：分类情绪并更新叠加层
                binding.tvNoFace.visibility = View.GONE

                // faceBlendshapes() 返回 Optional<List<List<Category>>>
                val blendshapesOpt = result.faceBlendshapes()
                val blendshapes: List<List<Category>> =
                    if (blendshapesOpt.isPresent) {
                        blendshapesOpt.get()
                    } else {
                        emptyList()
                    }

                val emotion = EmotionClassifier.classify(blendshapes)

                binding.overlayView.setResults(
                    result,
                    emotion,
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
        // 重新进入时恢复 FaceLandmarker（若已关闭）
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
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
