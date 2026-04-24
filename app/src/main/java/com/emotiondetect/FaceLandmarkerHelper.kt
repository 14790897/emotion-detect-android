package com.emotiondetect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService

/**
 * FaceLandmarkerHelper — MediaPipe Face Landmarker 封装类
 *
 * 负责初始化 FaceLandmarker，处理 CameraX 实时帧，
 * 输出包含 Blendshapes 的检测结果。
 */
class FaceLandmarkerHelper(
    var minFaceDetectionConfidence: Float = DEFAULT_FACE_DETECTION_CONFIDENCE,
    var minFaceTrackingConfidence: Float = DEFAULT_FACE_TRACKING_CONFIDENCE,
    var minFacePresenceConfidence: Float = DEFAULT_FACE_PRESENCE_CONFIDENCE,
    var maxNumFaces: Int = DEFAULT_NUM_FACES,
    val context: Context,
    val faceLandmarkerHelperListener: LandmarkerListener? = null
) {

    private var faceLandmarker: FaceLandmarker? = null
    private var backgroundExecutor: ScheduledExecutorService? = null

    init {
        setupFaceLandmarker()
    }

    fun isClosed(): Boolean {
        return faceLandmarker == null
    }

    /**
     * 初始化 FaceLandmarker，配置 Blendshapes 输出
     */
    fun setupFaceLandmarker() {
        val baseOptionsBuilder = BaseOptions.builder()
            .setModelAssetPath(MP_FACE_LANDMARKER_TASK)
            .setDelegate(Delegate.CPU)  // 使用 CPU 以保证最大兼容性

        val optionsBuilder = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptionsBuilder.build())
            .setMinFaceDetectionConfidence(minFaceDetectionConfidence)
            .setMinTrackingConfidence(minFaceTrackingConfidence)
            .setMinFacePresenceConfidence(minFacePresenceConfidence)
            .setNumFaces(maxNumFaces)
            .setOutputFaceBlendshapes(true)           // 输出 52 个表情系数
            .setOutputFacialTransformationMatrixes(false)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(this::returnLivestreamResult)
            .setErrorListener(this::returnLivestreamError)

        try {
            faceLandmarker = FaceLandmarker.createFromOptions(context, optionsBuilder.build())
            Log.d(TAG, "FaceLandmarker initialized successfully")
        } catch (e: Exception) {
            faceLandmarkerHelperListener?.onError(
                "FaceLandmarker 初始化失败: ${e.message}",
                GPU_ERROR
            )
            Log.e(TAG, "FaceLandmarker initialization failed", e)
        }
    }

    /**
     * 处理来自 CameraX 的 ImageProxy 帧
     * 将图像转换为 Bitmap 后传入 FaceLandmarker
     */
    fun detectLiveStream(
        imageProxy: ImageProxy,
        isFrontCamera: Boolean
    ) {
        val frameTime = SystemClock.uptimeMillis()

        // 将 ImageProxy 转为 Bitmap
        val bitmapBuffer = Bitmap.createBitmap(
            imageProxy.width,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )
        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }

        val matrix = Matrix().apply {
            // 旋转以适配摄像头方向
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            // 前置摄像头需要水平镜像
            if (isFrontCamera) {
                postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
            }
        }

        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer,
            0, 0,
            bitmapBuffer.width,
            bitmapBuffer.height,
            matrix,
            true
        )

        val mpImage = BitmapImageBuilder(rotatedBitmap).build()
        detectAsync(mpImage, frameTime)
    }

    /**
     * 异步执行检测（LIVE_STREAM 模式）
     */
    @VisibleForTesting
    fun detectAsync(mpImage: MPImage, frameTime: Long) {
        faceLandmarker?.detectAsync(mpImage, frameTime)
    }

    /**
     * 接收 LIVE_STREAM 检测结果回调
     */
    private fun returnLivestreamResult(
        result: FaceLandmarkerResult,
        input: MPImage
    ) {
        val finishTimeMs = SystemClock.uptimeMillis()
        val inferenceTime = finishTimeMs - result.timestampMs()

        faceLandmarkerHelperListener?.onResults(
            ResultBundle(
                listOf(result),
                inferenceTime,
                input.height,
                input.width
            )
        )
    }

    /**
     * 接收 LIVE_STREAM 检测错误回调
     */
    private fun returnLivestreamError(error: RuntimeException) {
        faceLandmarkerHelperListener?.onError(
            error.message ?: "未知错误",
            OTHER_ERROR
        )
    }

    /**
     * 关闭并释放资源
     */
    fun clearFaceLandmarker() {
        faceLandmarker?.close()
        faceLandmarker = null
    }

    companion object {
        const val TAG = "FaceLandmarkerHelper"
        const val MP_FACE_LANDMARKER_TASK = "face_landmarker.task"

        const val DEFAULT_FACE_DETECTION_CONFIDENCE = 0.5f
        const val DEFAULT_FACE_TRACKING_CONFIDENCE = 0.5f
        const val DEFAULT_FACE_PRESENCE_CONFIDENCE = 0.5f
        const val DEFAULT_NUM_FACES = 1

        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
    }

    /**
     * 检测结果数据类
     */
    data class ResultBundle(
        val results: List<FaceLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int
    )

    /**
     * 检测结果监听接口
     */
    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
    }
}
