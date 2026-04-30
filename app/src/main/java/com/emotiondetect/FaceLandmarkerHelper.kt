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
import java.util.concurrent.ScheduledExecutorService
import kotlin.math.max
import kotlin.math.min

/**
 * FaceLandmarkerHelper — MediaPipe Face Landmarker 封装类
 *
 * 负责初始化 FaceLandmarker，处理 CameraX 实时帧，
 * 输出包含 Blendshapes 的检测结果及 ONNX 所需的人脸裁切图。
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

    /** 最新旋转后的帧 Bitmap（用于 ONNX 人脸裁切） */
    private var latestFrameBitmap: Bitmap? = null
    private val bitmapLock = Any()

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
     * 将图像转换为 Bitmap 后传入 FaceLandmarker（异步）
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
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
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

        // 缓存最新帧，供异步结果回调时裁切人脸用
        synchronized(bitmapLock) {
            val oldBitmap = latestFrameBitmap
            latestFrameBitmap = rotatedBitmap
            oldBitmap?.recycle()
        }

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

        // 根据 landmarks 从缓存帧裁切所有检测到的人脸
        val faceCrops = cropAllFacesFromLandmarks(result, input.width, input.height)

        faceLandmarkerHelperListener?.onResults(
            ResultBundle(
                results = listOf(result),
                inferenceTime = inferenceTime,
                inputImageHeight = input.height,
                inputImageWidth = input.width,
                faceCropBitmaps = faceCrops
            )
        )
    }

    /**
     * 根据检测结果中的 landmarks，从帧 Bitmap 裁切出所有检测到的人脸区域（64×64）
     */
    private fun cropAllFacesFromLandmarks(
        result: FaceLandmarkerResult,
        imgWidth: Int,
        imgHeight: Int
    ): List<Bitmap> {
        val allLandmarks = result.faceLandmarks()
        if (allLandmarks.isEmpty()) return emptyList()

        val frame = synchronized(bitmapLock) {
            val f = latestFrameBitmap
            if (f == null || f.isRecycled) return emptyList()
            try {
                f.copy(f.config, false)
            } catch (e: Exception) {
                null
            }
        } ?: return emptyList()

        val crops = mutableListOf<Bitmap>()
        try {
            for (landmarks in allLandmarks) {
                if (landmarks.isEmpty()) continue

                var minX = Float.MAX_VALUE
                var maxX = Float.MIN_VALUE
                var minY = Float.MAX_VALUE
                var maxY = Float.MIN_VALUE
                landmarks.forEach { lm ->
                    val x = lm.x() * imgWidth
                    val y = lm.y() * imgHeight
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                }

                val padding = (maxX - minX) * FACE_CROP_PADDING
                val left = (minX - padding).coerceAtLeast(0f).toInt()
                val top = (minY - padding).coerceAtLeast(0f).toInt()
                val right = (maxX + padding).coerceAtMost(imgWidth.toFloat()).toInt()
                val bottom = (maxY + padding).coerceAtMost(imgHeight.toFloat()).toInt()

                val cropW = right - left
                val cropH = bottom - top
                if (cropW <= 0 || cropH <= 0 || left + cropW > frame.width || top + cropH > frame.height) {
                    continue
                }

                val cropped = Bitmap.createBitmap(frame, left, top, cropW, cropH)
                val scaled = Bitmap.createScaledBitmap(cropped, FACE_CROP_SIZE, FACE_CROP_SIZE, true)
                if (cropped != scaled) cropped.recycle()
                crops.add(scaled)
            }
        } catch (e: Exception) {
            Log.e(TAG, "人脸裁切失败", e)
        } finally {
            if (!frame.isRecycled) frame.recycle()
        }
        return crops
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
        synchronized(bitmapLock) {
            latestFrameBitmap?.recycle()
            latestFrameBitmap = null
        }
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
        const val FACE_CROP_SIZE = 64       // ONNX 模型输入尺寸
        const val FACE_CROP_PADDING = 0.2f  // 人脸框外扩 20%
    }

    /**
     * 检测结果数据类
     */
    data class ResultBundle(
        val results: List<FaceLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
        /** 所有检测到的人脸裁切图 */
        val faceCropBitmaps: List<Bitmap> = emptyList()
    )

    /**
     * 检测结果监听接口
     */
    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
    }
}
