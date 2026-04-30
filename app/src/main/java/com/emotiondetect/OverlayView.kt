package com.emotiondetect

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import android.view.animation.DecelerateInterpolator
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.max
import kotlin.math.min

/**
 * OverlayView — 人脸网格 + 情绪标注绘制层
 *
 * 叠加在 CameraX PreviewView 之上，绘制：
 * 1. 人脸轮廓（478 关键点 + 连接线）
 * 2. 浮动情绪标签（带渐变背景色 + 置信度进度条）
 * 3. 关键点点阵
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // ---------- 数据 ----------
    private var results: FaceLandmarkerResult? = null
    private var emotionResults: List<EmotionClassifier.EmotionResult>? = null
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    // ---------- 画笔 ----------
    private val landmarkPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#80FFFFFF")
        strokeWidth = 1.5f
        style = Paint.Style.FILL
    }

    private val connectionPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#4003DAC6")
        strokeWidth = 1.2f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 52f
        typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
        textAlign = Paint.Align.CENTER
    }

    private val subTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CCFFFFFF")
        textSize = 34f
        typeface = Typeface.create("sans-serif", Typeface.NORMAL)
        textAlign = Paint.Align.CENTER
    }

    private val progressBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#40FFFFFF")
        style = Paint.Style.FILL
    }

    private val progressFgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val faceOutlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#6006C20E")
        strokeWidth = 2f
        style = Paint.Style.STROKE
    }

    // ---------- 动画 ----------
    private var animatedConfidence = 0f
    private var confidenceAnimator: ValueAnimator? = null
    private var labelAlpha = 255

    // ---------- 公共方法 ----------

    fun setResults(
        faceLandmarkerResults: FaceLandmarkerResult,
        emotions: List<EmotionClassifier.EmotionResult>,
        imageWidth: Int,
        imageHeight: Int
    ) {
        results = faceLandmarkerResults
        emotionResults = emotions
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight

        // 置信度动画（暂时只取第一个，或者可以为每个脸做动画，这里简化处理）
        val targetConf = emotions.firstOrNull()?.confidence ?: 0f
        confidenceAnimator?.cancel()
        confidenceAnimator = ValueAnimator.ofFloat(animatedConfidence, targetConf).apply {
            duration = 300
            interpolator = DecelerateInterpolator()
            addUpdateListener {
                animatedConfidence = it.animatedValue as Float
                invalidate()
            }
            start()
        }

        invalidate()
    }

    fun clearResults() {
        results = null
        emotionResults = null
        EmotionClassifier.clearSmoothingBuffer()
        animatedConfidence = 0f
        invalidate()
    }

    // ---------- 绘制 ----------

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val result = results ?: return
        if (result.faceLandmarks().isEmpty()) return

        // --- 计算 FIT_CENTER 映射 ---
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val imgW = imageWidth.toFloat()
        val imgH = imageHeight.toFloat()

        val scale = minOf(viewW / imgW, viewH / imgH)
        val offsetX = (viewW - imgW * scale) / 2f
        val offsetY = (viewH - imgH * scale) / 2f

        // 绘制人脸连接线
        drawFaceConnections(canvas, result, scale, offsetX, offsetY)

        // 绘制关键点
        drawLandmarks(canvas, result, scale, offsetX, offsetY)

        // 绘制所有情绪标签
        emotionResults?.forEachIndexed { index, emotion ->
            if (index < result.faceLandmarks().size) {
                drawEmotionLabel(canvas, result, index, emotion, scale, offsetX, offsetY)
            }
        }
    }

    private fun drawFaceConnections(
        canvas: Canvas,
        result: FaceLandmarkerResult,
        scale: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        result.faceLandmarks().forEach { landmarks ->
            // 绘制 MediaPipe 标准连接线
            FaceLandmarker.FACE_LANDMARKS_TESSELATION.forEach { connection ->
                val start = landmarks[connection.start()]
                val end = landmarks[connection.end()]
                canvas.drawLine(
                    start.x() * imageWidth * scale + offsetX,
                    start.y() * imageHeight * scale + offsetY,
                    end.x() * imageWidth * scale + offsetX,
                    end.y() * imageHeight * scale + offsetY,
                    connectionPaint
                )
            }

            // 绘制面部轮廓（更明显的外轮廓线）
            FaceLandmarker.FACE_LANDMARKS_FACE_OVAL.forEach { connection ->
                val start = landmarks[connection.start()]
                val end = landmarks[connection.end()]
                canvas.drawLine(
                    start.x() * imageWidth * scale + offsetX,
                    start.y() * imageHeight * scale + offsetY,
                    end.x() * imageWidth * scale + offsetX,
                    end.y() * imageHeight * scale + offsetY,
                    faceOutlinePaint
                )
            }
        }
    }

    private fun drawLandmarks(
        canvas: Canvas,
        result: FaceLandmarkerResult,
        scale: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        result.faceLandmarks().forEach { landmarks ->
            // 只绘制部分关键点（避免过于密集）
            val step = 4
            for (i in landmarks.indices step step) {
                val lm = landmarks[i]
                canvas.drawCircle(
                    lm.x() * imageWidth * scale + offsetX,
                    lm.y() * imageHeight * scale + offsetY,
                    2.5f,
                    landmarkPaint
                )
            }
        }
    }

    private fun drawEmotionLabel(
        canvas: Canvas,
        result: FaceLandmarkerResult,
        faceIndex: Int,
        emotion: EmotionClassifier.EmotionResult,
        scale: Float,
        offsetX: Float,
        offsetY: Float
    ) {
        val landmarks = result.faceLandmarks().getOrNull(faceIndex) ?: return
        if (landmarks.isEmpty()) return

        // 计算人脸边界框
        var minX = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var minY = Float.MAX_VALUE

        landmarks.forEach { lm ->
            val x = lm.x() * imageWidth * scale + offsetX
            val y = lm.y() * imageHeight * scale + offsetY
            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
        }

        val faceCenterX = (minX + maxX) / 2f
        val faceTopY = minY

        // 标签尺寸
        val labelWidth = 320f
        val labelHeight = 130f
        val labelRadius = 22f
        val marginTop = 60f

        val labelLeft = faceCenterX - labelWidth / 2f
        val labelTop = faceTopY - marginTop - labelHeight
        val labelRight = faceCenterX + labelWidth / 2f
        val labelBottom = faceTopY - marginTop

        // 防止超出屏幕
        val clampedTop = labelTop.coerceAtLeast(10f)
        val clampedBottom = clampedTop + labelHeight

        val rect = RectF(labelLeft, clampedTop, labelRight, clampedBottom)

        // 绘制渐变背景
        val emotionColor = Color.parseColor(emotion.emotion.colorHex)
        val darkerColor = darkenColor(emotionColor, 0.6f)
        val gradient = LinearGradient(
            labelLeft, clampedTop, labelRight, clampedBottom,
            intArrayOf(emotionColor, darkerColor),
            null,
            Shader.TileMode.CLAMP
        )
        labelBgPaint.shader = gradient
        labelBgPaint.alpha = 210
        canvas.drawRoundRect(rect, labelRadius, labelRadius, labelBgPaint)

        // 绘制情绪文字（emoji + 名称）
        val textY = clampedTop + labelHeight * 0.42f
        labelTextPaint.color = if (emotion.emotion == EmotionClassifier.Emotion.SURPRISED) {
            Color.parseColor("#333333")
        } else {
            Color.WHITE
        }
        canvas.drawText(
            "${emotion.emotion.emoji}  ${emotion.emotion.displayName}",
            faceCenterX,
            textY,
            labelTextPaint
        )

        // 绘制置信度文字
        val currentConfidence = if (faceIndex == 0) animatedConfidence else emotion.confidence
        val confText = "${(currentConfidence * 100).toInt()}%"
        subTextPaint.color = if (emotion.emotion == EmotionClassifier.Emotion.SURPRISED) {
            Color.parseColor("#555555")
        } else {
            Color.parseColor("#CCFFFFFF")
        }
        canvas.drawText(confText, faceCenterX, clampedTop + labelHeight * 0.72f, subTextPaint)

        // 绘制置信度进度条
        val barMarginH = 24f
        val barHeight = 6f
        val barTop = clampedTop + labelHeight * 0.85f
        val barLeft = labelLeft + barMarginH
        val barRight = labelRight - barMarginH
        val barRect = RectF(barLeft, barTop, barRight, barTop + barHeight)
        canvas.drawRoundRect(barRect, barHeight / 2f, barHeight / 2f, progressBgPaint)

        val progressRight = barLeft + (barRight - barLeft) * currentConfidence
        if (progressRight > barLeft) {
            val progressRect = RectF(barLeft, barTop, progressRight, barTop + barHeight)
            val progressGrad = LinearGradient(
                barLeft, barTop, progressRight, barTop,
                intArrayOf(Color.WHITE, Color.parseColor("#AAFFFFFF")),
                null,
                Shader.TileMode.CLAMP
            )
            progressFgPaint.shader = progressGrad
            canvas.drawRoundRect(progressRect, barHeight / 2f, barHeight / 2f, progressFgPaint)
        }
    }

    // ---------- 工具方法 ----------

    private fun darkenColor(color: Int, factor: Float): Int {
        val r = (Color.red(color) * factor).toInt().coerceIn(0, 255)
        val g = (Color.green(color) * factor).toInt().coerceIn(0, 255)
        val b = (Color.blue(color) * factor).toInt().coerceIn(0, 255)
        return Color.rgb(r, g, b)
    }
}
