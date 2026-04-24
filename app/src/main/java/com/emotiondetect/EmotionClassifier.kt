package com.emotiondetect

import com.google.mediapipe.tasks.components.containers.Category

/**
 * EmotionClassifier — 基于 Blendshapes 的情绪分类器
 *
 * 输入：MediaPipe 输出的 52 个面部表情系数 (Blendshapes)
 * 输出：EmotionResult，包含情绪枚举和置信度
 *
 * 分类规则：基于关键 Blendshape 系数的加权计算
 */
object EmotionClassifier {

    // ---------- 情绪枚举 ----------
    enum class Emotion(
        val displayName: String,
        val emoji: String,
        val colorHex: String
    ) {
        HAPPY("开心", "😊", "#4CAF50"),
        SAD("悲伤", "😢", "#2196F3"),
        SURPRISED("惊讶", "😮", "#FFEB3B"),
        ANGRY("愤怒", "😠", "#F44336"),
        DISGUSTED("厌恶", "🤢", "#9C27B0"),
        FEARFUL("恐惧", "😨", "#607D8B"),
        NEUTRAL("中性", "😐", "#FFFFFF")
    }

    data class EmotionResult(
        val emotion: Emotion,
        val confidence: Float,
        val scores: Map<Emotion, Float> = emptyMap()
    )

    // ---------- Blendshape 名称常量 ----------
    private const val BS_MOUTH_SMILE_LEFT = "mouthSmileLeft"
    private const val BS_MOUTH_SMILE_RIGHT = "mouthSmileRight"
    private const val BS_MOUTH_FROWN_LEFT = "mouthFrownLeft"
    private const val BS_MOUTH_FROWN_RIGHT = "mouthFrownRight"
    private const val BS_BROW_DOWN_LEFT = "browDownLeft"
    private const val BS_BROW_DOWN_RIGHT = "browDownRight"
    private const val BS_BROW_INNER_UP = "browInnerUp"
    private const val BS_JAW_OPEN = "jawOpen"
    private const val BS_EYE_WIDE_LEFT = "eyeWideLeft"
    private const val BS_EYE_WIDE_RIGHT = "eyeWideRight"
    private const val BS_NOSE_SNEER_LEFT = "noseSneerLeft"
    private const val BS_NOSE_SNEER_RIGHT = "noseSneerRight"
    private const val BS_CHEEK_SQUINT_LEFT = "cheekSquintLeft"
    private const val BS_CHEEK_SQUINT_RIGHT = "cheekSquintRight"
    private const val BS_EYE_SQUINT_LEFT = "eyeSquintLeft"
    private const val BS_EYE_SQUINT_RIGHT = "eyeSquintRight"
    private const val BS_MOUTH_UPPER_UP_LEFT = "mouthUpperUpLeft"
    private const val BS_MOUTH_UPPER_UP_RIGHT = "mouthUpperUpRight"
    private const val BS_BROW_OUTER_UP_LEFT = "browOuterUpLeft"
    private const val BS_BROW_OUTER_UP_RIGHT = "browOuterUpRight"

    // 平滑滤波历史记录（用于平滑情绪抖动）
    private val smoothingBuffer = ArrayDeque<Map<Emotion, Float>>(SMOOTHING_WINDOW)
    private const val SMOOTHING_WINDOW = 5

    /**
     * 主分类方法
     *
     * @param blendshapes MediaPipe 输出的 Blendshape 系数列表（每张脸）
     * @return 情绪结果，若无输入则返回 NEUTRAL
     */
    fun classify(blendshapes: List<List<Category>>): EmotionResult {
        if (blendshapes.isEmpty()) {
            clearSmoothingBuffer()
            return EmotionResult(Emotion.NEUTRAL, 0f)
        }

        // 取第一张脸的 Blendshapes
        val faceBlendshapes = blendshapes[0]
        val bsMap = faceBlendshapes.associate { it.categoryName() to it.score() }

        // 计算各情绪分数
        val rawScores = computeEmotionScores(bsMap)

        // 平滑处理
        val smoothed = applySmoothing(rawScores)

        // 找出得分最高情绪
        val topEntry = smoothed.maxByOrNull { it.value } ?: return EmotionResult(Emotion.NEUTRAL, 0f)

        return EmotionResult(
            emotion = topEntry.key,
            confidence = topEntry.value.coerceIn(0f, 1f),
            scores = smoothed
        )
    }

    /**
     * 计算各情绪原始分数（基于 Blendshape 加权公式）
     */
    private fun computeEmotionScores(bs: Map<String, Float>): Map<Emotion, Float> {
        fun get(name: String) = bs[name] ?: 0f

        // --- 开心 ---
        // 双侧嘴角上扬 + 眼部眯起（真诚笑容的杜乡标志）
        val happy = (
            get(BS_MOUTH_SMILE_LEFT) * 0.35f +
            get(BS_MOUTH_SMILE_RIGHT) * 0.35f +
            get(BS_CHEEK_SQUINT_LEFT) * 0.15f +
            get(BS_CHEEK_SQUINT_RIGHT) * 0.15f
        )

        // --- 悲伤 ---
        // 嘴角下垂 + 眉头聚拢上扬
        val sad = (
            get(BS_MOUTH_FROWN_LEFT) * 0.30f +
            get(BS_MOUTH_FROWN_RIGHT) * 0.30f +
            get(BS_BROW_INNER_UP) * 0.25f +
            (1f - get(BS_JAW_OPEN)) * 0.15f
        )

        // --- 惊讶 ---
        // 下巴张开 + 眼睛睁大 + 眉毛外侧上扬
        val surprised = (
            get(BS_JAW_OPEN) * 0.35f +
            get(BS_EYE_WIDE_LEFT) * 0.20f +
            get(BS_EYE_WIDE_RIGHT) * 0.20f +
            get(BS_BROW_OUTER_UP_LEFT) * 0.125f +
            get(BS_BROW_OUTER_UP_RIGHT) * 0.125f
        )

        // --- 愤怒 ---
        // 眉头下压 + 鼻子皱起 + 嘴角上扬（蔑视）
        val angry = (
            get(BS_BROW_DOWN_LEFT) * 0.30f +
            get(BS_BROW_DOWN_RIGHT) * 0.30f +
            get(BS_NOSE_SNEER_LEFT) * 0.20f +
            get(BS_NOSE_SNEER_RIGHT) * 0.20f
        )

        // --- 厌恶 ---
        // 鼻子皱起 + 上唇上扬
        val disgusted = (
            get(BS_NOSE_SNEER_LEFT) * 0.30f +
            get(BS_NOSE_SNEER_RIGHT) * 0.30f +
            get(BS_MOUTH_UPPER_UP_LEFT) * 0.20f +
            get(BS_MOUTH_UPPER_UP_RIGHT) * 0.20f
        )

        // --- 恐惧 ---
        // 眼睛睁大 + 眉头内侧上扬 + 下巴微张
        val fearful = (
            get(BS_EYE_WIDE_LEFT) * 0.25f +
            get(BS_EYE_WIDE_RIGHT) * 0.25f +
            get(BS_BROW_INNER_UP) * 0.30f +
            get(BS_JAW_OPEN) * 0.10f +
            get(BS_BROW_OUTER_UP_LEFT) * 0.05f +
            get(BS_BROW_OUTER_UP_RIGHT) * 0.05f
        )

        // --- 中性 ---
        // 其他情绪均低时为中性（基础分 + 各情绪负面贡献）
        val maxOther = maxOf(happy, sad, surprised, angry, disgusted, fearful)
        val neutral = (1f - maxOther * 1.5f).coerceAtLeast(0f)

        return mapOf(
            Emotion.HAPPY to happy,
            Emotion.SAD to sad,
            Emotion.SURPRISED to surprised,
            Emotion.ANGRY to angry,
            Emotion.DISGUSTED to disgusted,
            Emotion.FEARFUL to fearful,
            Emotion.NEUTRAL to neutral
        )
    }

    /**
     * 时间维度平滑：对最近 N 帧的分数取均值，减少抖动
     */
    private fun applySmoothing(current: Map<Emotion, Float>): Map<Emotion, Float> {
        if (smoothingBuffer.size >= SMOOTHING_WINDOW) {
            smoothingBuffer.removeFirst()
        }
        smoothingBuffer.addLast(current)

        // 各情绪分数的滑动均值
        return Emotion.values().associate { emotion ->
            val avg = smoothingBuffer.map { it[emotion] ?: 0f }.average().toFloat()
            emotion to avg
        }
    }

    fun clearSmoothingBuffer() {
        smoothingBuffer.clear()
    }
}
