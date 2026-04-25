package com.emotiondetect

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * FerEmotionClassifier — 基于 ONNX Runtime 的情绪分类器
 *
 * 模型来源：微软 ONNX Model Zoo — emotion-ferplus-8
 *   repo: onnxmodelzoo/emotion-ferplus-8
 *   file: emotion-ferplus-8.onnx
 *
 * 输入：1×1×64×64 灰度图（float32，归一化到 [0,1]）
 * 输出：1×8 向量，对应 8 种情绪（Softmax 归一化概率）
 *
 * 标签顺序（微软 FER+ 定义）：
 *   0: neutral   (中性)
 *   1: happiness (开心)
 *   2: surprise  (惊讶)
 *   3: sadness   (悲伤)
 *   4: anger     (愤怒)
 *   5: disgust   (厌恶)
 *   6: fear      (恐惧)
 *   7: contempt  (轻蔑)
 */
class FerEmotionClassifier(private val context: Context) {

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null

    /** ONNX 输入节点名称 */
    private var inputName: String = ""

    /** ONNX 输出节点名称 */
    private var outputName: String = ""

    // ---------- 情绪枚举（与模型输出索引一一对应）----------
    enum class Emotion(
        val index: Int,
        val displayName: String,
        val emoji: String,
        val colorHex: String
    ) {
        NEUTRAL(0, "中性", "😐", "#9E9E9E"),
        HAPPINESS(1, "开心", "😊", "#4CAF50"),
        SURPRISE(2, "惊讶", "😮", "#FFEB3B"),
        SADNESS(3, "悲伤", "😢", "#2196F3"),
        ANGER(4, "愤怒", "😠", "#F44336"),
        DISGUST(5, "厌恶", "🤢", "#9C27B0"),
        FEAR(6, "恐惧", "😨", "#607D8B"),
        CONTEMPT(7, "轻蔑", "😏", "#795548");

        companion object {
            fun fromIndex(index: Int): Emotion? = entries.find { it.index == index }
        }
    }

    data class Result(
        val emotion: Emotion,
        val confidence: Float,
        /** 8 维原始分数（Softmax 前） */
        val rawScores: FloatArray,
        /** 8 维 Softmax 归一化概率 */
        val probScores: FloatArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as Result
            if (emotion != other.emotion) return false
            if (confidence != other.confidence) return false
            if (!rawScores.contentEquals(other.rawScores)) return false
            if (!probScores.contentEquals(other.probScores)) return false
            return true
        }

        override fun hashCode(): Int {
            var result = emotion.hashCode()
            result = 31 * result + confidence.hashCode()
            result = 31 * result + rawScores.contentHashCode()
            result = 31 * result + probScores.contentHashCode()
            return result
        }
    }

    // ---------- 初始化 ----------

    fun initialize() {
        try {
            // 初始化 ONNX Runtime
            env = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions().apply {
                // 使用 CPU 推理
                setIntraOpNumThreads(2)
                setInterOpNumThreads(2)
            }

            // 加载 ONNX 模型
            val modelPath = "${context.filesDir.absolutePath}/$MODEL_FILENAME"

            // 如果 assets 中有模型，复制到 filesDir（ONNX Runtime Android 只支持从文件加载）
            copyAssetToFile(context, MODEL_FILENAME, modelPath)

            session = env!!.createSession(modelPath, sessionOptions)

            // 获取输入输出节点名称
            inputName = session!!.inputNames.iterator().next()
            outputName = session!!.outputNames.iterator().next()

            Log.d(TAG, "ONNX Runtime 初始化成功")
            Log.d(TAG, "输入节点: $inputName, 输出节点: $outputName")

        } catch (e: Exception) {
            Log.e(TAG, "ONNX Runtime 初始化失败", e)
        }
    }

    fun isReady(): Boolean = session != null

    // ---------- 推理 ----------

    /**
     * 对单张人脸 Bitmap 进行情绪分类
     *
     * @param faceBitmap 64×64 人脸图片（灰度或彩色均可）
     * @return 情绪结果
     */
    fun classify(faceBitmap: Bitmap): Result? {
        val session = session ?: return null
        val env = env ?: return null

        // 预处理：缩放到 64×64，转灰度 FloatBuffer
        val resized = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = preprocessToGrayscale(resized)

        try {
            val inputShape = longArrayOf(1, 1, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            val inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)

            val inputs = mapOf(inputName to inputTensor)

            // 执行推理
            val results = session.run(inputs)
            val outputTensor = results.get(0).value as Array<FloatArray>

            // 解析输出 1×8 (FER+ 模型输出的是原始分数，需要 Softmax)
            val rawScores = outputTensor[0].copyOf()

            // 执行 Softmax 归一化
            val expSum = rawScores.sumOf { kotlin.math.exp(it.toDouble()) }.toFloat()
            val probScores = rawScores.map { 
                kotlin.math.exp(it.toDouble()).toFloat() / expSum 
            }.toFloatArray()

            // 找最大概率
            var maxIdx = 0
            var maxProb = probScores[0]
            for (i in 1 until NUM_CLASSES) {
                if (probScores[i] > maxProb) {
                    maxProb = probScores[i]
                    maxIdx = i
                }
            }

            val emotion = Emotion.fromIndex(maxIdx) ?: return null
            return Result(
                emotion = emotion,
                confidence = maxProb,
                rawScores = rawScores,
                probScores = probScores
            )

        } catch (e: Exception) {
            Log.e(TAG, "ONNX 推理失败", e)
            return null
        }
    }

    /**
     * 将 Bitmap 预处理为 FloatBuffer
     * FER+ 模型期望输入范围为 [0, 255] 的灰度值，无须均值/标准差归一化
     */
    private fun preprocessToGrayscale(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val floatArray = FloatArray(INPUT_SIZE * INPUT_SIZE)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            
            // 使用标准灰度公式
            // 结果保持在 [0, 255] 范围，不进行 /255 归一化
            floatArray[i] = (0.299f * r + 0.587f * g + 0.114f * b)
        }
        return FloatBuffer.wrap(floatArray)
    }

    // ---------- 资源释放 ----------

    fun close() {
        try {
            session?.close()
            env?.close()
        } catch (e: Exception) {
            Log.e(TAG, "ONNX Runtime 关闭失败", e)
        }
        session = null
        env = null
    }

    companion object {
        private const val TAG = "FerEmotionClassifier"
        /** 模型文件名（需手动放入 assets 目录） */
        const val MODEL_FILENAME = "emotion-ferplus-8.onnx"
        const val INPUT_SIZE = 64
        private const val NUM_CLASSES = 8

        /**
         * 将 assets 中的文件复制到应用私有目录
         */
        private fun copyAssetToFile(context: Context, assetName: String, outPath: String) {
            val outFile = java.io.File(outPath)
            if (outFile.exists() && outFile.length() > 1000) {
                // 已存在，跳过复制
                Log.d(TAG, "模型已存在于 $outPath，跳过复制")
                return
            }

            context.assets.open(assetName).use { input ->
                java.io.FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "模型已复制到 $outPath，大小: ${outFile.length() / 1024 / 1024} MB")
        }
    }
}
