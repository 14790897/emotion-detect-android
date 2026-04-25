package com.emotiondetect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * HseEmotionClassifier — 基于 HSEmotion (EfficientNet-B0) 的现代情绪分类器
 *
 * 输入：1×3×224×224 RGB图像
 * 归一化：ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 * 输出：1×8 向量
 */
class HseEmotionClassifier(private val context: Context) {

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    private var inputName: String = ""

    fun initialize() {
        Log.d(TAG, "开始初始化 HSEmotion...")
        try {
            env = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions()
            
            val modelPath = "${context.filesDir.absolutePath}/$MODEL_FILENAME"
            val dataPath = "$modelPath.data"

            // 1. 复制模型主文件
            copyAssetToFile(context, MODEL_FILENAME, modelPath)
            
            // 2. 尝试复制配套的 .data 文件 (如果 assets 中有的话)
            try {
                copyAssetToFile(context, "$MODEL_FILENAME.data", dataPath)
            } catch (e: Exception) {
                // 如果没有 .data 文件很正常，大部分模型不需要
            }

            val modelFile = java.io.File(modelPath)
            if (!modelFile.exists()) {
                Log.e(TAG, "致命错误：模型文件在磁盘上不存在！")
                return
            }
            Log.d(TAG, "模型就绪: $modelPath (${modelFile.length()} 字节)")

            Log.d(TAG, "正在创建 ONNX Session...")
            session = env!!.createSession(modelPath, sessionOptions)
            inputName = session!!.inputNames.iterator().next()
            
            Log.d(TAG, "HSEmotion 初始化成功！输入节点: $inputName")
        } catch (e: Exception) {
            Log.e(TAG, "HSEmotion 初始化发生异常", e)
            if (e.message?.contains(".data") == true) {
                Log.e(TAG, "检测到缺失 .data 权重文件。请确保下载了完整的权重数据。")
            }
        }
    }

    fun isReady(): Boolean = session != null

    fun classify(faceBitmap: Bitmap): FerEmotionClassifier.Result? {
        val session = session ?: return null
        val env = env ?: return null

        // 预处理：缩放到 224x224
        val resized = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = preprocessRGB(resized)

        try {
            val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            val inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)

            val results = session.run(mapOf(inputName to inputTensor))
            val outputTensor = results.get(0).value as Array<FloatArray>
            val rawScores = outputTensor[0].copyOf()

            // Softmax
            val expSum = rawScores.sumOf { kotlin.math.exp(it.toDouble()) }.toFloat()
            val probScores = rawScores.map { 
                kotlin.math.exp(it.toDouble()).toFloat() / expSum 
            }.toFloatArray()

            var maxIdx = 0
            var maxProb = probScores[0]
            for (i in 1 until probScores.size) {
                if (probScores[i] > maxProb) {
                    maxProb = probScores[i]
                    maxIdx = i
                }
            }

            // 获取 HSEmotion 专属的映射
            val emotion = mapHseIndexToEmotion(maxIdx)
            
            resized.recycle()
            
            return FerEmotionClassifier.Result(
                emotion = emotion,
                confidence = maxProb,
                rawScores = rawScores,
                probScores = probScores
            )
        } catch (e: Exception) {
            Log.e(TAG, "HSEmotion 推理失败", e)
            return null
        }
    }

    /**
     * HSEmotion (vgaf) 模型的标签映射通常是:
     * 0: anger, 1: contempt, 2: disgust, 3: fear, 4: happiness, 5: neutral, 6: sadness, 7: surprise
     * 我们需要映射到 FerEmotionClassifier.Emotion 枚举
     */
    private fun mapHseIndexToEmotion(hseIndex: Int): FerEmotionClassifier.Emotion {
        return when (hseIndex) {
            0 -> FerEmotionClassifier.Emotion.ANGER
            1 -> FerEmotionClassifier.Emotion.CONTEMPT
            2 -> FerEmotionClassifier.Emotion.DISGUST
            3 -> FerEmotionClassifier.Emotion.FEAR
            4 -> FerEmotionClassifier.Emotion.HAPPINESS
            5 -> FerEmotionClassifier.Emotion.NEUTRAL
            6 -> FerEmotionClassifier.Emotion.SADNESS
            7 -> FerEmotionClassifier.Emotion.SURPRISE
            else -> FerEmotionClassifier.Emotion.NEUTRAL
        }
    }

    private fun preprocessRGB(bitmap: Bitmap): FloatBuffer {
        val floatArray = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // ImageNet 归一化参数
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        // NCHW 格式: RRR...GGG...BBB...
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = Color.red(pixel) / 255f
            val g = Color.green(pixel) / 255f
            val b = Color.blue(pixel) / 255f

            floatArray[i] = (r - mean[0]) / std[0]
            floatArray[i + INPUT_SIZE * INPUT_SIZE] = (g - mean[1]) / std[1]
            floatArray[i + 2 * INPUT_SIZE * INPUT_SIZE] = (b - mean[2]) / std[2]
        }
        return FloatBuffer.wrap(floatArray)
    }

    fun close() {
        session?.close()
        env?.close()
        session = null
        env = null
    }

    private fun copyAssetToFile(context: Context, assetName: String, outPath: String) {
        val outFile = java.io.File(outPath)
        // 增加文件大小校验：如果文件存在但太小（说明复制不完整），则重新复制
        if (outFile.exists() && outFile.length() > 1000) {
            Log.d(TAG, "模型已存在: $outPath")
            return
        }
        
        try {
            Log.d(TAG, "正在从 assets 复制模型: $assetName -> $outPath")
            context.assets.open(assetName).use { input ->
                java.io.FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "模型复制完成，大小: ${outFile.length()} 字节")
        } catch (e: Exception) {
            Log.e(TAG, "复制模型失败，请确认 assets 目录下是否存在 $assetName", e)
        }
    }

    companion object {
        private const val TAG = "HseEmotionClassifier"
        const val MODEL_FILENAME = "enet_b0_8_best_vgaf.onnx"
        const val INPUT_SIZE = 224
    }
}
