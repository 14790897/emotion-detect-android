package com.emotiondetect

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.emotiondetect.databinding.ActivityPrivacyBinding

class PrivacyActivity : AppCompatActivity() {

    private lateinit var binding: ActivityPrivacyBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPrivacyBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        binding.toolbar.setNavigationOnClickListener { finish() }

        // 在线隐私政策链接
        binding.btnOnlinePrivacy.setOnClickListener {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://mygithub.14790897.xyz/Privacy-Policy/"))
            startActivity(intent)
        }

        // 设置隐私政策正文
        binding.tvPrivacyContent.text = """
            隐私政策
            
            最近更新日期：2024年10月
            
            本应用（以下简称“我们”）非常重视用户的隐私保护。本政策旨在说明我们如何处理您的个人信息。
            
            1. 权限使用说明
            本应用需要使用“摄像头”权限来实现实时情绪检测功能。这是应用的核心功能所必需的。
            
            2. 数据处理与存储
            - 图像处理：所有由摄像头捕获的图像仅在您的设备本地进行实时分析。
            - 本地处理：我们使用了 MediaPipe 和 ONNX 框架，所有 AI 推理过程均在您的手机 CPU/GPU 上完成。
            - 无上传：我们承诺不会将您的照片、视频或任何识别出的情绪数据上传到任何云端服务器或第三方机构。
            - 统计数据：情绪统计功能仅在您本地的设备上存储简单的计数信息。
            
            3. 照片保存
            当您点击拍照按钮时，合并了情绪标签的照片将保存到您设备的相册中。我们仅对您主动触发的保存行为进行写入操作。
            
            4. 权限撤回
            您可以随时在系统的“设置”中关闭摄像头权限，但这将导致应用无法进行检测。
            
            5. 第三方 SDK
            本应用使用了 Google MediaPipe 和 Microsoft ONNX Runtime 库，这些库同样在本地运行。
            
            6. 联系我们
            如果您对本政策有任何疑问，请联系开发者。
        """.trimIndent()
    }
}
