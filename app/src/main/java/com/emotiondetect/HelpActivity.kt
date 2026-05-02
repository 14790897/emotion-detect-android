package com.emotiondetect

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.emotiondetect.databinding.ActivityHelpBinding

class HelpActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHelpBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHelpBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        
        binding.toolbar.setNavigationOnClickListener {
            finish()
        }

        binding.btnPrivacyPolicy.setOnClickListener {
            val intent = android.content.Intent(this, PrivacyActivity::class.java)
            startActivity(intent)
        }
    }
}
