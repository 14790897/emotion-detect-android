package com.emotiondetect

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.emotiondetect.databinding.ActivityStatsBinding
import com.emotiondetect.databinding.ItemEmotionStatBinding
import java.text.SimpleDateFormat
import java.util.*

class StatsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityStatsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityStatsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        binding.toolbar.setNavigationOnClickListener { finish() }

        loadStats()

        binding.btnReset.setOnClickListener {
            resetStats()
        }
    }

    private fun loadStats() {
        binding.llStatsContainer.removeAllViews()
        val prefs = getSharedPreferences("emotion_stats", MODE_PRIVATE)
        val today = SimpleDateFormat("yyyyMMdd", Locale.getDefault()).format(Date())
        
        val stats = EmotionClassifier.Emotion.values().map { emotion ->
            val key = "${today}_${emotion.name}"
            emotion to prefs.getInt(key, 0)
        }.filter { it.second > 0 }.sortedByDescending { it.second }

        if (stats.isEmpty()) {
            binding.tvEmptyHint.visibility = View.VISIBLE
            return
        } else {
            binding.tvEmptyHint.visibility = View.GONE
        }

        val maxCount = stats.first().second

        stats.forEach { (emotion, count) ->
            val itemBinding = ItemEmotionStatBinding.inflate(LayoutInflater.from(this), binding.llStatsContainer, false)
            itemBinding.tvEmoji.text = emotion.emoji
            itemBinding.tvEmotionName.text = emotion.displayName
            itemBinding.tvCount.text = getString(R.string.stat_count_format, count)
            
            itemBinding.progressBar.max = maxCount
            itemBinding.progressBar.progress = count
            itemBinding.progressBar.setIndicatorColor(android.graphics.Color.parseColor(emotion.colorHex))
            
            binding.llStatsContainer.addView(itemBinding.root)
        }
    }

    private fun resetStats() {
        val prefs = getSharedPreferences("emotion_stats", MODE_PRIVATE)
        val today = SimpleDateFormat("yyyyMMdd", Locale.getDefault()).format(Date())
        val editor = prefs.edit()
        
        EmotionClassifier.Emotion.values().forEach { emotion ->
            editor.remove("${today}_${emotion.name}")
        }
        editor.apply()
        loadStats()
    }
}
