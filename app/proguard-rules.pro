# Default ProGuard rules file for the Android application
-keepattributes *Annotation*
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# --- MediaPipe Rules ---
-keep class com.google.mediapipe.** { *; }
-keep class com.google.android.libraries.mediapipe.** { *; }
# For JNI
-keepclassmembers class * {
    native <methods>;
}

# --- ONNX Runtime Rules ---
-keep class ai.onnxruntime.** { *; }
# Prevent obfuscation of JNI methods
-keepclassmembers class ai.onnxruntime.OnnxRuntime {
    native <methods>;
}

# --- CameraX Rules ---
-keep class androidx.camera.core.** { *; }
-dontwarn androidx.camera.core.**

# --- View Binding / Data Binding ---
-keep class com.emotiondetect.databinding.** { *; }

# --- Keep EmotionClassifier Enums (Used by name in SharedPreferences) ---
-keepclassmembers enum com.emotiondetect.EmotionClassifier$Emotion { *; }
