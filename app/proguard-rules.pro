# Default ProGuard rules file for the Android application
-keepattributes *Annotation*
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# --- MediaPipe Rules ---
-keep class com.google.mediapipe.** { *; }
-keep class com.google.android.libraries.mediapipe.** { *; }
-dontwarn com.google.mediapipe.proto.CalculatorProfileProto$CalculatorProfile
-dontwarn com.google.mediapipe.proto.GraphTemplateProto$CalculatorGraphTemplate

# MediaPipe needs to see the stack trace for some internal checks
-keepattributes Signature, InnerClasses, EnclosingMethod, AnnotationDefault, *Annotation*, SourceFile, LineNumberTable

# Prevent R8 from removing/optimizing code that MediaPipe relies on via reflection/JNI
-keepclassmembers class * {
    native <methods>;
}
-keep class com.google.mediapipe.framework.** { *; }
-keep class com.google.mediapipe.tasks.** { *; }
-keep class com.google.android.libraries.mediapipe.** { *; }


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
-keepclassmembers enum com.emotiondetect.FerEmotionClassifier$Emotion { *; }

# --- Additional JNI / Native Rules ---
-keepclasseswithmembernames class * {
    native <methods>;
}
