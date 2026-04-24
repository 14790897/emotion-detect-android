import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.emotiondetect"
    compileSdk = 34

    val signingPropsFile = rootProject.file("app/signing.properties")
    val signingProps = Properties()
    val hasSigningProps = signingPropsFile.exists().also { exists ->
        if (exists) {
            signingPropsFile.inputStream().use(signingProps::load)
        }
    }

    val storeFileName = signingProps.getProperty("storeFile")
    val storePasswordValue = signingProps.getProperty("storePassword")
    val keyAliasValue = signingProps.getProperty("keyAlias")
    val keyPasswordValue = signingProps.getProperty("keyPassword")

    if (hasSigningProps) {
        check(!storeFileName.isNullOrBlank() && !storePasswordValue.isNullOrBlank() && !keyAliasValue.isNullOrBlank() && !keyPasswordValue.isNullOrBlank()) {
            "app/signing.properties is missing required keys: storeFile, storePassword, keyAlias, keyPassword"
        }
    }

    signingConfigs {
        if (hasSigningProps) {
            create("release") {
                storeFile = rootProject.file("app/$storeFileName")
                storePassword = storePasswordValue
                keyAlias = keyAliasValue
                keyPassword = keyPasswordValue
            }
        }
    }

    defaultConfig {
        applicationId = "com.emotiondetect"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            if (hasSigningProps) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    buildFeatures {
        viewBinding = true
    }

    // Exclude conflicting meta-inf files from MediaPipe
    packaging {
        resources {
            excludes += setOf(
                "META-INF/LICENSE",
                "META-INF/LICENSE.txt",
                "META-INF/NOTICE",
                "META-INF/NOTICE.txt",
                "META-INF/*.kotlin_module"
            )
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)

    // MediaPipe Face Landmarker
    implementation(libs.mediapipe.tasks.vision)

    // CameraX
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)
}
