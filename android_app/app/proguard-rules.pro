# ProGuard rules for TSBIRGallery

# Keep Room database and entity classes
-keep class androidx.room.** { *; }
-dontwarn androidx.room.**
-keep class com.tsbir.gallery.data.** { *; }

# Keep ONNX Runtime native bindings
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Keep WorkManager worker
-keep class com.tsbir.gallery.worker.** { *; }
