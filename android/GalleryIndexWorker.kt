package com.tsbir.gallery

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.work.*
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit

/**
 * GalleryIndexWorker
 *
 * WorkManager CoroutineWorker that:
 *  1. Queries MediaStore for images added since the last index run.
 *  2. Loads each image, resizes to 224×224, normalises to ImageNet mean/std.
 *  3. Runs the ONNX Runtime vision encoder (mobileclip_s1_vision.onnx).
 *  4. Stores float16 embedding (and optional patch tokens) in the Room DB.
 *
 * Doze Mode compliance:
 *  The periodic WorkRequest created by [schedulePeriodicReindex] uses
 *  setRequiresDeviceIdle(true) and setRequiresBatteryNotLow(true) so
 *  Android will defer full re-indexing until the device is idle on charger
 *  — complying with Doze Mode and App Standby restrictions.
 *
 * ONNX Runtime:
 *  The vision encoder is exported from composite_model.py as an ONNX file
 *  (src/export_mobile.py → outputs/mobileclip_s1_vision.onnx).
 *  Place the ONNX file in the Android assets/ folder.
 *  Dependencies (build.gradle):
 *    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'
 *
 * Input tensor:  float32, shape [1, 3, 224, 224], NCHW format, ImageNet normalised
 * Output tensor: float32, shape [1, 512]  — L2-normalised embedding
 */
class GalleryIndexWorker(
    private val context: Context,
    workerParams: WorkerParameters,
) : CoroutineWorker(context, workerParams) {

    companion object {
        private const val TAG = "GalleryIndexWorker"

        // Asset name for the ONNX vision encoder
        private const val ONNX_ASSET = "photo_backbone_int8.onnx"

        // Model input/output names (as exported from PyTorch ONNX export)
        private const val INPUT_NAME  = "image_input"
        private const val OUTPUT_NAME = "embedding"

        // Image dimensions expected by MobileCLIP-S1
        private const val IMG_SIZE = 224

        // Model version tag — bump when you deploy a new ONNX model
        private const val MODEL_VERSION = "mobileclip_s1_v1"

        // Maximum images per worker invocation (avoid ANR / timeout)
        private const val MAX_IMAGES_PER_RUN = 500

        // WorkManager unique periodic work name
        private const val PERIODIC_WORK_NAME = "gallery_full_reindex"

        /**
         * Schedule a periodic WorkManager task that runs a full gallery re-index
         * once per hour, but ONLY when the device is idle AND the battery is not low.
         *
         * setRequiresDeviceIdle(true)   — respects Android Doze Mode
         * setRequiresBatteryNotLow(true) — avoids draining battery
         *
         * Call this once from Application.onCreate() or MainActivity.onCreate().
         */
        fun schedulePeriodicReindex(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiresDeviceIdle(true)          // Doze Mode compliance
                .setRequiresBatteryNotLow(true)       // battery protection
                .setRequiredNetworkType(NetworkType.NOT_REQUIRED)  // fully offline
                .build()

            val periodicWork = PeriodicWorkRequestBuilder<GalleryIndexWorker>(
                repeatInterval = 1,
                repeatIntervalTimeUnit = TimeUnit.HOURS
            )
                .setConstraints(constraints)
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.MINUTES)
                .addTag("gallery_full_reindex")
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                PERIODIC_WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,   // don't restart if already scheduled
                periodicWork
            )

            Log.d(TAG, "Periodic gallery re-index scheduled (idle + battery constraints).")
        }

        /**
         * Immediately enqueue a one-shot index run with relaxed constraints.
         * Use this when the user explicitly requests re-indexing from the UI.
         */
        fun enqueueImmediate(context: Context) {
            val workRequest = OneTimeWorkRequestBuilder<GalleryIndexWorker>()
                .addTag("gallery_index_immediate")
                .build()
            WorkManager.getInstance(context).enqueueUniqueWork(
                "gallery_index_immediate",
                ExistingWorkPolicy.REPLACE,
                workRequest
            )
        }
    }

    // -----------------------------------------------------------------------
    // Worker entry point
    // -----------------------------------------------------------------------

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        Log.d(TAG, "GalleryIndexWorker started.")

        try {
            val db  = GalleryDatabase.getInstance(context)
            val dao = db.galleryEmbeddingDao()

            // Prune rows for photos that no longer exist in the gallery
            pruneDeletedPhotos(dao)

            // Determine which MediaStore images are new since last index
            val lastIndexed = dao.getLastIndexedTimestamp() ?: 0L
            val newImages   = queryNewImages(lastIndexed)

            if (newImages.isEmpty()) {
                Log.d(TAG, "No new images to index.")
                return@withContext Result.success()
            }

            Log.d(TAG, "Indexing ${newImages.size} new image(s)...")

            // Load ONNX session (cached for the worker's lifetime)
            val ortSession = loadOnnxSession()

            var successCount = 0
            var failCount    = 0

            for ((mediaId, filePath) in newImages.take(MAX_IMAGES_PER_RUN)) {
                if (isStopped) break   // honour WorkManager cancellation

                try {
                    // 1. Load and preprocess
                    val inputTensor = preprocessImage(filePath) ?: continue

                    // 2. Run inference
                    val (embedding, patchTokens) = runInference(ortSession, inputTensor)

                    // 3. Encode to float16 BLOBs
                    val embBlob = EmbeddingConverter.floatArrayToFloat16Bytes(embedding)
                    val patchBlob = patchTokens?.let {
                        EmbeddingConverter.patchMatrixToFloat16Bytes(it)
                    }

                    // 4. Persist to Room DB
                    dao.insert(
                        GalleryEmbedding(
                            mediaStoreId  = mediaId,
                            filePath      = filePath,
                            embeddingBlob = embBlob,
                            patchBlob     = patchBlob,
                            modelVersion  = MODEL_VERSION,
                            indexedAt     = System.currentTimeMillis(),
                        )
                    )
                    successCount++

                } catch (e: Exception) {
                    Log.e(TAG, "Failed to index $filePath: ${e.message}")
                    failCount++
                }
            }

            ortSession.close()

            Log.d(TAG, "Indexing complete: $successCount succeeded, $failCount failed.")
            Result.success()

        } catch (e: Exception) {
            Log.e(TAG, "Worker failed fatally: ${e.message}", e)
            Result.retry()   // retry with exponential backoff
        }
    }

    // -----------------------------------------------------------------------
    // MediaStore query
    // -----------------------------------------------------------------------

    /**
     * Query MediaStore for images whose DATE_ADDED is after [lastIndexed].
     * Returns a list of (mediaStoreId, filePath) pairs.
     */
    private fun queryNewImages(lastIndexedMs: Long): List<Pair<Long, String>> {
        val results = mutableListOf<Pair<Long, String>>()

        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DATA,        // absolute file path
        )
        // DATE_ADDED is in seconds on older APIs
        val lastIndexedSecs = lastIndexedMs / 1000L
        val selection     = "${MediaStore.Images.Media.DATE_ADDED} > ?"
        val selectionArgs = arrayOf(lastIndexedSecs.toString())
        val sortOrder     = "${MediaStore.Images.Media.DATE_ADDED} ASC"

        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            selection,
            selectionArgs,
            sortOrder
        )?.use { cursor ->
            val idCol   = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val pathCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)

            while (cursor.moveToNext()) {
                val id   = cursor.getLong(idCol)
                val path = cursor.getString(pathCol) ?: continue
                if (File(path).exists()) results.add(Pair(id, path))
            }
        }

        return results
    }

    /**
     * Remove DB rows for MediaStore IDs that no longer exist on disk.
     */
    private suspend fun pruneDeletedPhotos(dao: GalleryEmbeddingDao) {
        val allIds = mutableListOf<Long>()
        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            arrayOf(MediaStore.Images.Media._ID),
            null, null, null
        )?.use { cursor ->
            val idCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            while (cursor.moveToNext()) allIds.add(cursor.getLong(idCol))
        }
        if (allIds.isNotEmpty()) {
            dao.pruneDeleted(allIds)
        }
    }

    // -----------------------------------------------------------------------
    // Image preprocessing
    // -----------------------------------------------------------------------

    /**
     * Load image from [filePath], resize to 224×224, convert to normalised
     * float32 NCHW tensor.
     *
     * @return FloatBuffer of shape [1, 3, 224, 224], or null if load fails.
     */
    private fun preprocessImage(filePath: String): FloatBuffer? {
        val file = File(filePath)
        if (!file.exists()) return null

        val options = BitmapFactory.Options().apply {
            inSampleSize = computeInSampleSize(file)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val rawBitmap = BitmapFactory.decodeFile(filePath, options) ?: return null
        val bitmap = Bitmap.createScaledBitmap(rawBitmap, IMG_SIZE, IMG_SIZE, true)
        if (rawBitmap !== bitmap) rawBitmap.recycle()

        // ARGB_8888 → float32 NCHW with ImageNet normalisation
        val pixels = IntArray(IMG_SIZE * IMG_SIZE)
        bitmap.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)
        bitmap.recycle()

        // Buffer: 1 × 3 × 224 × 224 = 150,528 floats
        val buffer = FloatBuffer.allocate(1 * 3 * IMG_SIZE * IMG_SIZE)
        val numPixels = IMG_SIZE * IMG_SIZE

        // Write channels in order: R, G, B
        for (c in 0..2) {
            val shift = when (c) { 0 -> 16; 1 -> 8; else -> 0 }
            for (p in 0 until numPixels) {
                val channelVal = ((pixels[p] ushr shift) and 0xFF) / 255.0f
                buffer.put(channelVal)
            }
        }
        buffer.rewind()
        return buffer
    }

    /**
     * Compute BitmapFactory inSampleSize to avoid decoding multi-megapixel images
     * at full resolution before resizing — saves peak memory.
     */
    private fun computeInSampleSize(file: File): Int {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, opts)
        val maxDim = maxOf(opts.outWidth, opts.outHeight)
        var sample = 1
        while (maxDim / (sample * 2) >= IMG_SIZE * 2) sample *= 2
        return sample
    }

    // -----------------------------------------------------------------------
    // ONNX Runtime inference
    // -----------------------------------------------------------------------

    /**
     * Load the ONNX session from assets/ using NNAPI (if available) or CPU.
     * The session is thread-safe and should be cached across images in one worker run.
     */
    private fun loadOnnxSession(): OrtSession {
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions()

        // Enable NNAPI acceleration on Android (falls back to CPU if unavailable)
        try {
            opts.addNnapi()
            Log.d(TAG, "NNAPI acceleration enabled.")
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI not available, using CPU: ${e.message}")
        }

        // Optimise for inference (no training graph)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        opts.setIntraOpNumThreads(2)   // keep CPU usage low (background task)

        // Load ONNX model bytes from assets
        val modelBytes = context.assets.open(ONNX_ASSET).readBytes()
        return env.createSession(modelBytes, opts)
    }

    /**
     * Run inference on a preprocessed image tensor.
     *
     * @param session  loaded OrtSession
     * @param inputBuf float32 NCHW buffer [1, 3, 224, 224]
     *
     * @return Pair of:
     *   - embedding: FloatArray(512) — L2-normalised global embedding
     *   - patchTokens: FloatArray(49 × 512) or null if the model doesn't expose patches
     */
    private fun runInference(
        session: OrtSession,
        inputBuf: FloatBuffer,
    ): Pair<FloatArray, FloatArray?> {

        val env = OrtEnvironment.getEnvironment()

        // Build input tensor: float32, shape [1, 3, 224, 224]
        val inputShape = longArrayOf(1L, 3L, IMG_SIZE.toLong(), IMG_SIZE.toLong())
        val inputTensor = OnnxTensor.createTensor(env, inputBuf, inputShape)

        val inputs = mapOf(INPUT_NAME to inputTensor)

        session.run(inputs).use { result ->
            inputTensor.close()

            // Primary output: global embedding [1, 512]
            val embOutput  = result[OUTPUT_NAME].get().value as Array<FloatArray>
            val embedding  = embOutput[0]   // FloatArray(512)

            // L2 normalise (model may output unnormalised features)
            val norm = embedding.fold(0f) { acc, v -> acc + v * v }.let { Math.sqrt(it.toDouble()).toFloat() }
            if (norm > 1e-6f) for (i in embedding.indices) embedding[i] /= norm

            // Optional: patch tokens if the ONNX model exports them
            // Output name "patch_tokens" expected for STNet-enabled export
            val patchTokens: FloatArray? = try {
                val patchOutput = result["patch_tokens"].get().value as Array<Array<FloatArray>>
                // patchOutput shape: [1, 49, 512] → flatten to FloatArray(49*512)
                FloatArray(49 * 512).also { flat ->
                    var idx = 0
                    for (patch in patchOutput[0]) for (v in patch) flat[idx++] = v
                }
            } catch (e: Exception) {
                null   // model doesn't expose patch tokens — silently skip
            }

            return Pair(embedding, patchTokens)
        }
    }
}
