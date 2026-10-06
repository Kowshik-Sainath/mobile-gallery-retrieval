package com.tsbir.gallery.ml

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * ModelManager — Central ONNX Runtime coordinator for T+SBIR Mobile.
 *
 * Manages 5 INT8/FP32 ONNX sessions + Kotlin BPE Tokenizer:
 *   1. photoSession      : photo_backbone_int8.onnx      [image_input -> (1, 512)]
 *   2. sketchSession     : sketch_encoder_int8.onnx     [image_input/sketch_input -> (1, 512)]
 *   3. textSession       : text_encoder_int8.onnx        [token_ids -> (1, 512)]
 *   4. fusionSession     : composite_fusion_mobile.onnx  [sketch_embed, text_embed -> (1, 512)]
 *   5. combinerSession   : combiner_mobile.onnx          [shown_candidate, refined_query -> (1, 512)]
 *
 * Hardware Acceleration:
 *   Attempts NNAPI provider first; logs fallback to CPU/XNNPACK if unsupported.
 */
class ModelManager private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"
        const val EMBED_DIM = 512
        const val CONTEXT_LENGTH = 77

        @Volatile
        private var INSTANCE: ModelManager? = null

        fun getInstance(context: Context): ModelManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: ModelManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }

    val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    private var photoSession: OrtSession? = null
    private var sketchSession: OrtSession? = null
    private var textSession: OrtSession? = null
    private var fusionSession: OrtSession? = null
    private var combinerSession: OrtSession? = null

    var tokenizer: ClipTokenizer? = null
        private set

    var executionProvider: String = "Unknown"
        private set

    enum class ProviderPreference {
        AUTO_NNAPI,
        CPU_ONLY,
        XNNPACK
    }

    var providerPreference: ProviderPreference = ProviderPreference.AUTO_NNAPI
    var backgroundThreads: Int = 1
    var interactiveThreads: Int = 2

    val isLoaded: Boolean
        get() = photoSession != null && sketchSession != null && textSession != null &&
                fusionSession != null && combinerSession != null && tokenizer != null

    private fun createSessionForAsset(assetName: String, numThreads: Int): OrtSession {
        val modelBytes = readAssetBytes(assetName)

        if (providerPreference == ProviderPreference.AUTO_NNAPI) {
            try {
                val nnapiOpts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(numThreads)
                    addNnapi()
                }
                val session = env.createSession(modelBytes, nnapiOpts)
                executionProvider = "NNAPI (HW Accelerated)"
                Log.i(TAG, "Loaded $assetName (threads=$numThreads) with NNAPI.")
                return session
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI failed for $assetName (${e.message}), falling back to CPU.")
            }
        } else if (providerPreference == ProviderPreference.XNNPACK) {
            try {
                val xnnpackOpts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(numThreads)
                    addXnnpack(mapOf("intra_op_num_threads" to numThreads.toString()))
                }
                val session = env.createSession(modelBytes, xnnpackOpts)
                executionProvider = "XNNPACK"
                Log.i(TAG, "Loaded $assetName (threads=$numThreads) with XNNPACK.")
                return session
            } catch (e: Exception) {
                Log.w(TAG, "XNNPACK failed for $assetName (${e.message}), falling back to CPU.")
            }
        }

        // Clean CPU Fallback
        val cpuOpts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(numThreads)
        }
        val session = env.createSession(modelBytes, cpuOpts)
        if (executionProvider == "Unknown") {
            executionProvider = "CPU (threads=$numThreads)"
        }
        Log.i(TAG, "Loaded $assetName (threads=$numThreads) on CPU.")
        return session
    }

    @Synchronized
    fun ensurePhotoLoaded(threads: Int = backgroundThreads) {
        if (photoSession == null) {
            Log.i(TAG, "Lazy loading Photo Backbone (threads=$threads)...")
            photoSession = createSessionForAsset("photo_backbone_int8.onnx", threads)
        }
    }

    @Synchronized
    fun ensureSketchLoaded(threads: Int = interactiveThreads) {
        if (sketchSession == null) {
            Log.i(TAG, "Lazy loading Sketch Encoder (threads=$threads)...")
            sketchSession = createSessionForAsset("sketch_encoder_int8.onnx", threads)
        }
    }

    @Synchronized
    fun ensureTextLoaded(threads: Int = interactiveThreads) {
        if (textSession == null) {
            Log.i(TAG, "Lazy loading Text Encoder (threads=$threads)...")
            textSession = createSessionForAsset("text_encoder_int8.onnx", threads)
        }
        if (tokenizer == null) {
            Log.i(TAG, "Initializing CLIP Tokenizer...")
            val (vocabStream, isGz) = try {
                context.assets.open("bpe_simple_vocab_16e6.txt") to false
            } catch (e: Exception) {
                context.assets.open("bpe_simple_vocab_16e6.txt.gz") to true
            }
            tokenizer = ClipTokenizer(vocabStream, isGzipped = isGz)
        }
    }

    @Synchronized
    fun ensureFusionLoaded(threads: Int = interactiveThreads) {
        if (fusionSession == null) {
            Log.i(TAG, "Lazy loading Multimodal Fusion (threads=$threads)...")
            fusionSession = createSessionForAsset("composite_fusion_mobile.onnx", threads)
        }
    }

    @Synchronized
    fun ensureCombinerLoaded(threads: Int = interactiveThreads) {
        if (combinerSession == null) {
            Log.i(TAG, "Lazy loading Feedback Combiner (threads=$threads)...")
            combinerSession = createSessionForAsset("combiner_mobile.onnx", threads)
        }
    }

    @Synchronized
    fun resetInteractiveSessions() {
        try { sketchSession?.close() } catch (e: Exception) {}
        sketchSession = null
        try { textSession?.close() } catch (e: Exception) {}
        textSession = null
        try { fusionSession?.close() } catch (e: Exception) {}
        fusionSession = null
        try { combinerSession?.close() } catch (e: Exception) {}
        combinerSession = null
        executionProvider = "Unknown"
        Log.i(TAG, "Interactive search sessions closed and reset.")
    }

    @Synchronized
    fun resetAllSessions() {
        resetInteractiveSessions()
        try { photoSession?.close() } catch (e: Exception) {}
        photoSession = null
        Log.i(TAG, "All ONNX sessions closed and reset.")
    }

    /**
     * Warms up all 4 interactive search sessions (Text, Sketch, Fusion, Combiner)
     * in the background by loading them and executing one dummy forward pass.
     * This eliminates the ~4-5s cold compilation pause when the user taps search.
     */
    fun warmupSearchSessions() {
        try {
            Log.i(TAG, "Starting background search sessions warmup...")
            val t0 = System.nanoTime()

            // 1. Text Session + Tokenizer
            val tTx0 = System.nanoTime()
            ensureTextLoaded(interactiveThreads)
            encodeText("warmup query photo")
            val tText = (System.nanoTime() - tTx0) / 1_000_000
            Log.i(TAG, "Text session warmed up in ${tText}ms")

            // 2. Sketch Session
            val tSk0 = System.nanoTime()
            ensureSketchLoaded(interactiveThreads)
            val dummyBmp = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
            val sketchEmb = encodeSketch(dummyBmp)
            dummyBmp.recycle()
            val tSketch = (System.nanoTime() - tSk0) / 1_000_000
            Log.i(TAG, "Sketch session warmed up in ${tSketch}ms")

            // 3. Multimodal Fusion
            val tFu0 = System.nanoTime()
            ensureFusionLoaded(interactiveThreads)
            val dummyVec = FloatArray(EMBED_DIM)
            val fused = fuseComposite(sketchEmb, dummyVec)
            val tFusion = (System.nanoTime() - tFu0) / 1_000_000
            Log.i(TAG, "Fusion session warmed up in ${tFusion}ms")

            // 4. Feedback Combiner
            val tCb0 = System.nanoTime()
            ensureCombinerLoaded(interactiveThreads)
            refineFeedback(fused, fused)
            val tCombiner = (System.nanoTime() - tCb0) / 1_000_000
            Log.i(TAG, "Combiner session warmed up in ${tCombiner}ms")

            val total = (System.nanoTime() - t0) / 1_000_000
            Log.i(TAG, "All interactive search sessions warm and ready in ${total}ms (Provider: $executionProvider)")
        } catch (e: Exception) {
            Log.w(TAG, "Warmup error: ${e.message}", e)
        }
    }

    /**
     * Isolated single-configuration text encoder benchmark:
     * Measures Cold latency (session creation + first inference pass)
     * and Warm latency (mean of 3 subsequent inference passes).
     */
    fun benchmarkTextProvider(preference: ProviderPreference, threads: Int): Pair<Long, Long> {
        resetInteractiveSessions()
        System.gc()
        Thread.sleep(400)

        providerPreference = preference
        val tCold0 = System.nanoTime()
        ensureTextLoaded(threads)
        val dummyText = "a photo of a cat sitting on a couch"
        encodeText(dummyText)
        val coldMs = (System.nanoTime() - tCold0) / 1_000_000

        var warmSum = 0L
        val warmRuns = 3
        for (i in 0 until warmRuns) {
            val t0 = System.nanoTime()
            encodeText(dummyText)
            warmSum += (System.nanoTime() - t0) / 1_000_000
        }
        val warmMs = warmSum / warmRuns

        val detectedEp = executionProvider
        resetInteractiveSessions()
        System.gc()
        Thread.sleep(400)

        Log.i(TAG, "Benchmark [$preference, threads=$threads] -> EP: $detectedEp | Cold: ${coldMs}ms | Warm: ${warmMs}ms")
        return Pair(coldMs, warmMs)
    }

    private fun readAssetBytes(assetName: String): ByteArray {
        return context.assets.open(assetName).use { inputStream ->
            inputStream.readBytes()
        }
    }

    /**
     * Preprocesses a Bitmap to float32 (1, 3, 224, 224) in [0, 1] range (NCHW format).
     * Correct MobileCLIP normalization (NO ImageNet mean/std subtraction).
     */
    fun preprocessBitmap(bitmap: Bitmap): FloatBuffer {
        val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val pixels = IntArray(224 * 224)
        scaled.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        val buffer = ByteBuffer.allocateDirect(1 * 3 * 224 * 224 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        val rOffset = 0
        val gOffset = 224 * 224
        val bOffset = 2 * 224 * 224

        val rArray = FloatArray(224 * 224)
        val gArray = FloatArray(224 * 224)
        val bArray = FloatArray(224 * 224)

        for (i in pixels.indices) {
            val p = pixels[i]
            rArray[i] = ((p shr 16) and 0xFF) / 255.0f
            gArray[i] = ((p shr 8) and 0xFF) / 255.0f
            bArray[i] = (p and 0xFF) / 255.0f
        }

        buffer.put(rArray)
        buffer.put(gArray)
        buffer.put(bArray)
        buffer.rewind()

        if (scaled != bitmap) {
            scaled.recycle()
        }

        return buffer
    }

    /**
     * Encodes a gallery photo into a 512-D L2-normalized float32 embedding.
     */
    fun encodePhoto(bitmap: Bitmap): FloatArray {
        ensurePhotoLoaded()
        val session = photoSession ?: throw IllegalStateException("photoSession not loaded")
        val floatBuffer = preprocessBitmap(bitmap)
        val shape = longArrayOf(1, 3, 224, 224)
        val tensor = OnnxTensor.createTensor(env, floatBuffer, shape)

        val inputName = session.inputNames.iterator().next()
        val outputs = session.run(mapOf(inputName to tensor))
        tensor.close()

        val raw = (outputs[0].value as Array<FloatArray>)[0]
        outputs.close()
        return l2Normalize(raw)
    }

    /**
     * Encodes a sketch image into a 512-D L2-normalized float32 embedding.
     */
    fun encodeSketch(bitmap: Bitmap): FloatArray {
        ensureSketchLoaded()
        val session = sketchSession ?: throw IllegalStateException("sketchSession not loaded")
        val floatBuffer = preprocessBitmap(bitmap)
        val shape = longArrayOf(1, 3, 224, 224)
        val tensor = OnnxTensor.createTensor(env, floatBuffer, shape)

        val inputName = session.inputNames.iterator().next()
        val outputs = session.run(mapOf(inputName to tensor))
        tensor.close()

        val raw = (outputs[0].value as Array<FloatArray>)[0]
        outputs.close()
        return l2Normalize(raw)
    }

    /**
     * Encodes a text query string into a 512-D L2-normalized float32 embedding.
     */
    fun encodeText(text: String): FloatArray {
        ensureTextLoaded()
        val session = textSession ?: throw IllegalStateException("textSession not loaded")
        val tok = tokenizer ?: throw IllegalStateException("tokenizer not loaded")

        val tokens = tok.tokenize(text)
        val shape = longArrayOf(1, CONTEXT_LENGTH.toLong())
        val tensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), shape)

        val inputName = session.inputNames.iterator().next()
        val outputs = session.run(mapOf(inputName to tensor))
        tensor.close()

        val raw = (outputs[0].value as Array<FloatArray>)[0]
        outputs.close()
        return l2Normalize(raw)
    }

    /**
     * Multimodal Composite Fusion: Fuses sketch (512) and text (512) embeddings into a single query.
     */
    fun fuseComposite(sketchEmb: FloatArray, textEmb: FloatArray): FloatArray {
        ensureFusionLoaded()
        val session = fusionSession ?: throw IllegalStateException("fusionSession not loaded")

        val sketchTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(sketchEmb),
            longArrayOf(1, EMBED_DIM.toLong())
        )
        val textTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(textEmb),
            longArrayOf(1, EMBED_DIM.toLong())
        )

        val inputs = mapOf("sketch_embed" to sketchTensor, "text_embed" to textTensor)
        val outputs = session.run(inputs)
        sketchTensor.close()
        textTensor.close()

        val raw = (outputs[0].value as Array<FloatArray>)[0]
        outputs.close()
        return l2Normalize(raw)
    }

    /**
     * Feedback Combiner (CIRR-style):
     * Takes (shown_candidate_embed, refined_query_embed) -> outputs new composed query embedding.
     */
    fun refineFeedback(shownCandidateEmb: FloatArray, refinedQueryEmb: FloatArray): FloatArray {
        ensureCombinerLoaded()
        val session = combinerSession ?: throw IllegalStateException("combinerSession not loaded")

        val candidateTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(shownCandidateEmb),
            longArrayOf(1, EMBED_DIM.toLong())
        )
        val queryTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(refinedQueryEmb),
            longArrayOf(1, EMBED_DIM.toLong())
        )

        val inputs = mapOf("shown_candidate" to candidateTensor, "refined_query" to queryTensor)
        val outputs = session.run(inputs)
        candidateTensor.close()
        queryTensor.close()

        val raw = (outputs[0].value as Array<FloatArray>)[0]
        outputs.close()
        return l2Normalize(raw)
    }

    fun l2Normalize(v: FloatArray): FloatArray {
        var normSq = 0.0
        for (x in v) {
            normSq += (x * x)
        }
        val invNorm = 1.0f / (sqrt(normSq).toFloat() + 1e-12f)
        val res = FloatArray(v.size)
        for (i in v.indices) {
            res[i] = v[i] * invNorm
        }
        return res
    }

    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0.0f
        val len = minOf(a.size, b.size)
        for (i in 0 until len) {
            dot += a[i] * b[i]
        }
        return dot
    }
}
