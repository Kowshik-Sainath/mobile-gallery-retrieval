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

    init {
        loadAllSessions()
    }

    private fun createSessionOptions(): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        options.setIntraOpNumThreads(4)

        try {
            // Attempt NNAPI hardware acceleration on ARM NPU/DSP/GPU
            options.addNnapi()
            executionProvider = "NNAPI (Hardware Accelerated)"
            Log.i(TAG, "Hardware acceleration enabled: NNAPI")
        } catch (e: Exception) {
            executionProvider = "CPU / XNNPACK"
            Log.w(TAG, "NNAPI unavailable on this device, falling back to CPU/XNNPACK: ${e.message}")
        }

        return options
    }

    @Synchronized
    fun loadAllSessions() {
        if (photoSession != null) return

        val opts = createSessionOptions()

        try {
            Log.i(TAG, "Loading photo_backbone_int8.onnx...")
            photoSession = env.createSession(readAssetBytes("photo_backbone_int8.onnx"), opts)

            Log.i(TAG, "Loading sketch_encoder_int8.onnx...")
            sketchSession = env.createSession(readAssetBytes("sketch_encoder_int8.onnx"), opts)

            Log.i(TAG, "Loading text_encoder_int8.onnx...")
            textSession = env.createSession(readAssetBytes("text_encoder_int8.onnx"), opts)

            Log.i(TAG, "Loading composite_fusion_mobile.onnx...")
            fusionSession = env.createSession(readAssetBytes("composite_fusion_mobile.onnx"), opts)

            Log.i(TAG, "Loading combiner_mobile.onnx...")
            combinerSession = env.createSession(readAssetBytes("combiner_mobile.onnx"), opts)

            Log.i(TAG, "Initializing CLIP BPE Tokenizer...")
            val vocabStream = context.assets.open("bpe_simple_vocab_16e6.txt.gz")
            tokenizer = ClipTokenizer(vocabStream, isGzipped = true)

            Log.i(TAG, "All 5 ONNX components and Tokenizer loaded successfully. Provider: $executionProvider")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ONNX models", e)
            throw RuntimeException("Failed to initialize ONNX sessions: ${e.message}", e)
        }
    }

    private fun readAssetBytes(assetName: String): ByteArray {
        context.assets.open(assetName).use { inputStream ->
            val byteBuffer = ByteArrayOutputStream()
            val buffer = ByteArray(65536)
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                byteBuffer.write(buffer, 0, bytesRead)
            }
            return byteBuffer.toByteArray()
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
