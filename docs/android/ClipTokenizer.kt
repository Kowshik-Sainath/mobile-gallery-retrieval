package com.mobilegallery.retrieval.tokenizer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.LongBuffer
import java.nio.charset.StandardCharsets
import java.util.regex.Pattern
import java.util.zip.GZIPInputStream

/**
 * Android-native CLIP BPE Tokenizer for MobileCLIP / OpenAI CLIP models.
 *
 * Implements token-for-token parity with open_clip / MobileCLIP:
 * - Vocabulary Size: 49,408
 * - Context Length: 77
 * - SOT (Start of Text) Token ID: 49406 (<start_of_text>)
 * - EOT (End of Text) Token ID: 49407 (<end_of_text>)
 * - PAD Token ID: 0
 *
 * Zero Python/C++ dependency: runs entirely in JVM/Kotlin on Android.
 */
class ClipTokenizer(vocabInputStream: InputStream, isGzipped: Boolean = true) {

    companion object {
        const val CONTEXT_LENGTH = 77
        const val SOT_TOKEN_ID = 49406L
        const val EOT_TOKEN_ID = 49407L
        const val PAD_TOKEN_ID = 0L

        private val REGEX_PATTERN = Pattern.compile(
            """<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+""",
            Pattern.CASE_INSENSITIVE
        )

        /**
         * Reversible bytes-to-unicode character mapping as specified in OpenAI CLIP / GPT-2.
         * Maps UTF-8 byte values (0..255) to readable unicode codepoints.
         */
        val BYTES_TO_UNICODE: Map<Int, Char> by lazy {
            val bs = mutableListOf<Int>()
            // '!'..'~'
            for (i in '!'.code..'~'.code) bs.add(i)
            // '¡'..'¬'
            for (i in '¡'.code..'¬'.code) bs.add(i)
            // '®'..'ÿ'
            for (i in '®'.code..'ÿ'.code) bs.add(i)

            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (!bs.contains(b)) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            bs.indices.associate { bs[it] to cs[it].toChar() }
        }
    }

    private val encoder: Map<String, Int>
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val cache = mutableMapOf<String, String>()

    init {
        val stream = if (isGzipped) GZIPInputStream(vocabInputStream) else vocabInputStream
        val reader = BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8))

        val merges = mutableListOf<Pair<String, String>>()
        var lineIndex = 0
        reader.forEachLine { line ->
            val trimmed = line.trim()
            if (lineIndex > 0 && trimmed.isNotEmpty() && lineIndex <= 49152 - 256 - 2) {
                val parts = trimmed.split(" ")
                if (parts.size == 2) {
                    merges.add(Pair(parts[0], parts[1]))
                }
            }
            lineIndex++
        }
        reader.close()

        val vocabList = mutableListOf<String>()
        val byteValues = BYTES_TO_UNICODE.values.map { it.toString() }
        vocabList.addAll(byteValues)
        vocabList.addAll(byteValues.map { "$it</w>" })

        for (merge in merges) {
            vocabList.add(merge.first + merge.second)
        }

        // Special tokens
        vocabList.add("<start_of_text>")
        vocabList.add("<end_of_text>")

        encoder = vocabList.mapIndexed { idx, token -> token to idx }.toMap()
        bpeRanks = merges.mapIndexed { idx, pair -> pair to idx }.toMap()
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        for (i in 0 until word.size - 1) {
            pairs.add(Pair(word[i], word[i + 1]))
        }
        return pairs
    }

    private fun bpe(token: String): String {
        cache[token]?.let { return it }

        if (token.isEmpty()) return ""

        var word = mutableListOf<String>()
        for (i in 0 until token.length - 1) {
            word.add(token[i].toString())
        }
        word.add(token.last().toString() + "</w>")

        var pairs = getPairs(word)
        if (pairs.isEmpty()) {
            val res = "$token</w>"
            cache[token] = res
            return res
        }

        while (true) {
            var minRank = Int.MAX_VALUE
            var bestBigram: Pair<String, String>? = null

            for (pair in pairs) {
                val rank = bpeRanks[pair] ?: Int.MAX_VALUE
                if (rank < minRank) {
                    minRank = rank
                    bestBigram = pair
                }
            }

            if (bestBigram == null || !bpeRanks.containsKey(bestBigram)) {
                break
            }

            val first = bestBigram.first
            val second = bestBigram.second
            val newWord = mutableListOf<String>()
            var i = 0

            while (i < word.size) {
                val j = word.subList(i, word.size).indexOf(first)
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                val actualJ = i + j
                newWord.addAll(word.subList(i, actualJ))
                i = actualJ

                if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }

            word = newWord
            if (word.size == 1) break
            pairs = getPairs(word)
        }

        val result = word.joinToString(" ")
        cache[token] = result
        return result
    }

    /**
     * Cleans whitespace and converts to lower case, matching open_clip default cleaning.
     */
    private fun cleanText(text: String): String {
        return text.trim().replace("\\s+".toRegex(), " ").lowercase()
    }

    /**
     * Tokenizes a single text string into a fixed-length LongArray of size 77.
     * Formats with SOT (49406) prefix, EOT (49407) suffix, and PAD (0) trailing zeros.
     */
    fun tokenize(text: String): LongArray {
        val cleaned = cleanText(text)
        val matcher = REGEX_PATTERN.matcher(cleaned)
        val tokenIds = mutableListOf<Long>()

        tokenIds.add(SOT_TOKEN_ID)

        while (matcher.find()) {
            val matchedToken = matcher.group()
            val utf8Bytes = matchedToken.toByteArray(StandardCharsets.UTF_8)
            val unicodeToken = buildString {
                for (b in utf8Bytes) {
                    val unsigned = b.toInt() and 0xFF
                    append(BYTES_TO_UNICODE[unsigned] ?: '?')
                }
            }

            val bpeResult = bpe(unicodeToken)
            for (subword in bpeResult.split(" ")) {
                if (subword.isNotEmpty()) {
                    encoder[subword]?.let { tokenIds.add(it.toLong()) }
                }
            }
        }

        tokenIds.add(EOT_TOKEN_ID)

        // Pad or truncate to CONTEXT_LENGTH (77)
        val result = LongArray(CONTEXT_LENGTH) { PAD_TOKEN_ID }
        if (tokenIds.size > CONTEXT_LENGTH) {
            for (i in 0 until CONTEXT_LENGTH - 1) {
                result[i] = tokenIds[i]
            }
            result[CONTEXT_LENGTH - 1] = EOT_TOKEN_ID
        } else {
            for (i in tokenIds.indices) {
                result[i] = tokenIds[i]
            }
        }

        return result
    }

    /**
     * Helper to create an ONNX Runtime tensor [1, 77] directly for inference.
     */
    fun createOnnxTensor(env: OrtEnvironment, text: String): OnnxTensor {
        val tokens = tokenize(text)
        val shape = longArrayOf(1, CONTEXT_LENGTH.toLong())
        val buffer = LongBuffer.wrap(tokens)
        return OnnxTensor.createTensor(env, buffer, shape)
    }
}
