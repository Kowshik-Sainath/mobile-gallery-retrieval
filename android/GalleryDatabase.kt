package com.tsbir.gallery

import android.content.Context
import androidx.room.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

// =============================================================================
// Entity
// =============================================================================

/**
 * GalleryEmbedding — one row per indexed gallery photo.
 *
 * Storage layout (float16 BLOBs):
 *   embeddingBlob  : 512 dims × 2 bytes = 1,024 bytes   (global feature, always stored)
 *   patchBlob      : 49 patches × 512 dims × 2 bytes = 50,176 bytes (optional, for STNet)
 *
 * At 10,000 gallery images:
 *   Global embeddings only : 10,000 × 1,024 bytes ≈  10 MB
 *   With patch tokens      : 10,000 × 51,200 bytes ≈ 512 MB  (store selectively)
 *
 * Decision: always store embeddingBlob; store patchBlob only for devices with
 * available storage > 1 GB (checked at index time in GalleryIndexWorker).
 */
@Entity(
    tableName = "gallery_embeddings",
    indices = [Index(value = ["mediaStoreId"], unique = true)]
)
data class GalleryEmbedding(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0L,

    /** MediaStore image ID — stable unique identifier for the photo. */
    @ColumnInfo(name = "mediaStoreId")
    val mediaStoreId: Long,

    /** Absolute file path at index time (may change if user moves file). */
    @ColumnInfo(name = "filePath")
    val filePath: String,

    /**
     * Global photo embedding in float16 (half precision).
     * Size: 512 × 2 = 1,024 bytes.
     * Use [EmbeddingConverter.float16BytesToFloatArray] to decode.
     */
    @ColumnInfo(name = "embeddingBlob", typeAffinity = ColumnInfo.BLOB)
    val embeddingBlob: ByteArray,

    /**
     * Optional patch token matrix in float16.
     * Shape: (49, 512) flattened = 49 × 512 × 2 = 50,176 bytes.
     * Null if device storage is constrained or if patchBlob was not requested.
     */
    @ColumnInfo(name = "patchBlob", typeAffinity = ColumnInfo.BLOB)
    val patchBlob: ByteArray?,

    /** Unix timestamp (ms) when this image was indexed. */
    @ColumnInfo(name = "indexedAt")
    val indexedAt: Long = System.currentTimeMillis(),

    /** Model version tag (e.g. "mobileclip_s1_v1") for cache invalidation. */
    @ColumnInfo(name = "modelVersion")
    val modelVersion: String = "mobileclip_s1_v1"
)

// =============================================================================
// DAO
// =============================================================================

@Dao
interface GalleryEmbeddingDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(embedding: GalleryEmbedding): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(embeddings: List<GalleryEmbedding>)

    @Query("SELECT * FROM gallery_embeddings WHERE mediaStoreId = :mediaStoreId LIMIT 1")
    suspend fun getByMediaStoreId(mediaStoreId: Long): GalleryEmbedding?

    /** Return all global embeddings (no patch BLOBs) for fast similarity search. */
    @Query("SELECT id, mediaStoreId, filePath, embeddingBlob, indexedAt, modelVersion FROM gallery_embeddings")
    suspend fun getAllGlobalEmbeddings(): List<GalleryEmbeddingGlobal>

    /** Return entries that need patch tokens but don't have them yet. */
    @Query("SELECT * FROM gallery_embeddings WHERE patchBlob IS NULL LIMIT :limit")
    suspend fun getMissingPatchBlobs(limit: Int = 100): List<GalleryEmbedding>

    @Query("SELECT COUNT(*) FROM gallery_embeddings")
    suspend fun count(): Int

    @Query("SELECT MAX(indexedAt) FROM gallery_embeddings")
    suspend fun getLastIndexedTimestamp(): Long?

    @Query("DELETE FROM gallery_embeddings WHERE mediaStoreId = :mediaStoreId")
    suspend fun deleteByMediaStoreId(mediaStoreId: Long)

    /** Remove entries for photos that no longer exist on disk. */
    @Query("DELETE FROM gallery_embeddings WHERE mediaStoreId NOT IN (:validIds)")
    suspend fun pruneDeleted(validIds: List<Long>)

    @Query("DELETE FROM gallery_embeddings WHERE modelVersion != :currentVersion")
    suspend fun pruneOldModelVersion(currentVersion: String)
}

/** Projection for fast global-only retrieval (omit patchBlob for bandwidth). */
data class GalleryEmbeddingGlobal(
    val id: Long,
    val mediaStoreId: Long,
    val filePath: String,
    val embeddingBlob: ByteArray,
    val indexedAt: Long,
    val modelVersion: String
)

// =============================================================================
// Database
// =============================================================================

@Database(
    entities = [GalleryEmbedding::class],
    version = 2,
    exportSchema = true
)
abstract class GalleryDatabase : RoomDatabase() {

    abstract fun galleryEmbeddingDao(): GalleryEmbeddingDao

    companion object {
        @Volatile
        private var INSTANCE: GalleryDatabase? = null

        fun getInstance(context: Context): GalleryDatabase {
            return INSTANCE ?: synchronized(this) {
                val db = Room.databaseBuilder(
                    context.applicationContext,
                    GalleryDatabase::class.java,
                    "gallery_embeddings.db"
                )
                    .fallbackToDestructiveMigration()   // re-index on schema change
                    .build()
                INSTANCE = db
                db
            }
        }
    }
}

// =============================================================================
// Float16 Encoding / Decoding Utilities
// =============================================================================

/**
 * EmbeddingConverter — converts between FloatArray and float16 ByteArray.
 *
 * float16 encoding saves 50% storage vs float32:
 *   512-dim float32 → 2,048 bytes
 *   512-dim float16 →  1,024 bytes
 *
 * Precision loss: float16 has ~3 decimal digits. For normalised unit vectors
 * (values in [-1, 1]), this is more than sufficient for cosine similarity ranking.
 *
 * The encoding follows the IEEE 754 half-precision standard:
 *   Sign:     1 bit
 *   Exponent: 5 bits (bias 15)
 *   Mantissa: 10 bits
 */
object EmbeddingConverter {

    /**
     * Encode a float32 embedding array to a compact float16 ByteArray.
     * @param floats normalised embedding values (typically in [-1.0, 1.0])
     * @return LITTLE_ENDIAN ByteArray of length floats.size * 2
     */
    fun floatArrayToFloat16Bytes(floats: FloatArray): ByteArray {
        val buf = ByteBuffer.allocate(floats.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (f in floats) {
            buf.putShort(floatToHalf(f))
        }
        return buf.array()
    }

    /**
     * Decode a float16 ByteArray back to a float32 FloatArray.
     * @param bytes LITTLE_ENDIAN float16 bytes
     * @return float32 FloatArray of length bytes.size / 2
     */
    fun float16BytesToFloatArray(bytes: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val result = FloatArray(bytes.size / 2)
        for (i in result.indices) {
            result[i] = halfToFloat(buf.short)
        }
        return result
    }

    /**
     * Encode a 2D patch token matrix (patches × dim) to float16 BLOB.
     * @param patches FloatArray of length (N_patches × embed_dim), row-major
     */
    fun patchMatrixToFloat16Bytes(patches: FloatArray): ByteArray =
        floatArrayToFloat16Bytes(patches)

    /**
     * Decode a patch BLOB back to FloatArray(N_patches × embed_dim).
     */
    fun float16BytesToPatchMatrix(bytes: ByteArray): FloatArray =
        float16BytesToFloatArray(bytes)

    // -----------------------------------------------------------------------
    // Internal IEEE 754 float16 conversion
    // -----------------------------------------------------------------------

    private fun floatToHalf(f: Float): Short {
        val bits = java.lang.Float.floatToRawIntBits(f)
        val sign     = (bits ushr 16) and 0x8000
        val exponent = ((bits ushr 23) and 0xFF) - 127 + 15
        val mantissa = (bits ushr 13) and 0x3FF

        return when {
            exponent <= 0  -> sign.toShort()                           // underflow → ±0
            exponent >= 31 -> (sign or 0x7C00).toShort()              // overflow → ±Inf
            else           -> (sign or (exponent shl 10) or mantissa).toShort()
        }
    }

    private fun halfToFloat(half: Short): Float {
        val h = half.toInt() and 0xFFFF
        val sign     = (h and 0x8000) shl 16
        val exponent = (h and 0x7C00) ushr 10
        val mantissa = (h and 0x03FF)

        val bits = when (exponent) {
            0    -> if (mantissa == 0) sign                           // ±0
                    else {
                        // Denormalised: shift mantissa until leading 1
                        var m = mantissa shl 1
                        var e = 0
                        while ((m and 0x400) == 0) { m = m shl 1; e++ }
                        sign or (((127 - 15 - e) + 1) shl 23) or ((m and 0x3FF) shl 13)
                    }
            31   -> sign or 0x7F800000 or (mantissa shl 13)          // Inf / NaN
            else -> sign or ((exponent + 127 - 15) shl 23) or (mantissa shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }
}
