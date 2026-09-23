package com.tsbir.gallery.data

import androidx.room.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * GalleryEmbedding — One row per indexed gallery photo.
 *
 * Stores MediaStore ID, file path, lastModified timestamp, and 512-D float32 embedding vector.
 */
@Entity(
    tableName = "gallery_embeddings",
    indices = [
        Index(value = ["mediaStoreId"], unique = true),
        Index(value = ["lastModified"])
    ]
)
data class GalleryEmbedding(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0L,

    @ColumnInfo(name = "mediaStoreId")
    val mediaStoreId: Long,

    @ColumnInfo(name = "filePath")
    val filePath: String,

    @ColumnInfo(name = "embeddingBlob", typeAffinity = ColumnInfo.BLOB)
    val embeddingBlob: ByteArray, // 512 * 4 bytes = 2048 bytes (float32)

    @ColumnInfo(name = "lastModified")
    val lastModified: Long,

    @ColumnInfo(name = "indexedAt")
    val indexedAt: Long = System.currentTimeMillis()
)

data class MediaStoreTimestamp(
    val mediaStoreId: Long,
    val lastModified: Long
)

@Dao
interface GalleryEmbeddingDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(embedding: GalleryEmbedding): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(embeddings: List<GalleryEmbedding>)

    @Query("SELECT * FROM gallery_embeddings")
    suspend fun getAll(): List<GalleryEmbedding>

    @Query("SELECT COUNT(*) FROM gallery_embeddings")
    suspend fun count(): Int

    @Query("SELECT mediaStoreId, lastModified FROM gallery_embeddings")
    suspend fun getAllModifiedTimestamps(): List<MediaStoreTimestamp>

    @Query("DELETE FROM gallery_embeddings WHERE mediaStoreId = :mediaStoreId")
    suspend fun deleteByMediaStoreId(mediaStoreId: Long)

    @Query("DELETE FROM gallery_embeddings WHERE mediaStoreId IN (:ids)")
    suspend fun deleteByMediaStoreIds(ids: List<Long>)

    @Query("DELETE FROM gallery_embeddings")
    suspend fun clearAll()
}

object EmbeddingConverter {
    fun floatArrayToBytes(floats: FloatArray): ByteArray {
        val bytes = ByteArray(floats.size * 4)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (f in floats) {
            buf.putFloat(f)
        }
        return bytes
    }

    fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val floats = FloatArray(bytes.size / 4)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (i in floats.indices) {
            floats[i] = buf.float
        }
        return floats
    }
}
