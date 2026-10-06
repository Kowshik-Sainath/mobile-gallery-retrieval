package com.tsbir.gallery.worker

import android.content.ContentUris
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.work.*
import com.tsbir.gallery.data.EmbeddingConverter
import com.tsbir.gallery.data.GalleryDatabase
import com.tsbir.gallery.data.GalleryEmbedding
import com.tsbir.gallery.ml.ModelManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.InputStream
import java.util.concurrent.TimeUnit

/**
 * GalleryIndexWorker — Background MediaStore indexing and reconciliation.
 *
 * Implements CLIP-Finder2 lifecycle:
 *   1. Full launch-time MediaStore sync via SQL diff.
 *   2. Zero-inference short-circuit on unchanged relaunch.
 *   3. Incremental encoding of new/modified photos in batches of 16.
 *   4. Pruning of deleted photos.
 */
class GalleryIndexWorker(
    private val context: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(context, workerParams) {

    companion object {
        private const val TAG = "GalleryIndexWorker"
        const val WORK_NAME_PERIODIC = "tsbir_periodic_index"
        const val WORK_NAME_ONESHOT = "tsbir_oneshot_index"
        const val BATCH_SIZE = 16

        fun schedulePeriodic(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiresBatteryNotLow(true)
                .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
                .build()

            val workRequest = PeriodicWorkRequestBuilder<GalleryIndexWorker>(
                repeatInterval = 1,
                repeatIntervalTimeUnit = TimeUnit.HOURS
            )
                .setConstraints(constraints)
                .addTag(WORK_NAME_PERIODIC)
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME_PERIODIC,
                ExistingPeriodicWorkPolicy.KEEP,
                workRequest
            )
        }

        fun enqueueOneShot(context: Context) {
            val workRequest = OneTimeWorkRequestBuilder<GalleryIndexWorker>()
                .addTag(WORK_NAME_ONESHOT)
                .build()

            WorkManager.getInstance(context).enqueueUniqueWork(
                WORK_NAME_ONESHOT,
                ExistingWorkPolicy.REPLACE,
                workRequest
            )
        }
    }

    data class MediaStorePhoto(
        val id: Long,
        val path: String,
        val dateModified: Long,
        val uri: Uri
    )

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        Log.i(TAG, "Starting MediaStore gallery reconciliation...")

        val db = GalleryDatabase.getInstance(context)
        val dao = db.galleryEmbeddingDao()
        val modelManager = ModelManager.getInstance(context)

        // 1. Query device MediaStore for all images
        val mediaStorePhotos = queryMediaStorePhotos()
        Log.i(TAG, "MediaStore contains ${mediaStorePhotos.size} photos.")

        // 2. Fetch existing database timestamps
        val existingEntries = dao.getAllModifiedTimestamps().associateBy({ it.mediaStoreId }, { it.lastModified })

        // 3. Prune deleted photos
        val currentIds = mediaStorePhotos.map { it.id }.toSet()
        val deletedIds = existingEntries.keys.filter { it !in currentIds }
        if (deletedIds.isNotEmpty()) {
            dao.deleteByMediaStoreIds(deletedIds)
            Log.i(TAG, "Pruned ${deletedIds.size} deleted photos from database.")
        }

        // 4. Identify new and modified photos
        val toIndex = mediaStorePhotos.filter { photo ->
            val existingModified = existingEntries[photo.id]
            existingModified == null || existingModified != photo.dateModified
        }

        if (toIndex.isEmpty()) {
            Log.i(TAG, "[Short-Circuit] All ${mediaStorePhotos.size} photos unchanged. Exactly 0 embedding calls made.")
            return@withContext Result.success()
        }

        Log.i(TAG, "Found ${toIndex.size} new/modified photos to index (batch size: $BATCH_SIZE).")

        // 5. Batch encode photos
        var indexedCount = 0
        val batchList = mutableListOf<GalleryEmbedding>()

        for (i in toIndex.indices) {
            // Deprioritize background indexing if user has an interactive search in flight
            SearchCoordinator.yieldIfSearchActive()

            val photo = toIndex[i]
            val bitmap = loadOptimizedBitmap(photo.uri)

            if (bitmap != null) {
                try {
                    val emb = modelManager.encodePhoto(bitmap)
                    val embBytes = EmbeddingConverter.floatArrayToBytes(emb)

                    batchList.add(
                        GalleryEmbedding(
                            mediaStoreId = photo.id,
                            filePath = photo.path,
                            embeddingBlob = embBytes,
                            lastModified = photo.dateModified,
                            indexedAt = System.currentTimeMillis()
                        )
                    )
                    indexedCount++
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to encode photo ID ${photo.id}: ${e.message}")
                } finally {
                    bitmap.recycle()
                }
            }

            // Flush batch to database
            if (batchList.size >= BATCH_SIZE || i == toIndex.size - 1) {
                if (batchList.isNotEmpty()) {
                    dao.insertAll(batchList)
                    batchList.clear()
                    Log.d(TAG, "Committed batch to database. Progress: $indexedCount / ${toIndex.size}")
                }
            }
        }

        val elapsedSec = (System.currentTimeMillis() - startTime) / 1000.0
        Log.i(TAG, "Reconciliation complete: indexed $indexedCount photos in ${elapsedSec}s (${indexedCount / maxOf(elapsedSec, 0.001)} photos/sec).")

        Result.success()
    }

    private fun queryMediaStorePhotos(): List<MediaStorePhoto> {
        val photos = mutableListOf<MediaStorePhoto>()
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DATA,
            MediaStore.Images.Media.DATE_MODIFIED
        )
        val sortOrder = "${MediaStore.Images.Media.DATE_MODIFIED} DESC"

        val cursor = context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            null,
            null,
            sortOrder
        )

        cursor?.use {
            val idCol = it.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val dataCol = it.getColumnIndexOrThrow(MediaStore.Images.Media.DATA)
            val dateCol = it.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_MODIFIED)

            while (it.moveToNext()) {
                val id = it.getLong(idCol)
                val path = it.getString(dataCol) ?: ""
                val dateModified = it.getLong(dateCol) * 1000L // convert to millis
                val contentUri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id)

                photos.add(MediaStorePhoto(id, path, dateModified, contentUri))
            }
        }

        return photos
    }

    private fun loadOptimizedBitmap(uri: Uri): Bitmap? {
        return try {
            // First decode bounds
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            var input: InputStream? = context.contentResolver.openInputStream(uri)
            BitmapFactory.decodeStream(input, null, options)
            input?.close()

            // Calculate sample size for 224x224 target
            val targetSize = 224
            var sampleSize = 1
            if (options.outHeight > targetSize || options.outWidth > targetSize) {
                val halfHeight = options.outHeight / 2
                val halfWidth = options.outWidth / 2
                while (halfHeight / sampleSize >= targetSize && halfWidth / sampleSize >= targetSize) {
                    sampleSize *= 2
                }
            }

            options.inJustDecodeBounds = false
            options.inSampleSize = sampleSize
            options.inPreferredConfig = Bitmap.Config.RGB_565

            input = context.contentResolver.openInputStream(uri)
            val bmp = BitmapFactory.decodeStream(input, null, options)
            input?.close()
            bmp
        } catch (e: Exception) {
            Log.w(TAG, "Error loading bitmap from uri $uri: ${e.message}")
            null
        }
    }
}
