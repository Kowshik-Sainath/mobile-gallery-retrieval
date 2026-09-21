package com.tsbir.gallery

import android.content.Context
import android.database.ContentObserver
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import androidx.work.*
import java.util.concurrent.TimeUnit

/**
 * GalleryContentObserver
 *
 * Registers a ContentObserver against MediaStore.Images.Media.EXTERNAL_CONTENT_URI.
 * When the user adds, deletes, or modifies photos in the gallery, Android delivers
 * an onChange() callback. We respond by enqueuing a GalleryIndexWorker job via
 * WorkManager — the battery and idle constraints are configured in the worker itself.
 *
 * Lifecycle:
 *   - Register in Application.onCreate() or a long-lived Service.
 *   - Unregister in Application.onTerminate() / Service.onDestroy().
 *
 * Usage (Kotlin):
 *   val observer = GalleryContentObserver(applicationContext)
 *   observer.register()
 *   // ... later ...
 *   observer.unregister()
 */
class GalleryContentObserver(private val context: Context) :
    ContentObserver(Handler(Looper.getMainLooper())) {

    companion object {
        private const val TAG = "GalleryContentObserver"
        private const val DEBOUNCE_DELAY_MS = 3_000L   // avoid rapid re-triggers
    }

    private val handler = Handler(Looper.getMainLooper())
    private val debounceRunnable = Runnable { enqueueIndexWork() }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /**
     * Register this observer against all external image changes.
     * Set descendantsInclude=true so sub-folder additions are captured.
     */
    fun register() {
        context.contentResolver.registerContentObserver(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            /* notifyForDescendants = */ true,
            this
        )
        Log.d(TAG, "ContentObserver registered for gallery changes.")
    }

    fun unregister() {
        context.contentResolver.unregisterContentObserver(this)
        handler.removeCallbacks(debounceRunnable)
        Log.d(TAG, "ContentObserver unregistered.")
    }

    // -----------------------------------------------------------------------
    // Callback
    // -----------------------------------------------------------------------

    override fun onChange(selfChange: Boolean) = onChange(selfChange, null)

    override fun onChange(selfChange: Boolean, uri: Uri?) {
        Log.d(TAG, "Gallery change detected (uri=$uri, selfChange=$selfChange)")
        // Debounce: wait DEBOUNCE_DELAY_MS after the last change before triggering
        handler.removeCallbacks(debounceRunnable)
        handler.postDelayed(debounceRunnable, DEBOUNCE_DELAY_MS)
    }

    // -----------------------------------------------------------------------
    // WorkManager enqueue
    // -----------------------------------------------------------------------

    /**
     * Enqueue a one-shot GalleryIndexWorker.
     *
     * Two flavours:
     *   A. User-triggered (immediate): no battery/idle constraint — the user just
     *      added a photo and expects it to be searchable soon.
     *   B. Background batch: uses Doze-safe constraints (idle + battery not low).
     *
     * Here we enqueue an EXPEDITED one-shot work for immediate indexing of the
     * new image, plus a periodic background worker (defined separately in
     * GalleryIndexWorker.schedulePeriodicReindex) for full re-scans.
     */
    private fun enqueueIndexWork() {
        val workRequest = OneTimeWorkRequestBuilder<GalleryIndexWorker>()
            .setExpedited(OutOfQuotaPolicy.RUN_AS_NON_EXPEDITED_WORK_REQUEST)
            .addTag("gallery_index_oneshot")
            .build()

        WorkManager.getInstance(context).enqueueUniqueWork(
            "gallery_index_oneshot",
            ExistingWorkPolicy.REPLACE,   // replace stale pending jobs
            workRequest
        )
        Log.d(TAG, "Enqueued one-shot GalleryIndexWorker (expedited).")
    }
}
