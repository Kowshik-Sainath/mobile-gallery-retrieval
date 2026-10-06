package com.tsbir.gallery.worker

import android.content.Context
import android.database.ContentObserver
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log

/**
 * MediaStoreObserver — ContentObserver on MediaStore.Images.Media.EXTERNAL_CONTENT_URI.
 *
 * Listens for system gallery changes (photos captured, imported, or deleted)
 * and schedules an expedited one-shot GalleryIndexWorker after a 3-second debounce.
 */
class MediaStoreObserver(private val context: Context) :
    ContentObserver(Handler(Looper.getMainLooper())) {

    companion object {
        private const val TAG = "MediaStoreObserver"
        private const val DEBOUNCE_DELAY_MS = 3000L
    }

    private val handler = Handler(Looper.getMainLooper())
    private val debounceRunnable = Runnable {
        Log.i(TAG, "Gallery change detected. Enqueuing one-shot indexing work...")
        GalleryIndexWorker.enqueueOneShot(context)
    }

    fun register() {
        try {
            context.contentResolver.registerContentObserver(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                true,
                this
            )
            Log.i(TAG, "MediaStoreObserver registered successfully.")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to register MediaStoreObserver: ${e.message}")
        }
    }

    fun unregister() {
        try {
            context.contentResolver.unregisterContentObserver(this)
            handler.removeCallbacks(debounceRunnable)
            Log.i(TAG, "MediaStoreObserver unregistered.")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to unregister MediaStoreObserver: ${e.message}")
        }
    }

    override fun onChange(selfChange: Boolean) = onChange(selfChange, null)

    override fun onChange(selfChange: Boolean, uri: Uri?) {
        handler.removeCallbacks(debounceRunnable)
        handler.postDelayed(debounceRunnable, DEBOUNCE_DELAY_MS)
    }
}
