package com.tsbir.gallery

import android.app.Application
import android.util.Log
import com.tsbir.gallery.ml.ModelManager
import com.tsbir.gallery.worker.GalleryIndexWorker
import com.tsbir.gallery.worker.MediaStoreObserver
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class TSBIRApplication : Application() {

    companion object {
        private const val TAG = "TSBIRApplication"
    }

    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var mediaStoreObserver: MediaStoreObserver? = null

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Initializing T+SBIR Mobile Gallery Application...")

        // Do not eagerly load all 5 ONNX models at startup (saves ~300MB RAM and thermal contention).
        // Models are loaded lazily on demand per modality when needed.

        // Schedule periodic background indexing (Doze-safe)
        GalleryIndexWorker.schedulePeriodic(this)

        // Register MediaStore push observer for real-time camera roll additions/deletions
        mediaStoreObserver = MediaStoreObserver(this).also { it.register() }
    }
}
