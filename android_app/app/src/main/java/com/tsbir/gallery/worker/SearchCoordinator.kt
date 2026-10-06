package com.tsbir.gallery.worker

import kotlinx.coroutines.delay

/**
 * Coordinates concurrency between interactive user searches in MainActivity
 * and background gallery indexing in GalleryIndexWorker.
 *
 * When an interactive query is executing, GalleryIndexWorker pauses to yield
 * CPU/NPU, thermal budget, and memory bandwidth to the foreground search.
 */
object SearchCoordinator {

    @Volatile
    var isSearchActive: Boolean = false

    /**
     * Called by GalleryIndexWorker between photo encodings.
     * If user is actively searching, yields execution until search completes.
     */
    suspend fun yieldIfSearchActive() {
        while (isSearchActive) {
            delay(100)
        }
    }
}
