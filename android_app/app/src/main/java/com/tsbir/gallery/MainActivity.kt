package com.tsbir.gallery

import android.Manifest
import android.app.Dialog
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.GridLayoutManager
import com.google.android.material.tabs.TabLayout
import com.tsbir.gallery.data.EmbeddingConverter
import com.tsbir.gallery.data.GalleryDatabase
import com.tsbir.gallery.databinding.ActivityMainBinding
import com.tsbir.gallery.ml.ModelManager
import com.tsbir.gallery.ui.RefineBottomSheetDialog
import com.tsbir.gallery.ui.ResultsAdapter
import com.tsbir.gallery.ui.SearchResultItem
import com.tsbir.gallery.worker.GalleryIndexWorker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var resultsAdapter: ResultsAdapter
    private lateinit var modelManager: ModelManager
    private lateinit var database: GalleryDatabase

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Log.i(TAG, "Storage permission granted. Starting index...")
            GalleryIndexWorker.enqueueOneShot(this)
            monitorIndexingProgress()
        } else {
            Toast.makeText(this, "Permission required to index device photo gallery.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        modelManager = ModelManager.getInstance(this)
        database = GalleryDatabase.getInstance(this)

        setupToolbar()
        setupTabs()
        setupCanvasControls()
        setupRecyclerView()
        checkPermissionsAndStartIndexing()
    }

    private fun setupToolbar() {
        binding.tvEngineStatus.text = modelManager.executionProvider
    }

    private fun setupTabs() {
        binding.tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                when (tab?.position) {
                    0 -> { // Both (Multimodal)
                        binding.cardTextQuery.visibility = View.VISIBLE
                        binding.cardSketchCanvas.visibility = View.VISIBLE
                    }
                    1 -> { // Text Only
                        binding.cardTextQuery.visibility = View.VISIBLE
                        binding.cardSketchCanvas.visibility = View.GONE
                    }
                    2 -> { // Sketch Only
                        binding.cardTextQuery.visibility = View.GONE
                        binding.cardSketchCanvas.visibility = View.VISIBLE
                    }
                }
            }
            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })
    }

    private fun setupCanvasControls() {
        binding.btnToggleEraser.setOnClickListener {
            val isEraser = !binding.sketchCanvas.isEraserMode
            binding.sketchCanvas.isEraserMode = isEraser
            binding.btnToggleEraser.text = if (isEraser) getString(R.string.action_draw) else getString(R.string.action_erase)
        }

        binding.btnUndo.setOnClickListener {
            binding.sketchCanvas.undo()
        }

        binding.btnClear.setOnClickListener {
            binding.sketchCanvas.clear()
        }

        binding.btnSearch.setOnClickListener {
            executeSearch()
        }
    }

    private fun setupRecyclerView() {
        resultsAdapter = ResultsAdapter(
            onItemClicked = { item -> showFullPhotoDialog(item) },
            onRejectClicked = { item -> openRefineDialog(item) }
        )
        binding.rvResults.layoutManager = GridLayoutManager(this, 3)
        binding.rvResults.adapter = resultsAdapter
    }

    private fun checkPermissionsAndStartIndexing() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        if (ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Storage permission already granted.")
            GalleryIndexWorker.enqueueOneShot(this)
            monitorIndexingProgress()
        } else {
            permissionLauncher.launch(permission)
        }
    }

    private fun monitorIndexingProgress() {
        lifecycleScope.launch {
            while (true) {
                val count = withContext(Dispatchers.IO) {
                    database.galleryEmbeddingDao().count()
                }
                binding.tvIndexingBanner.text = "Gallery Index: $count photos indexed (Ready)"
                delay(3000)
            }
        }
    }

    private fun executeSearch() {
        val text = binding.etSearchText.text?.toString()?.trim() ?: ""
        val hasText = text.isNotBlank()
        val hasSketch = binding.sketchCanvas.hasDrawing()

        if (!hasText && !hasSketch) {
            Toast.makeText(this, "Please enter a text prompt or draw a sketch.", Toast.LENGTH_SHORT).show()
            return
        }

        binding.progressBar.visibility = View.VISIBLE
        val t0 = System.currentTimeMillis()

        lifecycleScope.launch {
            val queryEmb = withContext(Dispatchers.Default) {
                when {
                    hasText && hasSketch -> {
                        // Mode 3: Composite Fused Query
                        val sketchBmp = binding.sketchCanvas.exportBitmap()
                        val sEmb = modelManager.encodeSketch(sketchBmp)
                        val tEmb = modelManager.encodeText(text)
                        modelManager.fuseComposite(sEmb, tEmb)
                    }
                    hasText -> {
                        // Mode 1: Text-Only Query (Direct text encoder, no dummy sketch)
                        modelManager.encodeText(text)
                    }
                    else -> {
                        // Mode 2: Sketch-Only Query (Direct sketch encoder, no dummy text)
                        val sketchBmp = binding.sketchCanvas.exportBitmap()
                        modelManager.encodeSketch(sketchBmp)
                    }
                }
            }

            val results = withContext(Dispatchers.IO) {
                rankGallery(queryEmb, excludedMediaStoreId = null)
            }

            val elapsedMs = System.currentTimeMillis() - t0
            binding.progressBar.visibility = View.GONE
            binding.tvResultsHeader.text = "Results (${results.size} photos found in ${elapsedMs}ms):"
            resultsAdapter.submitList(results)
        }
    }

    private fun openRefineDialog(rejectedItem: SearchResultItem) {
        val currentText = binding.etSearchText.text?.toString()?.trim() ?: ""
        val currentStrokes = binding.sketchCanvas.cloneStrokes()

        val dialog = RefineBottomSheetDialog(
            rejectedItem = rejectedItem,
            initialText = currentText,
            initialStrokes = currentStrokes
        ) { modifiedText, modifiedSketchBitmap, rejected ->
            executeRefinement(modifiedText, modifiedSketchBitmap, rejected)
        }
        dialog.show(supportFragmentManager, "RefineDialog")
    }

    private fun executeRefinement(
        modifiedText: String,
        modifiedSketchBitmap: Bitmap?,
        rejectedItem: SearchResultItem
    ) {
        binding.progressBar.visibility = View.VISIBLE
        val t0 = System.currentTimeMillis()

        lifecycleScope.launch {
            val refinedQueryEmb = withContext(Dispatchers.Default) {
                val hasText = modifiedText.isNotBlank()
                val hasSketch = modifiedSketchBitmap != null

                val newQuery = when {
                    hasText && hasSketch -> {
                        val sEmb = modelManager.encodeSketch(modifiedSketchBitmap!!)
                        val tEmb = modelManager.encodeText(modifiedText)
                        modelManager.fuseComposite(sEmb, tEmb)
                    }
                    hasText -> modelManager.encodeText(modifiedText)
                    hasSketch -> modelManager.encodeSketch(modifiedSketchBitmap!!)
                    else -> rejectedItem.embedding.clone()
                }

                // Call Combiner ONNX: (shown_candidate_embed, refined_query_embed) -> combined_embed
                modelManager.refineFeedback(rejectedItem.embedding, newQuery)
            }

            val results = withContext(Dispatchers.IO) {
                // Re-rank full gallery and explicitly exclude the rejected candidate
                rankGallery(refinedQueryEmb, excludedMediaStoreId = rejectedItem.mediaStoreId)
            }

            val elapsedMs = System.currentTimeMillis() - t0
            binding.progressBar.visibility = View.GONE
            binding.tvResultsHeader.text = "Refined Results (${results.size} photos, excluded rejected, in ${elapsedMs}ms):"
            resultsAdapter.submitList(results)
            Toast.makeText(this@MainActivity, "Refinement applied! Candidate excluded.", Toast.LENGTH_SHORT).show()
        }
    }

    private suspend fun rankGallery(queryEmb: FloatArray, excludedMediaStoreId: Long?): List<SearchResultItem> {
        val allEntries = database.galleryEmbeddingDao().getAll()
        val scoredList = mutableListOf<SearchResultItem>()

        for (entry in allEntries) {
            if (excludedMediaStoreId != null && entry.mediaStoreId == excludedMediaStoreId) {
                // Explicitly exclude rejected photo from CIRR-style results
                continue
            }
            val emb = EmbeddingConverter.bytesToFloatArray(entry.embeddingBlob)
            val score = modelManager.cosineSimilarity(queryEmb, emb)
            scoredList.add(
                SearchResultItem(
                    mediaStoreId = entry.mediaStoreId,
                    filePath = entry.filePath,
                    score = score,
                    embedding = emb
                )
            )
        }

        // Sort descending by cosine similarity score
        scoredList.sortByDescending { it.score }
        return scoredList
    }

    private fun showFullPhotoDialog(item: SearchResultItem) {
        val dialog = Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen)
        dialog.setContentView(R.layout.dialog_photo_viewer)

        val ivFull = dialog.findViewById<ImageView>(R.id.ivFullPhoto)
        val tvPath = dialog.findViewById<TextView>(R.id.tvPhotoPath)

        tvPath.text = "${File(item.filePath).name} (Score: ${String.format("%.4f", item.score)})"
        try {
            val bmp = BitmapFactory.decodeFile(item.filePath)
            ivFull.setImageBitmap(bmp)
        } catch (e: Exception) {
            ivFull.setImageDrawable(null)
        }

        ivFull.setOnClickListener { dialog.dismiss() }
        dialog.show()
    }
}
