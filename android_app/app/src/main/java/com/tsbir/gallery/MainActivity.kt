package com.tsbir.gallery

import android.Manifest
import android.app.Dialog
import android.content.Intent
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
import com.tsbir.gallery.worker.SearchCoordinator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.PriorityQueue

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val TOP_K = 50
    }

    private class MemoryGalleryIndex(
        val count: Int,
        val ids: LongArray,
        val paths: Array<String>,
        val embeddings: FloatArray // contiguous count * 512
    )

    private lateinit var binding: ActivityMainBinding
    private lateinit var resultsAdapter: ResultsAdapter
    private lateinit var modelManager: ModelManager
    private lateinit var database: GalleryDatabase

    @Volatile
    private var inMemoryIndex: MemoryGalleryIndex? = null
    private var lastTopKResults: List<SearchResultItem> = emptyList()

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

        // Background warmup for instant interactive search
        lifecycleScope.launch(Dispatchers.IO) {
            delay(1000) // allow UI to settle smoothly
            if (intent?.getBooleanExtra("BENCHMARK", false) != true) {
                modelManager.warmupSearchSessions()
                getOrBuildMemoryIndex() // Pre-load in-memory embeddings index into RAM
            }
        }

        if (intent?.getBooleanExtra("BENCHMARK", false) == true) {
            runBenchmarkSequence()
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        if (intent?.getBooleanExtra("BENCHMARK", false) == true) {
            runBenchmarkSequence()
        }
    }

    private fun setupToolbar() {
        binding.tvEngineStatus.text = "AI Ready (Lazy Loading)"
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

    private suspend fun getOrBuildMemoryIndex(): MemoryGalleryIndex = withContext(Dispatchers.IO) {
        val dbCount = database.galleryEmbeddingDao().count()
        val current = inMemoryIndex
        if (current != null && current.count == dbCount) {
            return@withContext current
        }
        val allEntries = database.galleryEmbeddingDao().getAll()
        val n = allEntries.size
        val ids = LongArray(n)
        val paths = Array(n) { "" }
        val flatEmbeddings = FloatArray(n * 512)

        for (i in 0 until n) {
            val entry = allEntries[i]
            ids[i] = entry.mediaStoreId
            paths[i] = entry.filePath
            val floats = EmbeddingConverter.bytesToFloatArray(entry.embeddingBlob)
            System.arraycopy(floats, 0, flatEmbeddings, i * 512, 512)
        }
        val built = MemoryGalleryIndex(n, ids, paths, flatEmbeddings)
        inMemoryIndex = built
        built
    }

    private suspend fun rankGalleryTopK(
        queryEmb: FloatArray,
        k: Int = TOP_K,
        excludedMediaStoreId: Long? = null
    ): Pair<List<SearchResultItem>, Long> = withContext(Dispatchers.Default) {
        val tScan0 = System.nanoTime()
        val index = getOrBuildMemoryIndex()
        val n = index.count
        if (n == 0) return@withContext Pair(emptyList(), (System.nanoTime() - tScan0) / 1_000_000)

        val minHeap = PriorityQueue<SearchResultItem>(k + 1, compareBy { it.score })
        val flat = index.embeddings

        for (i in 0 until n) {
            val id = index.ids[i]
            if (excludedMediaStoreId != null && id == excludedMediaStoreId) continue

            val offset = i * 512
            var dot = 0.0f
            for (d in 0 until 512) {
                dot += queryEmb[d] * flat[offset + d]
            }

            if (minHeap.size < k) {
                val embCopy = FloatArray(512)
                System.arraycopy(flat, offset, embCopy, 0, 512)
                minHeap.add(SearchResultItem(id, index.paths[i], dot, embCopy))
            } else if (dot > minHeap.peek()!!.score) {
                minHeap.poll()
                val embCopy = FloatArray(512)
                System.arraycopy(flat, offset, embCopy, 0, 512)
                minHeap.add(SearchResultItem(id, index.paths[i], dot, embCopy))
            }
        }

        val sortedResults = ArrayList<SearchResultItem>(minHeap.size)
        while (minHeap.isNotEmpty()) {
            sortedResults.add(minHeap.poll()!!)
        }
        sortedResults.reverse()
        val scanMs = (System.nanoTime() - tScan0) / 1_000_000
        Pair(sortedResults, scanMs)
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
        val tTotal0 = System.nanoTime()

        lifecycleScope.launch {
            // Signal coordinator to pause background indexing
            SearchCoordinator.isSearchActive = true
            try {
                var tTextMs = 0L
                var tSketchMs = 0L
                var tFusionMs = 0L
                var modeStr = "Unknown"

                val queryEmb = withContext(Dispatchers.Default) {
                    when {
                        hasText && hasSketch -> {
                            modeStr = "Composite (Sketch + Text)"
                            val tSk0 = System.nanoTime()
                            val sketchBmp = binding.sketchCanvas.exportBitmap()
                            val sEmb = modelManager.encodeSketch(sketchBmp)
                            tSketchMs = (System.nanoTime() - tSk0) / 1_000_000

                            val tTx0 = System.nanoTime()
                            val tEmb = modelManager.encodeText(text)
                            tTextMs = (System.nanoTime() - tTx0) / 1_000_000

                            val tFu0 = System.nanoTime()
                            val fused = modelManager.fuseComposite(sEmb, tEmb)
                            tFusionMs = (System.nanoTime() - tFu0) / 1_000_000
                            fused
                        }
                        hasText -> {
                            modeStr = "Text-Only"
                            val tTx0 = System.nanoTime()
                            val tEmb = modelManager.encodeText(text)
                            tTextMs = (System.nanoTime() - tTx0) / 1_000_000
                            tEmb
                        }
                        else -> {
                            modeStr = "Sketch-Only"
                            val tSk0 = System.nanoTime()
                            val sketchBmp = binding.sketchCanvas.exportBitmap()
                            val sEmb = modelManager.encodeSketch(sketchBmp)
                            tSketchMs = (System.nanoTime() - tSk0) / 1_000_000
                            sEmb
                        }
                    }
                }

                val (results, scanMs) = rankGalleryTopK(queryEmb, k = TOP_K, excludedMediaStoreId = null)
                lastTopKResults = results

                val tBind0 = System.nanoTime()
                resultsAdapter.submitList(results)
                val bindMs = (System.nanoTime() - tBind0) / 1_000_000
                val totalMs = (System.nanoTime() - tTotal0) / 1_000_000

                binding.progressBar.visibility = View.GONE
                binding.tvEngineStatus.text = modelManager.executionProvider
                binding.tvResultsHeader.text = "Results (Top ${results.size} in ${totalMs}ms | Scan: ${scanMs}ms):"

                // Log structured profiling table
                logProfilingTable(modeStr, tTextMs, tSketchMs, tFusionMs, scanMs, bindMs, totalMs, results.size)
            } finally {
                SearchCoordinator.isSearchActive = false
            }
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
        val tTotal0 = System.nanoTime()

        lifecycleScope.launch {
            SearchCoordinator.isSearchActive = true
            try {
                var tEncodeMs = 0L
                var tCombinerMs = 0L

                val refinedQueryEmb = withContext(Dispatchers.Default) {
                    val hasText = modifiedText.isNotBlank()
                    val hasSketch = modifiedSketchBitmap != null

                    val tEnc0 = System.nanoTime()
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
                    tEncodeMs = (System.nanoTime() - tEnc0) / 1_000_000

                    val tComb0 = System.nanoTime()
                    val combined = modelManager.refineFeedback(rejectedItem.embedding, newQuery)
                    tCombinerMs = (System.nanoTime() - tComb0) / 1_000_000
                    combined
                }

                // Scope refinement to Top-K candidates from initial search (Part K)
                val tRerank0 = System.nanoTime()
                val candidates = if (lastTopKResults.isNotEmpty()) lastTopKResults else emptyList()
                val results = if (candidates.isNotEmpty()) {
                    withContext(Dispatchers.Default) {
                        val reranked = mutableListOf<SearchResultItem>()
                        for (cand in candidates) {
                            if (cand.mediaStoreId == rejectedItem.mediaStoreId) continue
                            val score = modelManager.cosineSimilarity(refinedQueryEmb, cand.embedding)
                            reranked.add(cand.copy(score = score))
                        }
                        reranked.sortByDescending { it.score }
                        reranked
                    }
                } else {
                    rankGalleryTopK(refinedQueryEmb, k = TOP_K, excludedMediaStoreId = rejectedItem.mediaStoreId).first
                }
                val rerankMs = (System.nanoTime() - tRerank0) / 1_000_000
                lastTopKResults = results

                val totalMs = (System.nanoTime() - tTotal0) / 1_000_000
                binding.progressBar.visibility = View.GONE
                binding.tvEngineStatus.text = modelManager.executionProvider
                binding.tvResultsHeader.text = "Refined Results (Top ${results.size} in ${totalMs}ms | Rerank: ${rerankMs}ms):"
                resultsAdapter.submitList(results)
                Toast.makeText(this@MainActivity, "Refinement applied! Candidate excluded.", Toast.LENGTH_SHORT).show()

                Log.i(TAG, "=== REFINE PROFILING === Query Encode: ${tEncodeMs}ms | Combiner: ${tCombinerMs}ms | Top-K Rerank: ${rerankMs}ms | Total: ${totalMs}ms")
            } finally {
                SearchCoordinator.isSearchActive = false
            }
        }
    }

    private fun logProfilingTable(
        mode: String,
        tTextMs: Long,
        tSketchMs: Long,
        tFusionMs: Long,
        scanMs: Long,
        bindMs: Long,
        totalMs: Long,
        resultCount: Int
    ) {
        val sb = StringBuilder()
        sb.appendLine("\n================== TSBIR LATENCY BREAKDOWN (Mode: $mode) ==================")
        sb.appendLine(String.format("%-32s | %s", "Stage", "Latency (ms)"))
        sb.appendLine("-------------------------------------------------------------")
        if (tTextMs > 0) sb.appendLine(String.format("%-32s | %d ms", "Text Tokenize + ONNX INT8", tTextMs))
        if (tSketchMs > 0) sb.appendLine(String.format("%-32s | %d ms", "Sketch Preprocess + ONNX INT8", tSketchMs))
        if (tFusionMs > 0) sb.appendLine(String.format("%-32s | %d ms", "Multimodal Fusion ONNX", tFusionMs))
        sb.appendLine(String.format("%-32s | %d ms", "Matrix Vector Scan + Top-K", scanMs))
        sb.appendLine(String.format("%-32s | %d ms", "UI Adapter Submit", bindMs))
        sb.appendLine("-------------------------------------------------------------")
        sb.appendLine(String.format("%-32s | %d ms (Top-%d returned)", "Total End-to-End Latency", totalMs, resultCount))
        sb.appendLine("Provider: ${modelManager.executionProvider}")
        sb.appendLine("===========================================================================")
        Log.i("TSBIR_PROFILE", sb.toString())
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

    private fun runBenchmarkSequence() {
        lifecycleScope.launch(Dispatchers.IO) {
            SearchCoordinator.isSearchActive = true
            try {
                Log.i("TSBIR_BENCHMARK", "========== STARTING ON-DEVICE BENCHMARK SEQUENCE ==========")

                // Part 1: Execution Provider Comparison (Text Mode)
                Log.i("TSBIR_BENCHMARK", "Testing Provider 1: AUTO_NNAPI (2 threads)...")
                val (coldNnapi, warmNnapi) = modelManager.benchmarkTextProvider(ModelManager.ProviderPreference.AUTO_NNAPI, 2)

                Log.i("TSBIR_BENCHMARK", "Testing Provider 2: CPU_ONLY (2 threads)...")
                val (coldCpu2, warmCpu2) = modelManager.benchmarkTextProvider(ModelManager.ProviderPreference.CPU_ONLY, 2)

                Log.i("TSBIR_BENCHMARK", "Testing Provider 3: CPU_ONLY (4 threads)...")
                val (coldCpu4, warmCpu4) = modelManager.benchmarkTextProvider(ModelManager.ProviderPreference.CPU_ONLY, 4)

                Log.i("TSBIR_BENCHMARK", "Testing Provider 4: XNNPACK (2 threads)...")
                val (coldXnnpack, warmXnnpack) = try {
                    modelManager.benchmarkTextProvider(ModelManager.ProviderPreference.XNNPACK, 2)
                } catch (e: Exception) {
                    Log.w("TSBIR_BENCHMARK", "XNNPACK test failed: ${e.message}")
                    Pair(-1L, -1L)
                }

                val provReport = """
                    ================== EXECUTION PROVIDER COMPARISON (Text-Only) ==================
                    Provider Config                  | Cold Start (ms) | Warm / Steady (ms) | Notes
                    -----------------------------------------------------------------------------------------
                    AUTO_NNAPI (2 threads)           | ${coldNnapi} ms         | ${warmNnapi} ms           | HW Accelerated (Driver JIT on cold)
                    CPU_ONLY (2 threads)             | ${coldCpu2} ms         | ${warmCpu2} ms           | Predictable latency, no JIT compile
                    CPU_ONLY (4 threads)             | ${coldCpu4} ms         | ${warmCpu4} ms           | Higher CPU load / thermals
                    XNNPACK (2 threads)              | ${coldXnnpack} ms         | ${warmXnnpack} ms           | ${if (coldXnnpack > 0) "Optimized FP/INT kernels" else "Unsupported / Fallback"}
                    =========================================================================================
                """.trimIndent()
                Log.i("TSBIR_BENCHMARK", "\n" + provReport)

                // Part 2: Modalities Profiling on Active Provider (AUTO_NNAPI)
                Log.i("TSBIR_BENCHMARK", "Warming up interactive sessions on NNAPI...")
                modelManager.providerPreference = ModelManager.ProviderPreference.AUTO_NNAPI
                modelManager.interactiveThreads = 2
                modelManager.warmupSearchSessions()

                val index = getOrBuildMemoryIndex()
                Log.i("TSBIR_BENCHMARK", "In-memory index ready with ${index.count} photos.")

                val dummyBitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
                val dummyText = "photo of a dog running on the beach"

                // 1. Warm Text-only search
                val tTx0 = System.nanoTime()
                val textEmb = modelManager.encodeText(dummyText)
                val tTextEnc = (System.nanoTime() - tTx0) / 1_000_000
                val (textResults, tTextScan) = rankGalleryTopK(textEmb, k = TOP_K)
                val tTextTotal = tTextEnc + tTextScan

                // 2. Warm Sketch-only search
                val tSk0 = System.nanoTime()
                val sketchEmb = modelManager.encodeSketch(dummyBitmap)
                val tSketchEnc = (System.nanoTime() - tSk0) / 1_000_000
                val (sketchResults, tSketchScan) = rankGalleryTopK(sketchEmb, k = TOP_K)
                val tSketchTotal = tSketchEnc + tSketchScan

                // 3. Warm Composite search (Sketch + Text + Fusion)
                val tComp0 = System.nanoTime()
                val tSkEnc0 = System.nanoTime()
                val cSketchEmb = modelManager.encodeSketch(dummyBitmap)
                val tCompSketchEnc = (System.nanoTime() - tSkEnc0) / 1_000_000

                val tTxEnc0 = System.nanoTime()
                val cTextEmb = modelManager.encodeText(dummyText)
                val tCompTextEnc = (System.nanoTime() - tTxEnc0) / 1_000_000

                val tFu0 = System.nanoTime()
                val fusedEmb = modelManager.fuseComposite(cSketchEmb, cTextEmb)
                val tFusionMs = (System.nanoTime() - tFu0) / 1_000_000

                val (compResults, tCompScan) = rankGalleryTopK(fusedEmb, k = TOP_K)
                val tCompTotal = (System.nanoTime() - tComp0) / 1_000_000

                // 4. Warm Refinement (Combiner)
                val tRef0 = System.nanoTime()
                val shownCand = if (compResults.isNotEmpty()) compResults[0].embedding else FloatArray(512)
                val refinedEmb = modelManager.refineFeedback(shownCand, textEmb)
                val tCombinerMs = (System.nanoTime() - tRef0) / 1_000_000
                val tRerank0 = System.nanoTime()
                val reranked = compResults.drop(1).map {
                    it.copy(score = modelManager.cosineSimilarity(refinedEmb, it.embedding))
                }.sortedByDescending { it.score }
                val tRerankMs = (System.nanoTime() - tRerank0) / 1_000_000
                val tRefineTotal = tCombinerMs + tRerankMs

                val ratioStr = String.format("%.2f", tCompTotal.toDouble() / maxOf(1L, tTextTotal))
                val modalitiesReport = """
                    ================== STAGE-BY-STAGE MODALITY LATENCY PROFILING ==================
                    Search Modality       | Stage Breakdown                                 | Total (ms)
                    ---------------------------------------------------------------------------------
                    Text-Only             | Encode: ${tTextEnc}ms | Vector Scan: ${tTextScan}ms         | ${tTextTotal} ms
                    Sketch-Only           | Encode: ${tSketchEnc}ms | Vector Scan: ${tSketchScan}ms         | ${tSketchTotal} ms
                    Composite (Text+Sk)   | Sk: ${tCompSketchEnc}ms | Tx: ${tCompTextEnc}ms | Fusion: ${tFusionMs}ms | Scan: ${tCompScan}ms | ${tCompTotal} ms
                    Refine (Combiner)     | Combiner: ${tCombinerMs}ms | Scoped Top-K Rerank: ${tRerankMs}ms       | ${tRefineTotal} ms
                    =================================================================================
                    Composite vs Text Ratio: ${ratioStr}x (Historical unoptimized ratio was 23x)
                    Isolated Fusion Module Latency: ${tFusionMs} ms
                    =================================================================================
                """.trimIndent()
                Log.i("TSBIR_PROFILE", "\n" + modalitiesReport)
                dummyBitmap.recycle()
            } finally {
                SearchCoordinator.isSearchActive = false
            }
        }
    }
}
