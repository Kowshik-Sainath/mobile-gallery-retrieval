package com.tsbir.gallery.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.tsbir.gallery.databinding.ItemGalleryResultBinding
import kotlinx.coroutines.*
import java.io.File

data class SearchResultItem(
    val mediaStoreId: Long,
    val filePath: String,
    val score: Float,
    val embedding: FloatArray
)

class ResultsAdapter(
    private val onItemClicked: (SearchResultItem) -> Unit,
    private val onRejectClicked: (SearchResultItem) -> Unit
) : RecyclerView.Adapter<ResultsAdapter.ResultViewHolder>() {

    private val items = mutableListOf<SearchResultItem>()
    private val adapterScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    fun submitList(newItems: List<SearchResultItem>) {
        items.clear()
        items.addAll(newItems)
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ResultViewHolder {
        val binding = ItemGalleryResultBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ResultViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ResultViewHolder, position: Int) {
        holder.bind(items[position])
    }

    override fun getItemCount(): Int = items.size

    inner class ResultViewHolder(
        private val binding: ItemGalleryResultBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        private var loadJob: Job? = null

        fun bind(item: SearchResultItem) {
            binding.tvScore.text = String.format("%.3f", item.score)

            // Asynchronously load thumbnail
            loadJob?.cancel()
            binding.ivThumbnail.setImageDrawable(null)

            loadJob = adapterScope.launch {
                val bmp = withContext(Dispatchers.IO) {
                    loadThumbnail(item.filePath)
                }
                if (bmp != null) {
                    binding.ivThumbnail.setImageBitmap(bmp)
                }
            }

            binding.root.setOnClickListener {
                onItemClicked(item)
            }

            binding.btnReject.setOnClickListener {
                onRejectClicked(item)
            }
        }

        private fun loadThumbnail(path: String): Bitmap? {
            return try {
                val file = File(path)
                if (!file.exists()) return null

                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(path, options)

                val target = 250
                var sample = 1
                while (options.outHeight / sample > target && options.outWidth / sample > target) {
                    sample *= 2
                }

                options.inJustDecodeBounds = false
                options.inSampleSize = sample
                options.inPreferredConfig = Bitmap.Config.RGB_565

                BitmapFactory.decodeFile(path, options)
            } catch (e: Exception) {
                null
            }
        }
    }
}
