package com.tsbir.gallery.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.tsbir.gallery.R
import com.tsbir.gallery.databinding.DialogRefineBinding
import java.io.File

class RefineBottomSheetDialog(
    private val rejectedItem: SearchResultItem,
    private val initialText: String,
    private val initialStrokes: List<SketchCanvasView.Stroke>,
    private val onSubmitRefinement: (modifiedText: String, modifiedSketchBitmap: Bitmap?, rejectedItem: SearchResultItem) -> Unit
) : BottomSheetDialogFragment() {

    private var _binding: DialogRefineBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = DialogRefineBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // 1. Load rejected photo thumbnail
        try {
            val file = File(rejectedItem.filePath)
            if (file.exists()) {
                val opts = BitmapFactory.Options().apply { inSampleSize = 4 }
                val bmp = BitmapFactory.decodeFile(rejectedItem.filePath, opts)
                binding.ivRejectedThumbnail.setImageBitmap(bmp)
            }
        } catch (e: Exception) {
            // Ignore thumbnail load failure
        }
        binding.tvRejectedLabel.text = "Rejected (will be excluded): ${File(rejectedItem.filePath).name}"

        // 2. Pre-fill text modification prompt with previous text
        binding.etRefineText.setText(initialText)

        // 3. Pre-load canvas with previous sketch strokes
        if (initialStrokes.isNotEmpty()) {
            binding.refineCanvas.loadStrokes(initialStrokes)
        }

        // 4. Canvas control buttons
        binding.btnRefineToggleEraser.setOnClickListener {
            val newMode = !binding.refineCanvas.isEraserMode
            binding.refineCanvas.isEraserMode = newMode
            binding.btnRefineToggleEraser.text = if (newMode) getString(R.string.action_draw) else getString(R.string.action_erase)
        }

        binding.btnRefineUndo.setOnClickListener {
            binding.refineCanvas.undo()
        }

        binding.btnRefineClear.setOnClickListener {
            binding.refineCanvas.clear()
        }

        // 5. Submit feedback
        binding.btnSubmitRefine.setOnClickListener {
            val modifiedText = binding.etRefineText.text?.toString()?.trim() ?: ""
            val modifiedSketch = if (binding.refineCanvas.hasDrawing()) {
                binding.refineCanvas.exportBitmap()
            } else {
                null
            }

            onSubmitRefinement(modifiedText, modifiedSketch, rejectedItem)
            dismiss()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
