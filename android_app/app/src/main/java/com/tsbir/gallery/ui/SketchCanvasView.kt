package com.tsbir.gallery.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs

/**
 * QuickDraw-style interactive sketch canvas.
 *
 * Features:
 *   - Drawing & Eraser modes (pencil / eraser toggle)
 *   - Configurable stroke widths for drawing and erasing
 *   - Quadratic Bézier stroke smoothing for natural pen feel
 *   - Stroke history with Undo (pop last stroke) and Clear
 *   - Fast export to 224x224 RGB Bitmap for ONNX inference
 *   - Stroke serialization/cloning for compositional refinement dialogs
 */
class SketchCanvasView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    data class Stroke(
        val path: Path,
        val isEraser: Boolean,
        val strokeWidth: Float
    )

    private val strokeHistory = mutableListOf<Stroke>()
    private var currentPath: Path? = null
    private var currentIsEraser: Boolean = false
    private var prevX = 0f
    private var prevY = 0f

    var isEraserMode: Boolean = false
    var drawStrokeWidth: Float = 6f
    var eraserStrokeWidth: Float = 28f

    private val drawPaint = Paint().apply {
        color = Color.BLACK
        isAntiAlias = true
        isDither = true
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }

    private val eraserPaint = Paint().apply {
        color = Color.WHITE
        isAntiAlias = true
        isDither = true
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }

    private val backgroundPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    companion object {
        private const val TOUCH_TOLERANCE = 4f
    }

    init {
        setBackgroundColor(Color.WHITE)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), backgroundPaint)

        for (stroke in strokeHistory) {
            val paint = if (stroke.isEraser) eraserPaint else drawPaint
            paint.strokeWidth = stroke.strokeWidth
            canvas.drawPath(stroke.path, paint)
        }

        currentPath?.let { path ->
            val paint = if (currentIsEraser) eraserPaint else drawPaint
            paint.strokeWidth = if (currentIsEraser) eraserStrokeWidth else drawStrokeWidth
            canvas.drawPath(path, paint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                currentPath = Path().apply {
                    moveTo(x, y)
                }
                currentIsEraser = isEraserMode
                prevX = x
                prevY = y
                invalidate()
                return true
            }

            MotionEvent.ACTION_MOVE -> {
                val dx = abs(x - prevX)
                val dy = abs(y - prevY)
                if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
                    currentPath?.quadTo(prevX, prevY, (x + prevX) / 2, (y + prevY) / 2)
                    prevX = x
                    prevY = y
                    invalidate()
                }
                return true
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                currentPath?.let { path ->
                    path.lineTo(x, y)
                    strokeHistory.add(
                        Stroke(
                            path = path,
                            isEraser = currentIsEraser,
                            strokeWidth = if (currentIsEraser) eraserStrokeWidth else drawStrokeWidth
                        )
                    )
                }
                currentPath = null
                invalidate()
                parent?.requestDisallowInterceptTouchEvent(false)
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    fun undo() {
        if (strokeHistory.isNotEmpty()) {
            strokeHistory.removeAt(strokeHistory.size - 1)
            invalidate()
        }
    }

    fun clear() {
        strokeHistory.clear()
        currentPath = null
        invalidate()
    }

    fun hasDrawing(): Boolean {
        // Has drawing if there is at least one non-eraser stroke
        return strokeHistory.any { !it.isEraser }
    }

    fun cloneStrokes(): List<Stroke> {
        return strokeHistory.map {
            Stroke(Path(it.path), it.isEraser, it.strokeWidth)
        }
    }

    fun loadStrokes(strokes: List<Stroke>) {
        strokeHistory.clear()
        strokeHistory.addAll(strokes.map { Stroke(Path(it.path), it.isEraser, it.strokeWidth) })
        invalidate()
    }

    /**
     * Exports canvas contents as a 224x224 RGB Bitmap for MobileCLIP input.
     */
    fun exportBitmap(targetWidth: Int = 224, targetHeight: Int = 224): Bitmap {
        val bitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.WHITE)

        val w = if (width > 0) width.toFloat() else targetWidth.toFloat()
        val h = if (height > 0) height.toFloat() else targetHeight.toFloat()

        val scaleX = targetWidth / w
        val scaleY = targetHeight / h

        canvas.scale(scaleX, scaleY)

        for (stroke in strokeHistory) {
            val paint = if (stroke.isEraser) eraserPaint else drawPaint
            paint.strokeWidth = stroke.strokeWidth
            canvas.drawPath(stroke.path, paint)
        }

        return bitmap
    }
}
