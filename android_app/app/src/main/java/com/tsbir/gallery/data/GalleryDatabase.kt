package com.tsbir.gallery.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [GalleryEmbedding::class],
    version = 1,
    exportSchema = false
)
abstract class GalleryDatabase : RoomDatabase() {

    abstract fun galleryEmbeddingDao(): GalleryEmbeddingDao

    companion object {
        private const val DB_NAME = "tsbir_gallery.db"

        @Volatile
        private var INSTANCE: GalleryDatabase? = null

        fun getInstance(context: Context): GalleryDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    GalleryDatabase::class.java,
                    DB_NAME
                )
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
