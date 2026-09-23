"""
Unit Tests for CLIP-Finder2-style Android Gallery Indexing Lifecycle.

Tests:
  1. First-launch batch indexing (configurable batch size, e.g. 16).
  2. Launch-time reconciliation: zero embedding calls on unchanged relaunch (short-circuit).
  3. Incremental addition: embeds ONLY newly added photos.
  4. Deletion handling: prunes DB rows for photos deleted from the gallery.
"""

import os
import io
import sys
import unittest
import numpy as np
from PIL import Image

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mobile_gallery_demo as demo
from src.data.fscoco_dataset import FSCOCODataset


class TestIndexingLifecycle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_db = "test_lifecycle.db"
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        demo.GALLERY_DB_PATH = cls.test_db

        demo.initialize_server()
        cls.dataset = FSCOCODataset(root_dir="fscoco", split="test")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            try:
                os.remove(cls.test_db)
            except Exception:
                pass

    def test_indexing_lifecycle(self):
        # 1. Prepare initial photo set (10 photos)
        initial_indices = list(range(10))
        photo_records = []
        for idx in initial_indices:
            s = self.dataset.samples[idx]
            fn = os.path.basename(s['photo_path'])
            with open(s['photo_path'], 'rb') as f:
                img_bytes = f.read()
            mtime = 1000 + idx
            photo_records.append((fn, mtime, img_bytes))

        # Clear DB before test
        with demo.get_db() as conn:
            conn.execute("DELETE FROM gallery")
            conn.commit()

        # Step A: First-launch full index (batch size 4)
        print("\n--- Step A: First Launch Full Index ---")
        metrics_a = demo.reconcile_gallery(photo_records, batch_size=4)
        print(f"Metrics A: {metrics_a}")
        self.assertEqual(metrics_a['total_photos'], 10)
        self.assertEqual(metrics_a['new_indexed'], 10)
        self.assertEqual(metrics_a['unchanged_skipped'], 0)
        self.assertEqual(metrics_a['deleted_pruned'], 0)
        self.assertEqual(metrics_a['embedding_calls'], 10)

        # Step B: Unchanged relaunch (Short-circuit verification)
        print("\n--- Step B: Unchanged Relaunch Reconciliation ---")
        metrics_b = demo.reconcile_gallery(photo_records, batch_size=4)
        print(f"Metrics B: {metrics_b}")
        self.assertEqual(metrics_b['total_photos'], 10)
        self.assertEqual(metrics_b['new_indexed'], 0)
        self.assertEqual(metrics_b['unchanged_skipped'], 10)
        self.assertEqual(metrics_b['deleted_pruned'], 0)
        self.assertEqual(metrics_b['embedding_calls'], 0, "Unchanged relaunch must incur ZERO embedding calls!")

        # Step C: Add 2 photos and modify 1 existing photo
        print("\n--- Step C: Incremental Additions & Modifications ---")
        new_indices = [10, 11]
        for idx in new_indices:
            s = self.dataset.samples[idx]
            fn = os.path.basename(s['photo_path'])
            with open(s['photo_path'], 'rb') as f:
                img_bytes = f.read()
            photo_records.append((fn, 1000 + idx, img_bytes))

        # Modify photo 0 timestamp
        modified_rec = list(photo_records[0])
        modified_rec[1] = 9999
        photo_records[0] = tuple(modified_rec)

        metrics_c = demo.reconcile_gallery(photo_records, batch_size=4)
        print(f"Metrics C: {metrics_c}")
        self.assertEqual(metrics_c['total_photos'], 12)
        self.assertEqual(metrics_c['new_indexed'], 3, "2 new photos + 1 modified photo should be embedded")
        self.assertEqual(metrics_c['unchanged_skipped'], 9)
        self.assertEqual(metrics_c['embedding_calls'], 3)

        # Step D: Delete 2 photos from gallery (photos at indices 1 and 2)
        print("\n--- Step D: Deletion Pruning ---")
        deleted_filenames = {photo_records[1][0], photo_records[2][0]}
        photo_records_pruned = [r for r in photo_records if r[0] not in deleted_filenames]
        self.assertEqual(len(photo_records_pruned), 10)

        metrics_d = demo.reconcile_gallery(photo_records_pruned, batch_size=4)
        print(f"Metrics D: {metrics_d}")
        self.assertEqual(metrics_d['total_photos'], 10)
        self.assertEqual(metrics_d['deleted_pruned'], 2, "2 deleted photos must be pruned from SQLite")
        self.assertEqual(metrics_d['new_indexed'], 0)
        self.assertEqual(metrics_d['unchanged_skipped'], 10)
        self.assertEqual(metrics_d['embedding_calls'], 0)

        # Confirm deleted photos are absent from SQLite
        with demo.get_db() as conn:
            for fn in deleted_filenames:
                row = conn.execute("SELECT 1 FROM gallery WHERE filename = ?", (fn,)).fetchone()
                self.assertIsNone(row, f"Deleted photo '{fn}' must not exist in SQLite DB.")

        print("\n[LIFECYCLE TEST PASSED] Full index, zero-call relaunch, incremental add, and deletion pruning confirmed.")


if __name__ == '__main__':
    unittest.main()
