"""
End-to-End HTTP Integration Test for T+SBIR Mobile ONNX Server.

Validates the complete deployment pipeline:
  1. Server initialization with 5 isolated ONNX sessions.
  2. Indexing 18 photos (3 ground-truth targets + 15 distractors) via /upload_photos.
  3. Multimodal search (/search) with real sketch + caption queries.
  4. Corrective feedback refinement (/refine) with candidate exclusion.
"""

import os
import io
import base64
import unittest
import numpy as np
from PIL import Image

import mobile_gallery_demo as demo
from src.data.fscoco_dataset import FSCOCODataset


class TestMobileDemoE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use an isolated test database
        cls.test_db = "test_gallery.db"
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        demo.GALLERY_DB_PATH = cls.test_db

        # Initialize ONNX sessions and database
        demo.initialize_server()
        cls.client = demo.app.test_client()

        # Load FS-COCO test split
        cls.dataset = FSCOCODataset(root_dir="fscoco", split="test")
        print(f"[TestSetup] Loaded {len(cls.dataset)} samples from FS-COCO test split.")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            try:
                os.remove(cls.test_db)
            except Exception:
                pass

    def test_01_clear_and_count(self):
        """Test gallery database clear and count endpoint."""
        res = self.client.post('/clear_gallery')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()['status'], 'cleared')

        res = self.client.get('/gallery_count')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()['count'], 0)

    def test_02_indexing_and_search(self):
        """Index 3 ground-truth photos + 15 distractors and perform multimodal search."""
        # Pick 3 target samples and 15 distractors
        target_indices = [10, 50, 90]
        distractor_indices = [i for i in range(100, 160, 4)][:15]

        all_indices = target_indices + distractor_indices
        photos_to_upload = []

        target_filenames = []
        for idx in target_indices:
            s = self.dataset.samples[idx]
            fn = os.path.basename(s['photo_path'])
            target_filenames.append(fn)

        print(f"\n[Test] Target photos to retrieve: {target_filenames}")

        # Prepare multipart upload
        data = {}
        for i, idx in enumerate(all_indices):
            s = self.dataset.samples[idx]
            fn = os.path.basename(s['photo_path'])
            with open(s['photo_path'], 'rb') as f:
                img_bytes = f.read()
            data[f'photos_{i}'] = (io.BytesIO(img_bytes), fn)

        # Upload files
        form_data = [('photos', (val[0], val[1])) for val in data.values()]
        res = self.client.post('/upload_photos', data={'photos': [val for val in data.values()]}, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        json_resp = res.get_json()
        print(f"[Test] Upload response: {json_resp}")
        self.assertEqual(json_resp['total_photos'], len(all_indices))

        # Test search for each of the 3 target queries
        found_in_top3_count = 0
        for i, idx in enumerate(target_indices):
            target_fn = target_filenames[i]
            s = self.dataset.samples[idx]
            sketch_path = s['sketch_path']
            with open(s['text_path'], 'r', encoding='utf-8') as f:
                caption = f.read().strip()

            with open(sketch_path, 'rb') as f:
                sketch_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')

            # Search with multimodal query (sketch + caption)
            search_res = self.client.post('/search', json={
                'sketch_b64': sketch_b64,
                'text_prompt': caption
            })
            self.assertEqual(search_res.status_code, 200)
            search_data = search_res.get_json()
            results = search_data['results']
            ranked_filenames = [r['filename'] for r in results]

            print(f"\n[Search Query {i+1}] Target: '{target_fn}' | Caption: '{caption[:50]}...'")
            print(f"  Returned top {len(ranked_filenames)}: {ranked_filenames[:5]}")

            self.assertIn(target_fn, ranked_filenames, f"Target photo {target_fn} should be in search results")
            rank = ranked_filenames.index(target_fn) + 1
            print(f"  Target rank: #{rank} (out of {len(all_indices)} photos in gallery)")
            if rank <= 3:
                found_in_top3_count += 1

        print(f"\n[Search Results] {found_in_top3_count}/3 queries ranked target in top-3!")
        self.assertGreaterEqual(found_in_top3_count, 2, "At least 2 out of 3 targets should rank in top-3.")

    def test_03_refinement_and_exclusion(self):
        """Test /refine endpoint: query composition combiner and candidate exclusion."""
        target_idx = 10
        s = self.dataset.samples[target_idx]
        target_fn = os.path.basename(s['photo_path'])

        with open(s['text_path'], 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        with open(s['sketch_path'], 'rb') as f:
            sketch_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')

        # Initial search
        search_res = self.client.post('/search', json={
            'sketch_b64': sketch_b64,
            'text_prompt': caption
        })
        results = search_res.get_json()['results']
        ranked_filenames = [r['filename'] for r in results]

        # Pick a distractor candidate to refine/reject
        distractor_fn = next(fn for fn in ranked_filenames if fn != target_fn)
        print(f"\n[Test Refine] Rejected photo candidate: '{distractor_fn}' | Looking for target: '{target_fn}'")

        refine_res = self.client.post('/refine', json={
            'shown_filename': distractor_fn,
            'sketch_b64': sketch_b64,
            'text_prompt': caption
        })
        self.assertEqual(refine_res.status_code, 200)
        refine_data = refine_res.get_json()

        # Gate 1: Excluded photo check
        self.assertEqual(refine_data['excluded_photo'], distractor_fn)
        refined_filenames = [r['filename'] for r in refine_data['results']]
        self.assertNotIn(distractor_fn, refined_filenames, f"Rejected photo '{distractor_fn}' must be strictly excluded from results!")

        # Gate 2: Target is present in refined results
        self.assertIn(target_fn, refined_filenames, f"Target photo '{target_fn}' must be present in refined results!")
        target_refined_rank = refined_filenames.index(target_fn) + 1
        print(f"[Test Refine] Target '{target_fn}' refined rank: #{target_refined_rank}")
        print(f"[Test Refine] Exclusion verification PASSED: '{distractor_fn}' was excluded.")


if __name__ == '__main__':
    unittest.main()
