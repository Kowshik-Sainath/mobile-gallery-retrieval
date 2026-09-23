"""
Tokenizer Parity Verification: Standalone CLIP BPE vs MobileCLIP / open_clip Tokenizer.

Verifies:
  1. 100% token-for-token identical output across 50 real FS-COCO test captions.
  2. Identical handling of SOT (49406), EOT (49407), and PAD (0) tokens.
  3. Fixed length output of exactly 77 tokens.
  4. Truncation and special character edge cases.
"""

import os
import sys
import gzip
import regex as re
import unittest
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mobileclip
from src.data.fscoco_dataset import FSCOCODataset


class StandaloneClipBpeTokenizer:
    """
    Pure standalone Python implementation mirroring docs/android/ClipTokenizer.kt.
    Loads bpe_simple_vocab_16e6.txt.gz directly, with zero MobileCLIP wrapper dependencies.
    """
    def __init__(self, bpe_path: str):
        self.context_length = 77
        self.sot_token_id = 49406
        self.eot_token_id = 49407
        self.pad_token_id = 0

        # Build bytes to unicode mapping
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(x) for x in cs]
        self.byte_encoder = dict(zip(bs, cs))

        # Read merges
        if bpe_path.endswith(".gz"):
            content = gzip.open(bpe_path, "rt", encoding="utf-8").read()
        else:
            content = open(bpe_path, "r", encoding="utf-8").read()

        lines = content.split("\n")
        merges = lines[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges if merge.strip()]

        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))

        special_tokens = ["<start_of_text>", "<end_of_text>"]
        vocab.extend(special_tokens)

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}

        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def _get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self._get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text: str) -> list[int]:
        cleaned = " ".join(text.strip().split()).lower()
        tokens = [self.sot_token_id]

        for token in re.findall(self.pat, cleaned):
            encoded_bytes = token.encode("utf-8")
            token_unicode = "".join(self.byte_encoder[b] for b in encoded_bytes)
            for subword in self.bpe(token_unicode).split(" "):
                if subword:
                    tokens.append(self.encoder[subword])

        tokens.append(self.eot_token_id)

        # Pad / truncate to 77
        if len(tokens) > self.context_length:
            tokens = tokens[:self.context_length]
            tokens[-1] = self.eot_token_id
        else:
            tokens.extend([self.pad_token_id] * (self.context_length - len(tokens)))

        return tokens


class TestTokenizerParity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import open_clip.tokenizer as octok
        bpe_path = octok.default_bpe()
        cls.standalone = StandaloneClipBpeTokenizer(bpe_path)
        cls.mobileclip_tok = mobileclip.get_tokenizer("mobileclip_s1")
        cls.dataset = FSCOCODataset(root_dir="fscoco", split="test")

    def test_50_real_fscoco_captions_parity(self):
        """Test token-for-token parity across 50 real FS-COCO test captions."""
        num_tested = 0
        total_tokens_checked = 0
        mismatches = []

        for i in range(min(50, len(self.dataset))):
            item = self.dataset[i]
            text = item["caption"]

            # Ground truth from MobileCLIP tokenizer
            gt_tensor = self.mobileclip_tok(text)  # shape (1, 77)
            gt_tokens = gt_tensor[0].tolist()

            # Standalone BPE tokenizer output
            standalone_tokens = self.standalone.tokenize(text)

            self.assertEqual(len(gt_tokens), 77)
            self.assertEqual(len(standalone_tokens), 77)

            if gt_tokens != standalone_tokens:
                mismatches.append({
                    "sample_idx": i,
                    "text": text,
                    "gt": gt_tokens,
                    "standalone": standalone_tokens,
                })
            else:
                num_tested += 1
                total_tokens_checked += 77

        print(f"\n[Tokenizer Parity] Evaluated {num_tested}/50 real captions.")
        print(f"[Tokenizer Parity] Total tokens verified: {total_tokens_checked} / {50 * 77} (100% match)")
        self.assertEqual(len(mismatches), 0, f"Found {len(mismatches)} mismatches: {mismatches}")

    def test_edge_cases(self):
        """Test edge cases: empty string, punctuation, numbers, and long sentences."""
        edge_cases = [
            "",
            "hello world",
            "a cat, a dog, and a bird sitting on a tree!",
            "123 456 7890 2026",
            "A fast black dog jumps over 10 lazy sleeping puppies at 4:30 PM.",
            "very long query " * 20,  # Forces truncation past 77 tokens
        ]

        for text in edge_cases:
            gt_tokens = self.mobileclip_tok(text)[0].tolist()
            standalone_tokens = self.standalone.tokenize(text)

            self.assertEqual(
                gt_tokens,
                standalone_tokens,
                f"Mismatch on edge case: '{text[:40]}...'\nGT: {gt_tokens}\nMine: {standalone_tokens}",
            )

        print("[Tokenizer Parity] All 6 edge cases matched 100% token-for-token.")


if __name__ == "__main__":
    unittest.main()
