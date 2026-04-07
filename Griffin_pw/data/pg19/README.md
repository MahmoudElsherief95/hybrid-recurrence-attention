# PG-19 (official split) loader

This folder contains *code* to load the official-split PG-19 dataset (train/validation/test books + metadata.csv).

## Why byte-level?

PG-19 is released as open-vocabulary text. To make it usable with the existing integer-token models in this repo **without introducing external tokenizers**, this loader uses **byte-level language modeling**:

- tokens are bytes 0..255 mapped to ids 1..256
- id 0 is reserved for PAD (not used)
- vocab_size = 257

This preserves long-context structure and cleanly differentiates architectures, but the perplexity is **byte-level**, not word-level.

## Expected folder layout

Either of these layouts is supported:

- `<root>/train/train/*.txt` (Kaggle mirror zip)
- `<root>/train/*.txt`

And similarly for `validation` and `test`, plus `<root>/metadata.csv`.

## Extraction helper

The helper `ensure_pg19_extracted(cache_dir, zip_path=...)` can extract a zip (large; one-time) to `cache_dir/pg19_official/`.
