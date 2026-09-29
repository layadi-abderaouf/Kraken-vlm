"""
Central paths and training defaults.

Set KRAKEN_DATA_ROOT to your dataset root, or pass --data-root to training scripts.
Expected layout (adjust folder names in this file if yours differ):

  $KRAKEN_DATA_ROOT/
    vg/                    # Visual Genome images
    coco/train2014/
    coco/val2014/
    annotations/           # JSONL files (objects, counting, vqa, ...)
"""

from __future__ import annotations

import os



# Hugging Face identifiers (used by dataset and training)
CLIP_MODEL = "openai/clip-vit-base-patch16"
BERT_MODEL = "google/bert_uncased_L-4_H-512_A-8"

# Checkpoint filenames (project root)
CKPT_QFORMER_S1 = "QformerS1.pth"
CKPT_QFORMER = "Qformer.pth"
CKPT_VLFM_S2 = "VLFM-S2-checkpoint.pth"
CKPT_VLFM = "VLFM.pth"
CKPT_MOD = "Mod-checkpoint.pth"
CKPT_SINGL = "large-checkpoint.pth"
CKPT_ROUTER = "router-checkpoint.pth"

