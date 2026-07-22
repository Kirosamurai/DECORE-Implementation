"""Upload the DECORE-compressed VGG-16 to a Hugging Face model repo.

Usage:
    pip install huggingface_hub
    huggingface-cli login            # or pass --token / set HF_TOKEN
    python scripts/upload_hf.py --repo-id <username>/decore-vgg16-cifar10

Uploads:
    runs/pruned_vgg16_l50.pt   -> compressed model
    runs/baseline.pth          -> full baseline (for comparison)
    hf/model_card.md           -> repo README.md (model card)
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. <username>/decore-vgg16-cifar10")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or `huggingface-cli login`)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--pruned", default=os.path.join(ROOT, "runs", "pruned_vgg16_l50.pt"))
    ap.add_argument("--baseline", default=os.path.join(ROOT, "runs", "baseline.pth"))
    ap.add_argument("--card", default=os.path.join(ROOT, "hf", "model_card.md"))
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Install huggingface_hub first:  pip install huggingface_hub")

    for path in (args.pruned, args.baseline, args.card):
        if not os.path.exists(path):
            sys.exit(f"Missing file: {path}")

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    uploads = [
        (args.card, "README.md"),
        (args.pruned, "pruned_vgg16_l50.pt"),
        (args.baseline, "baseline.pth"),
    ]
    for local, remote in uploads:
        print(f"uploading {os.path.basename(local)} -> {remote}")
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=args.repo_id, repo_type="model")

    print(f"\nDone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
