"""Create and push the STATIC (in-browser) Space from the CLI.

Everything runs client-side (ONNX Runtime Web), so this works on HF's free
Static Spaces — no PRO subscription needed.

Usage:
    pip install huggingface_hub
    hf auth login                       # or --token / HF_TOKEN
    python scripts/export_onnx.py               # produces web/models/*.onnx
    python scripts/deploy_static_space.py --space-id <username>/decore-vgg16

Uploads the entire web/ folder (index.html, app.js, style.css, models/, examples/)
to a static Space.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-id", required=True, help="e.g. <username>/decore-vgg16")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    for f in ("index.html", "app.js", "style.css",
              "models/baseline.onnx", "models/pruned.onnx", "models/pruned.int8.onnx"):
        if not os.path.exists(os.path.join(WEB, f)):
            sys.exit(f"Missing {f} — run `python scripts/export_onnx.py` first.")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Install huggingface_hub first:  pip install huggingface_hub")

    api = HfApi(token=args.token)
    api.create_repo(args.space_id, repo_type="space", space_sdk="static",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=WEB, repo_id=args.space_id, repo_type="space",
                      ignore_patterns=["__pycache__/*", "*.pyc"])

    print(f"\nDeployed: https://huggingface.co/spaces/{args.space_id}")
    print("Static Space — builds in seconds, runs the models in the visitor's browser.")


if __name__ == "__main__":
    main()
