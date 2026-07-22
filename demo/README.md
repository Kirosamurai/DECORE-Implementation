# DECORE Demo — Compressed vs. Full VGG-16

An interactive [Gradio](https://gradio.app) app that classifies an image with **both**
the full VGG-16 and the **DECORE-pruned** model, and shows the compression story
side-by-side (params, model size, FLOPs, CPU latency).

![classes](examples/3_cat.png)

## What it shows

| Metric | Full VGG-16 | DECORE-pruned | Reduction |
|---|---:|---:|---:|
| Parameters | 14.72 M | 5.43 M | **63.1%** |
| Model size (fp32) | 58.9 MB | 21.7 MB | **63.1%** |
| FLOPs (MACs) | 313.2 M | 192.0 M | **38.7%** |
| CPU latency (median) | ~3.7 ms | ~2.0 ms | **~45%** |

Both models run on **CPU** to make the point: the compressed model is smaller *and*
faster with no accuracy loss.

## Run locally

From the repository root (with the venv set up — see the top-level README):

```bash
pip install -r requirements.txt          # includes gradio + pillow
python demo/app.py
```

Then open the printed URL (default `http://127.0.0.1:7860`). Pick an example or
upload any image; both models classify it and show top-3 probabilities.

### Model files
The app loads two artifacts (override paths with env vars if needed):

| Env var | Default | What |
|---|---|---|
| `DECORE_BASELINE` | `vgg16_cifar10_baseline.pth` | full VGG-16 state_dict |
| `DECORE_PRUNED` | `runs/pruned_vgg16_l50.pt` | portable pickled DECORE-pruned model |

The pruned artifact is produced by the pipeline (`python -m decore.cli run ...`) or
by re-saving the notebook's pruned model. It is a `decore.models.VGGCifar` pickle, so
`decore/` must be importable (the app adds the repo root to `sys.path` automatically).

## Deploy to Hugging Face Spaces

1. Create a **Gradio** Space.
2. Add `app.py` (this folder), `requirements.txt`, `decore/`, and the two model files.
   - Model files are large — commit them with **git-lfs**, or have `app.py` download
     them at startup from a HF **model repo** via `huggingface_hub.hf_hub_download`.
3. Point `DECORE_BASELINE` / `DECORE_PRUNED` at the uploaded file locations.

The Space will build automatically and serve the same UI publicly.
