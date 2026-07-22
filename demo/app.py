"""Interactive Gradio demo: full VGG-16 vs. the DECORE-pruned model.

Upload/pick a CIFAR-style image; both models classify it and we show the
compression story (params, size, FLOPs, CPU latency) side by side.

Run locally:
    python demo/app.py
The pruned model is rebuilt from the DECORE checkpoints via the `decore`
package (portable — no notebook pickles required).
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import gradio as gr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from decore.models import build_model                # noqa: E402
from decore.metrics import count_parameters, count_flops, benchmark_latency  # noqa: E402
from decore.data import CIFAR_MEAN, CIFAR_STD        # noqa: E402

DEVICE = torch.device("cpu")  # demo runs on CPU (the "minimal RAM" story)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Checkpoints (override via env vars).
#   baseline: a state_dict for the full VGG-16.
#   pruned:   a portable, fully-pickled DECORE-pruned model (decore.models.VGGCifar).
BASELINE_CKPT = os.environ.get("DECORE_BASELINE", os.path.join(ROOT, "vgg16_cifar10_baseline.pth"))
PRUNED_CKPT = os.environ.get("DECORE_PRUNED", os.path.join(ROOT, "runs", "pruned_vgg16_l50.pt"))

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def load_models():
    """Return (baseline_model, pruned_model), both on CPU in eval mode."""
    baseline = build_model("vgg16", 10, DEVICE)
    baseline.load_state_dict(torch.load(BASELINE_CKPT, map_location=DEVICE), strict=False)
    baseline.eval()

    pruned = torch.load(PRUNED_CKPT, map_location=DEVICE, weights_only=False).to(DEVICE).eval()
    return baseline, pruned


def metrics_row(model):
    p = count_parameters(model)
    f = count_flops(model)
    lat = benchmark_latency(model, device="cpu", warmup=5, iters=25)["median_ms"]
    return p, f, lat


def comparison_markdown(bm, pm):
    (bp, bf, bl), (pp, pf, pl) = bm, pm
    def mb(params):
        return params * 4 / 1e6  # fp32 bytes -> MB
    def pr(a, b):
        return 100 * (1 - b / a)
    return (
        "| Metric | Full VGG-16 | DECORE-pruned | Reduction |\n"
        "|---|---:|---:|---:|\n"
        f"| Parameters | {bp/1e6:.2f} M | {pp/1e6:.2f} M | **{pr(bp, pp):.1f}%** |\n"
        f"| Model size (fp32) | {mb(bp):.1f} MB | {mb(pp):.1f} MB | **{pr(bp, pp):.1f}%** |\n"
        f"| FLOPs (MACs) | {bf/1e6:.1f} M | {pf/1e6:.1f} M | **{pr(bf, pf):.1f}%** |\n"
        f"| CPU latency (median) | {bl:.2f} ms | {pl:.2f} ms | **{pr(bl, pl):.1f}%** |\n"
    )


# --- Load once at startup ---
BASELINE, PRUNED = load_models()
BM, PM = metrics_row(BASELINE), metrics_row(PRUNED)
COMPARISON_MD = comparison_markdown(BM, PM)


@torch.no_grad()
def _infer(model, x):
    t0 = time.perf_counter()
    probs = F.softmax(model(x), dim=1)[0]
    dt = (time.perf_counter() - t0) * 1000.0
    return {CLASSES[i]: float(probs[i]) for i in range(10)}, dt


def classify(image):
    if image is None:
        return {}, {}, "Pick an example or upload an image, then click **Classify**."
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    base_probs, base_dt = _infer(BASELINE, x)
    pruned_probs, pruned_dt = _infer(PRUNED, x)
    note = (f"Inference time this run — full: **{base_dt:.1f} ms**, "
            f"pruned: **{pruned_dt:.1f} ms**")
    return base_probs, pruned_probs, note


def build_demo():
    with gr.Blocks(title="DECORE — Compressed vs Full VGG-16") as demo:
        gr.Markdown(
            "# DECORE — Compressed vs. Full VGG-16 (CIFAR-10)\n"
            "One RL agent per channel learns which channels to drop. The pruned "
            "model is **~63% smaller** with **no accuracy loss**. Both models below "
            "run on CPU — pick an example or upload an image.")
        gr.Markdown(COMPARISON_MD)
        with gr.Row():
            inp = gr.Image(type="pil", label="Input image", height=240)
            with gr.Column():
                out_base = gr.Label(num_top_classes=3, label=f"Full VGG-16 ({BM[0]/1e6:.1f}M params)")
                out_pruned = gr.Label(num_top_classes=3, label=f"DECORE-pruned ({PM[0]/1e6:.1f}M params)")
        note = gr.Markdown()
        gr.Button("Classify", variant="primary").click(
            classify, inputs=inp, outputs=[out_base, out_pruned, note])

        ex_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
        if os.path.isdir(ex_dir):
            exs = [os.path.join(ex_dir, f) for f in sorted(os.listdir(ex_dir)) if f.endswith(".png")]
            if exs:
                gr.Examples(examples=exs, inputs=inp)
        gr.Markdown("*CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, "
                    "frog, horse, ship, truck. DECORE: [arXiv:2106.06091](https://arxiv.org/abs/2106.06091).*")
    return demo


if __name__ == "__main__":
    build_demo().launch()
