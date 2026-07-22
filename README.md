# DECORE — Deep Compression with Reinforcement Learning

[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/KiroSamurai/decore-vgg16)

A faithful, reproducible implementation of **DECORE** (Alwani, Wang & Madhavan, *CVPR 2022*, [arXiv:2106.06091](https://arxiv.org/abs/2106.06091)) for **structured channel pruning** of CNNs, plus an end-to-end, config-driven pipeline (train → prune → fine-tune → benchmark → export).

**▶ Try it live (runs in your browser):** https://huggingface.co/spaces/KiroSamurai/decore-vgg16

DECORE assigns a lightweight **reinforcement-learning agent to every channel**. Each agent has a *single* learnable weight and decides whether to keep or drop its channel. A tiny policy-gradient (REINFORCE) update — rewarding compression while penalizing accuracy loss — discovers which channels are redundant. The dropped channels are then **physically removed**, producing a genuinely smaller and faster model.

---

## Headline result (VGG-16 / CIFAR-10)

Reproduced on Apple-Silicon MPS. Matches the paper's `DECORE-500` operating point (63.0% params / 35.3% FLOPs / 94.02%):

| Metric | Baseline | Pruned | Reduction |
|---|---:|---:|---:|
| **Params** | 14.72M | **5.43M** | **63.1%** |
| **FLOPs** (MACs) | 313.20M | **192.03M** | **38.7%** |
| **Top-1 accuracy** | 93.48% | **93.76%** | **+0.28%** |

**No accuracy loss — it slightly improved.** Exactly the paper's finding: VGG-16 is so over-parameterized for CIFAR-10 that pruning acts as a regularizer. The physically pruned model is **bit-exact** with the masked model (max logit diff `0.0`).

## 🎮 Interactive demo (runs in your browser)

Both the full VGG-16 and the DECORE-pruned model run **entirely client-side** via
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) — no server, no GPU.
Pick/upload an image and both classify it, showing params / size / FLOPs / **live
latency** side-by-side: same accuracy, 63% smaller, ~2× faster.

- **Live demo:** **https://huggingface.co/spaces/KiroSamurai/decore-vgg16** (free static Space, runs in-browser)
- **Run locally:** `python scripts/export_onnx.py` then serve `web/` (`python -m http.server -d web 8899` → open `http://localhost:8899`).
- **Deploy (free, static):**
  ```bash
  python scripts/export_onnx.py                                  # -> web/models/*.onnx
  python scripts/deploy_static_space.py --space-id <user>/decore-vgg16
  ```
- **Publish the model** (optional): `python scripts/upload_hf.py --repo-id <user>/decore-vgg16-cifar10` (compressed model + [model card](hf/model_card.md)).

### What the policy learned
DECORE concentrates cuts where the redundancy (and the parameters) live — the wide late layers — while barely touching the precious early features:

```
layer  0:  keep  62/64    (early conv — almost untouched)
layer  6:  keep 180/256
layer 10:  keep 290/512   (late convs — ~44% dropped)
layer 16:  keep 280/512
Total: 2688/4224 channels kept  →  36% channels, but 63% of parameters
```

---

## How it works

For each layer *i* with *Cᵢ* channels, one agent per channel holds a weight *wⱼ*:

1. **Policy** — keep-probability `pⱼ = σ(wⱼ)`, action `aⱼ ~ Bernoulli(pⱼ)` (1 = keep, 0 = drop). Weights start at `6.9` so `σ(6.9) ≈ 0.99` (keep everything initially).
2. **Masking** — the sampled action multiplies the channel's activation (a learnable, structured dropout).
3. **Reward** — per layer, `Rᵢ = Rᵢ,C · R_acc` where `Rᵢ,C = Σ(1 − aⱼ)` (channels dropped) and `R_acc = 1` if the prediction is correct else `−λ`. High `λ` ⇒ conservative pruning; low `λ` ⇒ aggressive.
4. **Optimization** — agents are trained with **REINFORCE**, *separately* from the network's cross-entropy loss. One scalar per agent ⇒ extremely fast vs. RL methods that train a whole policy network.
5. **Prune & fine-tune** — after policy training, channels with `pⱼ < 0.5` are removed and the compact model is fine-tuned.

### Key design decisions (why this repo is correct)
- **One `nn.Parameter` per channel** (not a linear layer) — matches the paper's single-parameter agent.
- **Mask applied *after* `Conv→BN→ReLU`** — makes the masked model *exactly* equivalent to physical removal (a zeroed block output contributes nothing downstream), so accuracy doesn't change when channels are cut.
- **Deterministic fine-tuning** — after policy training the mask is frozen (`p ≥ 0.5`); we fine-tune the *actual* pruned subnetwork, not random ones.
- **CIFAR VGG-16 variant** (BatchNorm, 32×32, ~14.98M params) — the paper's baseline, not the 138M-param ImageNet VGG.
- **Real physical rebuild + FLOP/param counting** — the compressed model is reconstructed (conv, BN, and next-layer input channels sliced consistently), then measured.
- **Co-adaptation matters** — the network is kept *plastic* during joint training (it must relearn to work with fewer channels), which is what unlocks aggressive compression.

---

## Repository structure

```
DECORE-Implementation/
├── decore/                       # the pipeline package
│   ├── config.py                 # dataclass config (YAML load/dump)
│   ├── data.py                   # CIFAR-10 loaders (cached, 32×32)
│   ├── models.py                 # CIFAR VGG-16 + PrunableConvBNReLU
│   ├── agents.py                 # Agent (one weight/channel)
│   ├── pruning.py                # masks, compression %, physical rebuild
│   ├── metrics.py                # params, FLOPs, CPU-latency benchmark
│   ├── engine.py                 # baseline / DECORE / fine-tune / eval
│   ├── export.py                 # TorchScript / ONNX + parity check
│   ├── registry.py               # append-only runs.jsonl + queries
│   └── cli.py                    # run / benchmark / export
├── configs/vgg16_cifar10.yaml    # one YAML per experiment
├── notebooks/vgg16_cifar10.ipynb # annotated, from-scratch walkthrough
├── web/                          # static in-browser demo (ONNX Runtime Web)
│   ├── index.html · app.js · style.css
│   ├── models/                   # exported *.onnx (gitignored; regenerate)
│   └── examples/
├── scripts/                      # export_onnx, upload_hf, deploy_static_space
├── hf/model_card.md              # Hugging Face model card
├── paper/                        # the DECORE paper
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Runs on CUDA, Apple-Silicon **MPS**, or CPU (auto-detected).

## Usage

### Pipeline (CLI)
```bash
# Full reproduction: baseline (cached) → DECORE → prune → report → registry
python -m decore.cli run --config configs/vgg16_cifar10.yaml

# Quick end-to-end sanity check (~2 min, tiny schedule)
python -m decore.cli run --config configs/vgg16_cifar10.yaml --smoke

# Benchmark CPU latency / export the compressed model
python -m decore.cli benchmark --config configs/vgg16_cifar10.yaml
python -m decore.cli export    --config configs/vgg16_cifar10.yaml
```
Sweep the accuracy/compression trade-off by changing `lambda_penalty` in the YAML (lower ⇒ more compression). Every run appends params / FLOPs / accuracy / CPU-latency to `runs/runs.jsonl`.

### Notebook
`notebooks/vgg16_cifar10.ipynb` is a fully annotated, cell-by-cell walkthrough of the same method (data → model → agents → baseline → DECORE → physical pruning → report).

---

## Reproduction notes
- Schedule (paper): 300 epochs total, policy training stops at 260, then 40 epochs of fine-tuning; agents use Adam (`lr 0.01`), batch size 256.
- A trained baseline is pruned (the paper starts from a pretrained network); we cache it to skip re-training on subsequent runs.
- The reward is positive (pruning is rewarded) only when `train_acc > λ/(λ+1)`; the joint-phase learning rate is kept high enough for the network to co-adapt.

## Roadmap
- [ ] **ResNet-56 / CIFAR-10** — residual-aware pruning (shared policies across skip connections).
- [ ] **INT8 quantization** on top of pruning (multiplicative compression).
- [ ] **Minimal-RAM web app** — classify images with the compressed model, side-by-side vs. full VGG-16 (size / latency / accuracy).

## References
- M. Alwani, Y. Wang, V. Madhavan. *DECORE: Deep Compression with Reinforcement Learning.* CVPR 2022. [arXiv:2106.06091](https://arxiv.org/abs/2106.06091)
- R. J. Williams. *Simple statistical gradient-following algorithms for connectionist reinforcement learning (REINFORCE).* Machine Learning, 1992.
- Simonyan & Zisserman. *Very Deep Convolutional Networks (VGG).* ICLR 2015.
