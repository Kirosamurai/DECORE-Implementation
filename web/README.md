---
title: DECORE VGG16 CIFAR10
emoji: 🗜️
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
license: mit
---

# DECORE — Compressed vs. Full VGG-16 (in-browser)

A **static** Hugging Face Space: both the full VGG-16 and its **DECORE-pruned**
version (63% fewer parameters, no accuracy loss) run **entirely in the browser**
via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) — no server,
no GPU. Pick an example or upload an image and see both models classify it, with
live in-browser latency.

Contents: `index.html`, `app.js`, `style.css`, `models/{baseline,pruned}.onnx`,
`examples/`.

Project & training pipeline: https://github.com/Kirosamurai/DECORE-Implementation
