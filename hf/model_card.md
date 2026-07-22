---
license: mit
library_name: pytorch
tags:
  - image-classification
  - model-compression
  - pruning
  - reinforcement-learning
  - decore
  - vgg
datasets:
  - cifar10
metrics:
  - accuracy
model-index:
  - name: decore-vgg16-cifar10
    results:
      - task:
          type: image-classification
        dataset:
          name: CIFAR-10
          type: cifar10
        metrics:
          - type: accuracy
            value: 93.76
            name: Top-1 (pruned)
---

# DECORE — Compressed VGG-16 (CIFAR-10)

Channel-pruned VGG-16 produced with **DECORE** ([arXiv:2106.06091](https://arxiv.org/abs/2106.06091)):
a reinforcement-learning method that assigns one agent per channel to learn which
channels to drop. The dropped channels are physically removed.

| Metric | Full VGG-16 | DECORE-pruned | Reduction |
|---|---:|---:|---:|
| Parameters | 14.72 M | **5.43 M** | **63.1%** |
| FLOPs (MACs) | 313.2 M | **192.0 M** | **38.7%** |
| Top-1 accuracy | 93.48% | **93.76%** | **+0.28%** |

No accuracy loss — VGG-16 is over-parameterized for CIFAR-10, so pruning acts as a
regularizer.

## Files
- `pruned_vgg16_l50.pt` — the compressed model (a pickled `decore.models.VGGCifar`).
- `baseline.pth` — the full VGG-16 baseline (`state_dict`) for comparison.

## Usage
The pruned model is a standard PyTorch module but relies on the `decore` package for
its class definition. Install the package (from the project repo), then:

```python
import torch
pruned = torch.load("pruned_vgg16_l50.pt", map_location="cpu", weights_only=False).eval()

# CIFAR-10 preprocessing
import torchvision.transforms as T
tf = T.Compose([
    T.Resize((32, 32)), T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
])
logits = pruned(tf(img).unsqueeze(0))   # img: a PIL.Image
```

Classes: `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

## Reproduction
See the [project repository](https://github.com/Kirosamurai/DECORE-Implementation):
`python -m decore.cli run --config configs/vgg16_cifar10.yaml`.

## Citation
Alwani, Wang, Madhavan. *DECORE: Deep Compression with Reinforcement Learning.* CVPR 2022.
