"""DECORE: Deep Compression with Reinforcement Learning.

An end-to-end pipeline for channel pruning of CNNs via a lightweight
multi-agent policy-gradient method (one learnable weight per channel).

Reference: Alwani, Wang & Madhavan, "DECORE: Deep Compression with
Reinforcement Learning", CVPR 2022 (arXiv:2106.06091).
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
