"""
DeepDTA, ported to PyTorch.

Why a port exists at all
------------------------
The vendored DeepDTA (`baselines/DeepDTA/source/run_experiments.py`) is
TensorFlow/Keras from 2018 — `keras.layers.merge`, `plot_model`, a Keras 2.x
functional API and a TF1-era session model. Nothing else in this project uses
TensorFlow; `requirements.txt` pins torch and never mentions it. Getting that
file to run means installing a second deep-learning stack whose modern versions
have removed several of the APIs it calls.

The audit does not need the original binary. DeepDTA is in the grid as the
**accuracy anchor**: `provides_attention = False`, so it never contributes an
explanation, and its only job is to let a reviewer see whether the interpretable
models pay an accuracy cost. For that, a faithful implementation of the
published architecture is worth exactly as much as the original weights, and it
is far more likely to still run in November.

This is a Methods-section fact, not a footnote. Write it as: *"DeepDTA was
reimplemented in PyTorch following Öztürk et al. (2018); architecture and
hyperparameters are as published."*

Fidelity
--------
Layer for layer against `build_combined_categorical` in the vendored
`run_experiments.py`:

    drug     Embedding(65, 128) -> Conv1d(32, k=4) -> Conv1d(64, k=4)
             -> Conv1d(96, k=4) -> GlobalMaxPool
    protein  Embedding(26, 128) -> Conv1d(32, k=8) -> Conv1d(64, k=8)
             -> Conv1d(96, k=8) -> GlobalMaxPool
    head     concat -> 1024 -> Dropout(0.1) -> 1024 -> Dropout(0.1)
             -> 512 -> 1

All convolutions are `padding='valid'`, stride 1, ReLU — the PyTorch default
padding of 0 is the same thing. The final layer has no activation, matching the
vendored comment that it deliberately does not apply a sigmoid.

Vocabulary sizes are `CHARISOSMILEN + 1 = 65` and `CHARPROTLEN + 1 = 26`; the
`+1` is index 0 reserved for padding, exactly as the Keras version does.

Kernel widths are a published hyperparameter, not a constant: the paper sweeps
`smi_window_lengths` and `seq_window_lengths` and picks per dataset. 4 and 8 are
the defaults here; if the grid uses different values, record them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Reserved index 0 for padding, so vocabulary size is the charset size + 1.
DRUG_VOCAB_SIZE = 65        # CHARISOSMILEN 64 + 1
PROTEIN_VOCAB_SIZE = 26     # CHARPROTLEN 25 + 1

MAX_SMILES_LEN = 100
MAX_PROTEIN_LEN = 1000


class DeepDTA(nn.Module):
    """The published DeepDTA architecture.

    Deliberately has no `explain()`. DeepDTA has no attention, and inventing an
    explanation for it -- gradient saliency, occlusion, anything -- would put a
    different method's explanation into a table that reads as DeepDTA's. The
    adapter keeps `provides_attention = False` and `explain()` keeps raising.
    """

    def __init__(self, n_filters: int = 32, drug_kernel: int = 4,
                 protein_kernel: int = 8, embed_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.drug_kernel = drug_kernel
        self.protein_kernel = protein_kernel

        self.drug_embed = nn.Embedding(DRUG_VOCAB_SIZE, embed_dim, padding_idx=0)
        self.protein_embed = nn.Embedding(PROTEIN_VOCAB_SIZE, embed_dim,
                                          padding_idx=0)

        self.drug_conv = nn.Sequential(
            nn.Conv1d(embed_dim, n_filters, drug_kernel), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters * 2, drug_kernel), nn.ReLU(),
            nn.Conv1d(n_filters * 2, n_filters * 3, drug_kernel), nn.ReLU(),
        )
        self.protein_conv = nn.Sequential(
            nn.Conv1d(embed_dim, n_filters, protein_kernel), nn.ReLU(),
            nn.Conv1d(n_filters, n_filters * 2, protein_kernel), nn.ReLU(),
            nn.Conv1d(n_filters * 2, n_filters * 3, protein_kernel), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(n_filters * 3 * 2, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        # Keras Conv1D is (batch, length, channels); torch Conv1d is
        # (batch, channels, length). The permute is the whole of the difference.
        d = self.drug_conv(self.drug_embed(drug).permute(0, 2, 1))
        p = self.protein_conv(self.protein_embed(protein).permute(0, 2, 1))

        # GlobalMaxPooling1D over the length axis
        d = torch.max(d, dim=2).values
        p = torch.max(p, dim=2).values

        return self.head(torch.cat([d, p], dim=1)).squeeze(-1)


def _charsets():
    """DeepDTA's own charsets, imported from the vendored repo rather than copied.

    Copying them here would work today and drift silently later. Each audited
    model is tokenised with its *own* authors' table on purpose: if two repos
    ever disagree about a character, the audit should reproduce each model as
    published rather than quietly standardise them.
    """
    import os
    import sys

    root = os.path.join("baselines", "DeepDTA", "source")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found. Clone the baseline first:\n"
            f"    cd baselines && git clone https://github.com/hkmztrk/DeepDTA.git DeepDTA"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    from datahelper import CHARISOSMISET, CHARPROTSET  # noqa: E402
    return CHARISOSMISET, CHARPROTSET


def encode_smiles(smiles: str, max_len: int = MAX_SMILES_LEN) -> np.ndarray:
    charset, _ = _charsets()
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, char in enumerate(str(smiles)[:max_len]):
        encoded[i] = charset.get(char, 0)
    return encoded


def encode_protein(sequence: str, max_len: int = MAX_PROTEIN_LEN) -> np.ndarray:
    _, charset = _charsets()
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, char in enumerate(str(sequence)[:max_len]):
        encoded[i] = charset.get(char, 0)
    return encoded
