"""
Track B (124AD0015) — the drug-side encoder.

Turns a SMILES string (tokenized to integer IDs beforehand) into a fixed-size
vector. See docs/02_GUIDE_124AD0015.md Step 1.
"""
import torch
import torch.nn as nn


class DrugEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=4)
        self.conv2 = nn.Conv1d(64, 96, kernel_size=6)
        self.conv3 = nn.Conv1d(96, out_dim, kernel_size=8)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) integer-encoded SMILES characters -> (batch, out_dim)"""
        x = self.embedding(x).permute(0, 2, 1)   # (batch, embed_dim, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)              # (batch, out_dim)
        return x


def build_smiles_vocab(smiles_list: list) -> dict:
    """Quick character-level vocab builder. 0 is reserved for padding."""
    chars = sorted(set("".join(smiles_list)))
    return {ch: i + 1 for i, ch in enumerate(chars)}


if __name__ == "__main__":
    # smoke test on dummy data -- do this before real data exists
    vocab_size = 70
    model = DrugEncoder(vocab_size)
    dummy = torch.randint(0, vocab_size, (4, 50))  # batch of 4, length 50
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (4, 128)
