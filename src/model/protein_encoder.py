"""
Track B (124AD0015) — the protein-side encoder.

Deliberately mirrors the PCLAtt-style architecture from Dr. Dasari's own
TF-binding-site paper: parallel CNN + BiLSTM, then multi-head attention.
Attention weights are returned, not discarded -- Track C needs them.
See docs/02_GUIDE_124AD0015.md Step 2.
"""
import torch
import torch.nn as nn


class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, n_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size=8, padding="same")
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len) integer-encoded amino acids
        Returns:
            attn_out:     (batch, seq_len, hidden_dim) -- contextualized residue features
            attn_weights: (batch, seq_len, seq_len)     -- THE EXPLANATION SIGNAL,
                          keep this around for Track C's precision@k evaluation
        """
        x = self.embedding(x)                                    # (batch, seq_len, embed_dim)
        x = torch.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        x, _ = self.bilstm(x)                                     # (batch, seq_len, hidden_dim)
        attn_out, attn_weights = self.attention(x, x, x)
        return attn_out, attn_weights


def build_protein_vocab() -> dict:
    """Standard 20 amino acids + a few ambiguity codes. 0 is reserved for padding."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWYXBZJUO"
    return {aa: i + 1 for i, aa in enumerate(amino_acids)}


if __name__ == "__main__":
    vocab = build_protein_vocab()
    model = ProteinEncoder(vocab_size=len(vocab) + 1)
    dummy = torch.randint(0, len(vocab) + 1, (4, 300))  # batch of 4, protein length 300
    out, attn = model(dummy)
    print("Encoded output shape:", out.shape)   # expect (4, 300, 128)
    print("Attention weights shape:", attn.shape)  # expect (4, 300, 300)
