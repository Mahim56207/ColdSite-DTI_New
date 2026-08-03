"""
Track B (124AD0015) — the protein-side encoder.

Deliberately mirrors the PCLAtt-style architecture from Dr. Dasari's own
TF-binding-site paper: parallel CNN + BiLSTM, then multi-head attention.
Attention weights are returned, not discarded -- Track C needs them.
See docs/02_GUIDE_124AD0015.md Step 2.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.model.drug_encoder import PAD_IDX, sequence_mask


def real_lengths(tokens: torch.Tensor) -> torch.Tensor:
    """Length of each sequence, measured to the LAST non-pad position.

    Counting non-pad tokens instead would silently truncate any sequence that
    has an interior pad id, which is easy to produce accidentally in test data.
    """
    idx = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
    last = torch.where(tokens != PAD_IDX, idx, torch.full_like(idx, -1))
    return (last.max(dim=1).values + 1).clamp(min=1)


class ProteinEncoder(nn.Module):
    """Amino acids -> per-residue features + self-attention weights.

    The CNN and the BiLSTM run in PARALLEL on the same embedding and are then
    concatenated, rather than the conv feeding the LSTM. The convolution picks up
    short local motifs and the recurrent branch picks up long-range order
    independently, so neither branch filters what the other is allowed to see.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128,
                 n_heads: int = 8, kernel_size: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even (the BiLSTM splits it in half)")
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding="same")
        # embed_dim in, not hidden_dim: the LSTM reads the embedding directly
        # because it is a parallel branch, not a stage after the conv.
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2,
                              bidirectional=True, batch_first=True)
        self.merge = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads,
                                               dropout=dropout, batch_first=True)
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len) integer-encoded amino acids
        Returns:
            attn_out:     (batch, seq_len, hidden_dim) -- contextualized residue features
            attn_weights: (batch, seq_len, seq_len)    -- THE EXPLANATION SIGNAL,
                          keep this around for Track C's precision@k evaluation
        """
        mask = sequence_mask(x)                             # True at padding
        lengths = real_lengths(x).cpu()
        emb = self.embedding(x)                             # (batch, seq_len, embed_dim)

        # branch 1: local motifs
        cnn_out = torch.relu(self.conv(emb.permute(0, 2, 1))).permute(0, 2, 1)
        cnn_out = cnn_out.masked_fill(mask.unsqueeze(-1), 0.0)

        # branch 2: long-range context.
        # Packing is not cosmetic: unpacked, the LSTM reads the padding tokens and
        # the backward direction *starts* inside the padding, so every residue
        # representation in a padded batch is contaminated.
        packed = pack_padded_sequence(emb, lengths, batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True,
                                          total_length=x.size(1))

        merged = self.pre_norm(self.merge(torch.cat([cnn_out, lstm_out], dim=-1)))

        # key_padding_mask keeps attention off positions that do not exist.
        # Without it the model can place explanation weight on padding, which
        # would quietly corrupt exactly the number this paper is about.
        attn_out, attn_weights = self.attention(merged, merged, merged,
                                                key_padding_mask=mask)
        attn_out = self.post_norm(merged + self.dropout(attn_out))
        attn_out = attn_out.masked_fill(mask.unsqueeze(-1), 0.0)
        return attn_out, attn_weights


def build_protein_vocab() -> dict:
    """Standard 20 amino acids + a few ambiguity codes. 0 = padding, 1 = unknown."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWYXBZJUO"
    return {aa: i + 2 for i, aa in enumerate(amino_acids)}


def encode_protein(sequence: str, vocab: dict, max_len: int = 1000) -> list:
    """Protein sequence -> integer IDs, truncated to max_len.

    NOTE for Track C: truncation means residues past max_len receive no attention
    weight at all. Whether ground-truth binding sites beyond that cut count as
    misses or are excluded changes precision@k, and must be agreed before any
    numbers are recorded.
    """
    return [vocab.get(aa, 1) for aa in sequence.upper()[:max_len]]


if __name__ == "__main__":
    vocab = build_protein_vocab()
    model = ProteinEncoder(vocab_size=len(vocab) + 2)
    dummy = torch.randint(2, len(vocab) + 2, (4, 300))   # batch of 4, protein length 300
    out, attn = model(dummy)
    print("Encoded output shape:", out.shape)            # expect (4, 300, 128)
    print("Attention weights shape:", attn.shape)        # expect (4, 300, 300)

    # padding must receive no attention and must not change the real residues
    model.eval()
    with torch.no_grad():
        seq = torch.randint(2, len(vocab) + 2, (1, 40))
        padded = torch.cat([seq, torch.zeros(1, 60, dtype=torch.long)], dim=1)
        bare_out, _ = model(seq)
        pad_out, pad_attn = model(padded)
    print("Weight on padding:", pad_attn[0, :40, 40:].abs().max().item())   # expect 0.0
    print("Residue drift:", (bare_out[0] - pad_out[0, :40]).abs().max().item())  # ~0.0
