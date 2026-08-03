"""
Track B (124AD0015) — the drug-side encoder.

Turns a SMILES string (tokenized to integer IDs beforehand) into a fixed-size
vector. See docs/02_GUIDE_124AD0015.md Step 1.
"""
import torch
import torch.nn as nn

PAD_IDX = 0
UNK_IDX = 1


def sequence_mask(tokens: torch.Tensor) -> torch.Tensor:
    """True where the position is padding. Shape (batch, seq_len).

    This is the convention nn.MultiheadAttention expects for key_padding_mask:
    True means "ignore this key".
    """
    return tokens == PAD_IDX


def masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max-pool over time, ignoring padded positions.

    x:    (batch, channels, seq_len)
    mask: (batch, seq_len), True at padding
    """
    x = x.masked_fill(mask.unsqueeze(1), float("-inf"))
    pooled = x.max(dim=-1).values
    # A row that is entirely padding would come out as -inf and poison the
    # gradients everywhere downstream; zero it instead.
    return torch.nan_to_num(pooled, neginf=0.0)


class DrugEncoder(nn.Module):
    """SMILES characters -> one fixed-size vector.

    Kernel sizes widen (4 -> 6 -> 8) so deeper layers see longer substructures.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, out_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # padding="same" rather than valid padding: with valid padding a SMILES
        # shorter than the summed kernel widths gives a zero-length feature map
        # and crashes. Short fragments do occur in DAVIS/KIBA.
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=4, padding="same")
        self.conv2 = nn.Conv1d(64, 96, kernel_size=6, padding="same")
        self.conv3 = nn.Conv1d(96, out_dim, kernel_size=8, padding="same")
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) integer-encoded SMILES characters -> (batch, out_dim)"""
        mask = sequence_mask(x)
        channel_mask = mask.unsqueeze(1)                    # (batch, 1, seq_len)
        x = self.embedding(x).permute(0, 2, 1)              # (batch, embed_dim, seq_len)

        # Activations over padded positions are re-zeroed after every conv.
        # Without this the conv bias makes the padded region non-zero, the next
        # conv's window picks that up at the sequence boundary, and a molecule's
        # encoding silently changes depending on how much padding its batch-mates
        # forced onto it. AdaptiveMaxPool1d would then pool over that garbage too.
        for conv in (self.conv1, self.conv2, self.conv3):
            x = torch.relu(conv(x)).masked_fill(channel_mask, 0.0)

        x = masked_max_pool(x, mask)                        # (batch, out_dim)
        return self.dropout(self.norm(x))


def build_smiles_vocab(smiles_list: list) -> dict:
    """Character-level vocab builder. 0 = padding, 1 = unknown.

    The UNK slot is not optional here. The cold-drug and cold-pair splits put
    molecules in test that were never seen in training, so a test SMILES will
    eventually contain a character this dict was never built from. Without UNK
    that is a KeyError mid-evaluation; with it, the character degrades quietly.

    Build this from the TRAINING split only — building it from all splits leaks
    information about unseen drugs, which is the exact thing the cold splits
    exist to measure.
    """
    chars = sorted(set("".join(smiles_list)))
    return {ch: i + 2 for i, ch in enumerate(chars)}


def encode_smiles(smiles: str, vocab: dict, max_len: int = 100) -> list:
    """SMILES string -> list of integer IDs, truncated to max_len.

    Padding is not applied here; the collate function pads each batch to its own
    longest member so short batches don't pay for the longest molecule overall.
    """
    return [vocab.get(ch, UNK_IDX) for ch in smiles[:max_len]]


if __name__ == "__main__":
    # smoke test on dummy data -- do this before real data exists
    vocab_size = 70
    model = DrugEncoder(vocab_size)
    # low=2 so the dummy contains no PAD/UNK ids: randint(0, ...) would scatter
    # PAD tokens through the middle of a "molecule", which is not what real
    # batches look like and would make this test misleading.
    dummy = torch.randint(2, vocab_size, (4, 50))    # batch of 4, length 50
    out = model(dummy)
    print("Output shape:", out.shape)                # expect (4, 128)

    # padding invariance: the same molecule must encode identically whether or
    # not its batch-mates forced trailing padding onto it
    model.eval()
    with torch.no_grad():
        seq = torch.randint(2, vocab_size, (1, 20))
        padded = torch.cat([seq, torch.zeros(1, 30, dtype=torch.long)], dim=1)
        drift = (model(seq) - model(padded)).abs().max().item()
    print(f"Padding invariance drift: {drift:.2e}")   # expect ~0.0
