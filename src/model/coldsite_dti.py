"""
Track B (124AD0015) — the full ColdSite-DTI model.

Drug encoder + protein encoder -> cross-attention fusion -> binding prediction.
The cross-attention weights are the explanation Track C's precision@k
evaluation runs against real binding-site ground truth.
See docs/02_GUIDE_124AD0015.md Step 3.
"""
import torch
import torch.nn as nn

from src.model.drug_encoder import DrugEncoder, sequence_mask
from src.model.protein_encoder import ProteinEncoder, real_lengths


class ColdSiteDTI(nn.Module):
    def __init__(self, drug_vocab_size: int, protein_vocab_size: int,
                 hidden_dim: int = 128, n_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.drug_encoder = DrugEncoder(drug_vocab_size, out_dim=hidden_dim)
        self.protein_encoder = ProteinEncoder(protein_vocab_size, hidden_dim=hidden_dim,
                                              n_heads=n_heads)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, n_heads,
                                                     dropout=dropout, batch_first=True)
        # Input is 2 * hidden_dim: the raw drug vector is concatenated with the
        # attended protein context. Context alone leaves the head blind to which
        # drug it is looking at whenever attention collapses, which it tends to do
        # in early epochs on the cold splits.
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, drug_input: torch.Tensor, protein_input: torch.Tensor):
        """
        drug_input:    (batch, drug_seq_len)
        protein_input: (batch, protein_seq_len)
        Returns:
            pred:              (batch, 1) raw score -- binding affinity, or a LOGIT
                               for classification. No sigmoid is applied here;
                               BCEWithLogitsLoss applies it internally and is
                               numerically stabler. Apply sigmoid yourself when
                               reporting probabilities.
            cross_attn_weights: (batch, 1, protein_seq_len) -- the explanation used by
                                Track C's precision@k evaluation. Zero at padded
                                positions, and sums to 1 over the real residues.
        """
        drug_vec = self.drug_encoder(drug_input)                  # (batch, hidden_dim)
        protein_seq, _protein_self_attn = self.protein_encoder(protein_input)

        # The mask is recomputed from the token ids rather than threaded out of the
        # protein encoder, which keeps that encoder's two-value return contract intact.
        protein_mask = sequence_mask(protein_input)

        # the drug vector "queries" the protein sequence to find interaction points.
        # One query means the attention map is (batch, 1, protein_seq_len): exactly
        # one interpretable weight per residue, which is what precision@k needs.
        fused, cross_attn_weights = self.cross_attention(
            drug_vec.unsqueeze(1), protein_seq, protein_seq,
            key_padding_mask=protein_mask,
        )
        # Padded keys already get ~0 weight from the softmax; force exact zeros so a
        # saved weight vector can be sliced by residue index with no surprises.
        cross_attn_weights = cross_attn_weights.masked_fill(
            protein_mask.unsqueeze(1), 0.0
        )

        pred = self.predictor(torch.cat([drug_vec, fused.squeeze(1)], dim=-1))
        return pred, cross_attn_weights

    @torch.no_grad()
    def explain(self, drug_input: torch.Tensor, protein_input: torch.Tensor) -> list:
        """Attention weights as plain per-protein lists, padding stripped.

        Hand-off format for Track C: out[i][j] is the weight on residue j of
        protein i, indexed from 0, same length as that protein's real sequence.
        """
        self.eval()
        _pred, attn = self.forward(drug_input, protein_input)
        # real_lengths(), not (protein_input != 0).sum(): the two disagree the
        # moment a sequence contains an interior pad id, and the count version
        # returns a SHORTER array than the residues it covers. Every ground-truth
        # index past that point would then line up against the wrong residue --
        # a silent misalignment in exactly the number this paper reports.
        lengths = real_lengths(protein_input).tolist()
        attn = attn.squeeze(1).cpu()
        return [attn[i, :lengths[i]].tolist() for i in range(len(lengths))]


if __name__ == "__main__":
    # end-to-end smoke test on dummy data -- run this before real data exists
    drug_vocab_size, protein_vocab_size = 70, 28
    model = ColdSiteDTI(drug_vocab_size, protein_vocab_size)

    dummy_drug = torch.randint(2, drug_vocab_size, (4, 50))
    dummy_protein = torch.randint(2, protein_vocab_size, (4, 300))

    pred, attn = model(dummy_drug, dummy_protein)
    print("Prediction shape:", pred.shape)          # expect (4, 1)
    print("Cross-attention shape:", attn.shape)     # expect (4, 1, 300)

    # Attention rows only sum to exactly 1 in eval mode -- attention dropout
    # perturbs them during training. Always call .eval() before exporting
    # explanations for Track C.
    model.eval()
    with torch.no_grad():
        _p, eval_attn = model(dummy_drug, dummy_protein)
    print("Rows sum to 1:", [round(v, 4) for v in
                             eval_attn.sum(dim=-1).squeeze(-1).tolist()])

    # a padded protein must not shift its own explanation or its prediction
    with torch.no_grad():
        short = torch.zeros(4, 300, dtype=torch.long)
        short[:, :120] = torch.randint(2, protein_vocab_size, (4, 120))
        _p, a = model(dummy_drug, short)
    print("Weight past residue 120:", a[:, :, 120:].abs().max().item())   # expect 0.0
    print("Explanation lengths:", [len(w) for w in model.explain(dummy_drug, short)])
