"""
Track B (124AD0015) — the full ColdSite-DTI model.

Drug encoder + protein encoder -> cross-attention fusion -> binding prediction.
The cross-attention weights are the explanation Track C's precision@k
evaluation runs against real binding-site ground truth.
See docs/02_GUIDE_124AD0015.md Step 3.
"""
import torch
import torch.nn as nn

from src.model.drug_encoder import DrugEncoder
from src.model.protein_encoder import ProteinEncoder


class ColdSiteDTI(nn.Module):
    def __init__(self, drug_vocab_size: int, protein_vocab_size: int, hidden_dim: int = 128):
        super().__init__()
        self.drug_encoder = DrugEncoder(drug_vocab_size, out_dim=hidden_dim)
        self.protein_encoder = ProteinEncoder(protein_vocab_size, hidden_dim=hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, drug_input: torch.Tensor, protein_input: torch.Tensor):
        """
        drug_input:    (batch, drug_seq_len)
        protein_input: (batch, protein_seq_len)
        Returns:
            pred:               (batch, 1) predicted binding affinity / probability
            cross_attn_weights: (batch, 1, protein_seq_len) -- the explanation used by
                                 Track C's precision@k evaluation
        """
        drug_vec = self.drug_encoder(drug_input)                          # (batch, hidden_dim)
        protein_seq, _protein_self_attn = self.protein_encoder(protein_input)

        # the drug vector "queries" the protein sequence to find interaction points
        fused, cross_attn_weights = self.cross_attention(
            drug_vec.unsqueeze(1), protein_seq, protein_seq
        )
        pred = self.predictor(fused.squeeze(1))
        return pred, cross_attn_weights


if __name__ == "__main__":
    # end-to-end smoke test on dummy data -- run this before real data exists
    drug_vocab_size, protein_vocab_size = 70, 28
    model = ColdSiteDTI(drug_vocab_size, protein_vocab_size)

    dummy_drug = torch.randint(0, drug_vocab_size, (4, 50))
    dummy_protein = torch.randint(0, protein_vocab_size, (4, 300))

    pred, attn = model(dummy_drug, dummy_protein)
    print("Prediction shape:", pred.shape)          # expect (4, 1)
    print("Cross-attention shape:", attn.shape)      # expect (4, 1, 300)
