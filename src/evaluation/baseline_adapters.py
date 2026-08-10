"""
Baseline adapters — DeepDTA, HyperAttentionDTI, MolTrans as audit subjects.

Each adapter wraps a published model behind the two-method contract in
`model_registry.ExplainableDTIModel`:

    predict(drug, protein) -> float
    explain(drug, protein) -> one non-negative weight per REAL residue, 0-indexed

Everything hard about `explain()` — mapping a model's internal attention
positions back to residue indices — lives in
`src/evaluation/attention_projection.py`, built and tested by 124AD0015. These
adapters do the unglamorous half: locate the vendored repo, load a checkpoint,
tokenise the way that model's own authors tokenise, and hand the right *real
length* to the projection.

Each model is tokenised with its own vendored charset table, never a shared one.
If two repos ever disagree about a character, the audit should reproduce each
model as published rather than quietly standardise them.

STATUS: written but never executed against a live checkpoint -- the environment
they were written in has no torch. `validate_adapter` is the gate, and it is not
optional:

    from src.evaluation.model_registry import validate_adapter
    print(validate_adapter(adapter, drug_tensor, protein_tensor,
                           expected_length=min(len(sequence), 1000)))

`valid: True` is the bar. The failure that matters is not a crash -- it is an
adapter returning a plausible array of the wrong length, which misaligns every
ground-truth index and yields a wrong precision@k that looks entirely
reasonable.

See docs/PART2_GUIDE_124AD0008.md Priority 5.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from src.evaluation.model_registry import ExplainableDTIModel, register

BASELINE_ROOT = "baselines"


def _vendored(subdir: str, clone_hint: str) -> str:
    """Absolute path to a vendored baseline repo, added to sys.path."""
    path = os.path.abspath(os.path.join(BASELINE_ROOT, subdir))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{path} not found.\nClone it first:\n    {clone_hint}"
        )
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def _as_batch(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor if tensor.dim() == 2 else tensor.unsqueeze(0)


class _Unimplemented(ExplainableDTIModel):
    """Kept for the models that deliberately do not explain."""

    repo_url = ""
    what_to_do = ""

    def __init__(self, checkpoint_path: str = None, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device

    def _explain_error(self, method: str):
        return NotImplementedError(
            f"\n{type(self).__name__}.{method}() is not implemented.\n"
            f"  repo:  {self.repo_url}\n"
            f"  file:  src/evaluation/baseline_adapters.py\n"
            f"  why:   {self.what_to_do}\n"
        )

    def predict(self, drug, protein) -> float:
        raise self._explain_error("predict")

    def explain(self, drug, protein) -> np.ndarray:
        raise self._explain_error("explain")


# ---------------------------------------------------------------------------
# DeepDTA -- the accuracy anchor
# ---------------------------------------------------------------------------

@register("deepdta")
class DeepDTAAdapter(ExplainableDTIModel):
    """DeepDTA. CNN encoders, no attention, accuracy axis only.

    `provides_attention = False` is correct and not a gap. DeepDTA's presence
    lets a reviewer see whether the interpretable models pay an accuracy cost
    for their explanations. `explain()` raises deliberately: manufacturing an
    explanation for it (saliency, occlusion) would put a *different method's*
    output into a table that reads as DeepDTA's.

    Runs against `src/model/deepdta_torch.py`, a PyTorch port of the published
    architecture, because the vendored original is TF1-era Keras. See that
    module's docstring -- it is a Methods-section fact.
    """

    provides_attention = False
    citation = "Ozturk et al., Bioinformatics 2018"
    repo_url = "https://github.com/hkmztrk/DeepDTA"

    def __init__(self, checkpoint_path: str = None, device: str = "cpu",
                 drug_kernel: int = 4, protein_kernel: int = 8):
        import torch

        from src.model.deepdta_torch import DeepDTA

        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = DeepDTA(drug_kernel=drug_kernel, protein_kernel=protein_kernel)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device,
                               weights_only=False)
            self.model.load_state_dict(state.get("model_state", state))
        self.model.to(device).eval()

    def predict(self, drug, protein) -> float:
        import torch

        with torch.no_grad():
            out = self.model(_as_batch(drug).to(self.device),
                             _as_batch(protein).to(self.device))
        return float(out.squeeze().item())

    def explain(self, drug, protein) -> np.ndarray:
        raise NotImplementedError(
            "DeepDTA exposes no attention. It is in the audit as the accuracy "
            "anchor (provides_attention = False); nothing should be calling "
            "explain() on it. If the audit runner reached here, it is not "
            "honouring provides_attention."
        )


# ---------------------------------------------------------------------------
# HyperAttentionDTI -- attention over convolution positions
# ---------------------------------------------------------------------------

@register("hyperattentiondti")
class HyperAttentionDTIAdapter(ExplainableDTIModel):
    """HyperAttentionDTI. Attention lives on CONVOLUTION positions.

    Two traps, both handled in `attention_projection`:

    * The drug axis is already averaged away inside the vendored `forward`
      (`torch.mean(Atten_matrix, 1)`), so the axis left to reduce is 160
      channels, not drug positions.
    * A [4, 8, 12] kernel stack turns 1000 residues into 979 positions, each a
      22-residue window. Returning those 979 values as residues misaligns every
      ground-truth index past the first.

    The remaining job here is passing the REAL residue count -- not 1000, not
    979 -- which is `min(len(sequence), 1000)`.
    """

    provides_attention = True
    citation = "Zhao et al., Bioinformatics 2022"
    repo_url = "https://github.com/kexinhuang12345/HyperAttentionDTI"
    clone_hint = ("cd baselines && git clone <HyperAttentionDTI url> "
                  "HpyerAttentionDTI")

    def __init__(self, checkpoint_path: str = None, device: str = "cpu",
                 channel_reduce: str = "mean", projection_mode: str = "centre"):
        import torch

        _vendored("HpyerAttentionDTI", self.clone_hint)
        from hyperparameter import hyperparameter  # noqa: E402
        from model import AttentionDTI             # noqa: E402

        self.device = device
        self.checkpoint_path = checkpoint_path
        # Both are recorded rather than hardcoded: each is a defensible choice
        # that changes the numbers, so Methods has to state which was used.
        self.channel_reduce = channel_reduce
        self.projection_mode = projection_mode

        self.model = AttentionDTI(hyperparameter())
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device,
                               weights_only=False)
            self.model.load_state_dict(state.get("model_state", state))
        self.model.to(device).eval()

    @staticmethod
    def encode(smiles: str, sequence: str):
        """Tokenise with HyperAttentionDTI's own tables."""
        import torch

        _vendored("HpyerAttentionDTI", HyperAttentionDTIAdapter.clone_hint)
        from dataset import (CHARISOSMISET, CHARPROTSET,  # noqa: E402
                             label_sequence, label_smiles)

        drug = torch.from_numpy(label_smiles(smiles, CHARISOSMISET, 100))
        protein = torch.from_numpy(label_sequence(sequence, CHARPROTSET, 1000))
        return drug, protein

    def predict(self, drug, protein) -> float:
        import torch

        with torch.no_grad():
            out = self.model(_as_batch(drug).to(self.device),
                             _as_batch(protein).to(self.device))
        # The vendored head returns 2 logits (binary). Use the positive-class
        # logit so the value is monotone in predicted affinity, which is what
        # the faithfulness masking compares against.
        out = out.squeeze()
        return float(out[-1].item() if out.ndim else out.item())

    def explain(self, drug, protein) -> np.ndarray:
        from src.evaluation.attention_projection import (
            hyperattentiondti_protein_attention,
            project_conv_attention,
        )

        drug_b, protein_b = _as_batch(drug), _as_batch(protein)
        real_length = int((protein_b[0] != 0).sum().item())
        if real_length == 0:
            raise ValueError("protein tensor is entirely padding")

        conv = hyperattentiondti_protein_attention(
            self.model, drug_b.to(self.device), protein_b.to(self.device),
            channel_reduce=self.channel_reduce)
        return project_conv_attention(conv[0], real_length,
                                      mode=self.projection_mode)


# ---------------------------------------------------------------------------
# MolTrans -- attention over ESPF subword tokens
# ---------------------------------------------------------------------------

@register("moltrans")
class MolTransAdapter(ExplainableDTIModel):
    """MolTrans. Attention lives on ESPF SUBWORD tokens, not residues.

    `max_protein_seq = 545` counts tokens, and one token spans several amino
    acids, so 545 is never the residue count. `moltrans_covered_residues` is.

    Attention is captured with a forward hook rather than by editing the
    vendored `SelfAttention.forward`, which returns only `context_layer`.
    Hooking leaves prediction behaviour byte-identical to upstream -- an audit
    that silently modified a subject's code would be measuring something other
    than the published model.

    Note `stream.py` opens its ESPF tables by relative path (`./ESPF/...`), so
    importing it requires the working directory to be the vendored repo. That
    is handled in `encode`.
    """

    provides_attention = True
    citation = "Huang et al., Bioinformatics 2021"
    repo_url = "https://github.com/kexinhuang12345/MolTrans"
    clone_hint = "cd baselines && git clone <MolTrans url> MolTrans"

    def __init__(self, checkpoint_path: str = None, device: str = "cpu",
                 head_reduce: str = "mean"):
        import torch

        _vendored("MolTrans", self.clone_hint)
        from config import BIN_config_DBPE       # noqa: E402
        from models import BIN_Interaction_Flat  # noqa: E402

        self.device = device
        self.checkpoint_path = checkpoint_path
        self.head_reduce = head_reduce
        self.config = BIN_config_DBPE()

        self.model = BIN_Interaction_Flat(**self.config)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device,
                               weights_only=False)
            state = state.get("model_state", state)
            # MolTrans is commonly trained under DataParallel, which prefixes
            # every key with "module.". Loading that as-is fails with a wall of
            # missing-key errors that says nothing about the real cause.
            if any(k.startswith("module.") for k in state):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.model.load_state_dict(state)
        self.model.to(device).eval()

    def _fit_batch_size(self, batch_size: int) -> None:
        """Make the vendored reshapes agree with the real batch dimension.

        `BIN_Interaction_Flat.forward` reshapes twice using the *config* batch
        size rather than the tensor's actual first dimension:

            i_v = i.view(int(self.batch_size / self.gpus), -1, max_d, max_p)
            f   = f.view(int(self.batch_size / self.gpus), -1)

        With `config['batch_size'] = 16` and a single pair, the element count
        still divides evenly, so nothing raises -- `view` happily reshapes one
        sample's interaction map into 16 rows and the decoder returns 16
        scores. That is what "a Tensor with 16 elements cannot be converted to
        Scalar" was: not a broken adapter, a model that only computes correct
        scores when the batch happens to be exactly config['batch_size'].

        Worth knowing beyond this adapter: any run whose final batch is partial
        hits the same path, silently, with no error.

        Setting the attribute per call is the minimal fix and leaves the
        vendored file untouched, so the audited model stays byte-identical to
        what its authors published.

        `self.gpus` is `torch.cuda.device_count()`, which is 0 on a CPU-only
        machine -- the division would then be by zero. Floored at 1.
        """
        gpus = max(int(getattr(self.model, "gpus", 0) or 0), 1)
        self.model.gpus = gpus
        self.model.batch_size = int(batch_size) * gpus

    @staticmethod
    def encode(smiles: str, sequence: str):
        """ESPF-tokenise with MolTrans's own tables.

        Returns (drug, drug_mask, protein, protein_mask, protein_tokens). The
        tokens come back because `explain()` needs them to project attention
        onto residues -- the mask alone cannot say how many residues a token
        covered.

        Requires `subword_nmt`, which is NOT in requirements.txt:
            python -m pip install subword-nmt
        """
        import numpy as np
        import torch

        repo = _vendored("MolTrans", MolTransAdapter.clone_hint)
        previous = os.getcwd()
        os.chdir(repo)                     # stream.py reads './ESPF/...'
        try:
            from stream import drug2emb_encoder, protein2emb_encoder  # noqa: E402
            from subword_nmt.apply_bpe import BPE  # noqa: F401,E402
            import codecs

            d, d_mask = drug2emb_encoder(str(smiles))
            p, p_mask = protein2emb_encoder(str(sequence))

            bpe = BPE(codecs.open("./ESPF/protein_codes_uniprot.txt"),
                      merges=-1, separator="")
            tokens = bpe.process_line(str(sequence)).split()
        finally:
            os.chdir(previous)

        to_tensor = lambda x: torch.from_numpy(np.asarray(x))  # noqa: E731
        return (to_tensor(d), to_tensor(d_mask),
                to_tensor(p), to_tensor(p_mask), tokens)

    def predict(self, drug, protein, drug_mask=None, protein_mask=None) -> float:
        import torch

        d, p = _as_batch(drug), _as_batch(protein)
        dm = _as_batch(drug_mask) if drug_mask is not None else (d != 0).long()
        pm = _as_batch(protein_mask) if protein_mask is not None else (p != 0).long()

        self._fit_batch_size(d.shape[0])
        with torch.no_grad():
            out = self.model(d.to(self.device), p.to(self.device),
                             dm.to(self.device), pm.to(self.device))

        out = out.squeeze()
        if out.numel() != d.shape[0]:
            raise RuntimeError(
                f"MolTrans returned {out.numel()} scores for a batch of "
                f"{d.shape[0]}. The vendored reshape is still using a "
                f"mismatched batch size -- check _fit_batch_size against "
                f"baselines/MolTrans/models.py before trusting any number."
            )
        return float(out.reshape(-1)[0].item())

    def explain(self, drug, protein, protein_tokens=None,
                drug_mask=None, protein_mask=None) -> np.ndarray:
        """Per-residue attention.

        `protein_tokens` is required. Without it there is no way to know how
        many residues each token covered, and the only alternative -- assuming
        one token is one residue -- produces a 545-long array indexed by the
        wrong thing, which is exactly the silent failure this whole module
        exists to prevent.
        """
        import torch

        from src.evaluation.attention_projection import (
            capture_moltrans_attention,
            project_token_attention,
        )

        if protein_tokens is None:
            raise ValueError(
                "protein_tokens is required. Get them from "
                "MolTransAdapter.encode(smiles, sequence)[4]. Assuming one "
                "token per residue silently misaligns every ground-truth index."
            )

        d, p = _as_batch(drug), _as_batch(protein)
        dm = _as_batch(drug_mask) if drug_mask is not None else (d != 0).long()
        pm = _as_batch(protein_mask) if protein_mask is not None else (p != 0).long()

        target = self.model.p_encoder.layer[-1].attention.self
        store, handle = capture_moltrans_attention(target)
        try:
            self.model.eval()          # dropout is applied to attention_probs
            self._fit_batch_size(d.shape[0])
            with torch.no_grad():
                self.model(d.to(self.device), p.to(self.device),
                           dm.to(self.device), pm.to(self.device))
        finally:
            handle.remove()

        probs = store.get("probs")
        if probs is None:
            raise RuntimeError(
                "the attention hook captured nothing. The vendored module "
                "layout may differ from p_encoder.layer[-1].attention.self -- "
                "check baselines/MolTrans/models.py before trusting any number."
            )

        # (B, heads, tokens, tokens) -> one weight per token.
        # Reduce heads, then average over queries: how much every other token
        # attends TO this one. Recorded in Methods; see PROJECTION_DECISIONS.
        weights = probs[0]
        weights = (weights.mean(dim=0) if self.head_reduce == "mean"
                   else weights.max(dim=0).values)
        token_weights = weights.mean(dim=0).cpu().numpy()

        return project_token_attention(token_weights, protein_tokens)


# ---------------------------------------------------------------------------

def implementation_status() -> dict:
    """Which baseline adapters are ready. Prints a checklist for Track A."""
    from src.evaluation.model_registry import _REGISTRY

    status = {}
    for name, cls in _REGISTRY.items():
        stub = issubclass(cls, _Unimplemented)
        status[name] = {
            "implemented": not stub,
            "provides_attention": getattr(cls, "provides_attention", True),
            "auditable_for_explanations": (not stub)
            and getattr(cls, "provides_attention", True),
        }
    return status


if __name__ == "__main__":
    print(f"{'model':24s} {'implemented':>12s} {'attention':>10s} {'auditable':>10s}")
    print("-" * 60)
    for name, info in sorted(implementation_status().items()):
        print(f"{name:24s} {str(info['implemented']):>12s} "
              f"{str(info['provides_attention']):>10s} "
              f"{str(info['auditable_for_explanations']):>10s}")
    print("\nOnly 'auditable' models contribute to the explanation-quality "
          "result.\nDeepDTA is intentionally not auditable: no attention, "
          "accuracy anchor only.")
    print("\nNone of these has been run against a live checkpoint. Before "
          "trusting any number:\n"
          "    python -m src.evaluation.check_adapters")
