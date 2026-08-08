"""
Baseline adapters — skeletons for 124AD0008 to complete.

Each class below is a real, registered adapter that currently raises
NotImplementedError with instructions. They are registered deliberately: the
audit runner then lists them as available and fails with a useful message
naming the file and method to fill in, rather than an opaque KeyError.

Fill one in, then immediately run:

    from src.evaluation.model_registry import validate_adapter
    print(validate_adapter(adapter, drug, protein, expected_length=len(sequence)))

`valid: True` is the gate. The failure mode that matters here is not a crash --
it is an adapter that returns a plausible array of the wrong length, which
misaligns every ground-truth index and produces a wrong precision@k that looks
entirely reasonable.

See docs/PART2_GUIDE_124AD0008.md Priority 5.
"""
import numpy as np

from src.evaluation.model_registry import ExplainableDTIModel, register


class _Unimplemented(ExplainableDTIModel):
    """Shared scaffolding so each stub says something useful when called."""

    repo_url = ""
    what_to_do = ""

    def __init__(self, checkpoint_path: str = None, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device

    def _explain_error(self, method: str):
        return NotImplementedError(
            f"\n{type(self).__name__}.{method}() is not implemented yet.\n"
            f"  repo:  {self.repo_url}\n"
            f"  file:  src/evaluation/baseline_adapters.py\n"
            f"  todo:  {self.what_to_do}\n"
            f"  then:  validate_adapter(adapter, drug, protein, "
            f"expected_length=len(sequence))\n"
        )

    def predict(self, drug, protein) -> float:
        raise self._explain_error("predict")

    def explain(self, drug, protein) -> np.ndarray:
        raise self._explain_error("explain")


@register("deepdta")
class DeepDTAAdapter(_Unimplemented):
    """DeepDTA — CNN encoders, no attention.

    provides_attention = False is correct and not a gap. DeepDTA anchors the
    accuracy axis of the audit, letting a reviewer see whether the
    interpretable models pay an accuracy cost for their explanations. Leave
    explain() raising.
    """

    provides_attention = False
    citation = "Ozturk et al., Bioinformatics 2018"
    repo_url = "https://github.com/hkmztrk/DeepDTA"
    what_to_do = (
        "implement predict() only -- load the trained model and return the "
        "scalar affinity. explain() should keep raising; DeepDTA has no "
        "attention to expose."
    )


@register("hyperattentiondti")
class HyperAttentionDTIAdapter(_Unimplemented):
    """HyperAttentionDTI — attention over CONVOLUTION positions, not residues.

    Corrected against the vendored implementation (124AD0015, Part 2). An
    earlier version of this docstring said the attention is 2D over drug x
    protein positions and should be reduced with max over the drug axis. That
    is not what `baselines/HpyerAttentionDTI/model.py` computes:

        Atten_matrix   (B, 85, 979, 160)   drug pos x protein pos x channel
        Protein_atte   (B, 160, 979)       AFTER torch.mean(Atten_matrix, 1)

    The drug axis is averaged away inside `forward`, so the axis left to reduce
    is 160 channels. Taking a max over drug positions would mean recomputing
    from `Atten_matrix` — ~13M floats per sample. That is a real choice, not an
    oversight; see attention_projection.PROJECTION_DECISIONS.

    The 979 is the second trap: a [4, 8, 12] kernel stack turns 1000 residues
    into 979 positions, each a 22-residue window. Returning those 979 values as
    if they were residues misaligns every ground-truth index.

    `src/evaluation/attention_projection.py` has both steps, tested against the
    vendored model.
    """

    provides_attention = True
    citation = "Zhao et al., Bioinformatics 2022"
    repo_url = "https://github.com/kexinhuang12345/HyperAttentionDTI"
    what_to_do = (
        "implement predict() and explain(). For explain(), the extraction and "
        "the residue projection already exist:\n"
        "    from src.evaluation.attention_projection import (\n"
        "        hyperattentiondti_protein_attention, project_conv_attention)\n"
        "    conv = hyperattentiondti_protein_attention(model, drug, protein)\n"
        "    weights = project_conv_attention(conv[0], real_length)\n"
        "so what is left on this side is loading the checkpoint, tokenising "
        "with CHARPROTSET, and passing the REAL residue count (not 1000, and "
        "not 979)."
    )


@register("moltrans")
class MolTransAdapter(_Unimplemented):
    """MolTrans — interaction map over substructure pairs.

    Harder than it looks: MolTrans attends over ESPF substructure tokens, not
    residues. One token spans several amino acids, so the interaction map must
    be projected back onto residue indices using the ESPF decomposition before
    precision@k means anything. Skipping that projection produces a
    well-formed array indexed by the wrong thing.

    Two findings from the vendored code (124AD0015, Part 2):

    * `SelfAttention.forward` computes `attention_probs` and returns only
      `context_layer`, so the weights are unreachable without a hook. Use
      `attention_projection.capture_moltrans_attention` rather than editing
      the vendored file — prediction behaviour then stays byte-identical to
      upstream. Extract in eval mode: dropout is applied to `attention_probs`.
    * `max_protein_seq = 545` counts TOKENS, not residues, and a token spans
      several amino acids. `moltrans_covered_residues(tokens)` is the real
      length; 545 is never it.
    """

    provides_attention = True
    citation = "Huang et al., Bioinformatics 2021"
    repo_url = "https://github.com/kexinhuang12345/MolTrans"
    what_to_do = (
        "implement predict() and explain(). The token->residue projection "
        "already exists:\n"
        "    from src.evaluation.attention_projection import (\n"
        "        project_token_attention, moltrans_covered_residues)\n"
        "    weights = project_token_attention(token_weights, tokens)\n"
        "so what is left on this side is loading the checkpoint, running the "
        "ESPF tokeniser (baselines/MolTrans/stream.py, needs subword_nmt, "
        "which is NOT in requirements.txt), and deciding which attention to "
        "expose -- see attention_projection.PROJECTION_DECISIONS["
        "'moltrans.attention_source']."
    )


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
          "result.\nSee docs/PART2_GUIDE_124AD0008.md Priority 5.")
