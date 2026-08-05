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
    """HyperAttentionDTI — attention over drug-protein position pairs.

    The attention is 2D (drug positions x protein positions). precision@k needs
    ONE weight per residue, so it has to be reduced along the drug axis. Use
    max over drug positions rather than mean: the claim being audited is that
    attention localises a binding pocket, and averaging over every drug atom
    smears a sharp pocket signal into a flat one, which would understate the
    model rather than test it. Record the choice in Methods either way.
    """

    provides_attention = True
    citation = "Zhao et al., Bioinformatics 2022"
    repo_url = "https://github.com/kexinhuang12345/HyperAttentionDTI"
    what_to_do = (
        "implement predict() and explain(). explain() must reduce the 2D "
        "attention map to one weight per protein residue (max over the drug "
        "axis) and return exactly len(real residues) values -- not the padded "
        "length."
    )


@register("moltrans")
class MolTransAdapter(_Unimplemented):
    """MolTrans — interaction map over substructure pairs.

    Harder than it looks: MolTrans attends over ESPF substructure tokens, not
    residues. One token spans several amino acids, so the interaction map must
    be projected back onto residue indices using the ESPF decomposition before
    precision@k means anything. Skipping that projection produces a
    well-formed array indexed by the wrong thing.
    """

    provides_attention = True
    citation = "Huang et al., Bioinformatics 2021"
    repo_url = "https://github.com/kexinhuang12345/MolTrans"
    what_to_do = (
        "implement predict() and explain(). explain() must map ESPF "
        "substructure attention back to residue positions -- each token covers "
        "a span of residues, so distribute its weight across that span "
        "(uniformly is fine; document the choice)."
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
