"""
A second explanation method, so the audit is not a verdict on attention alone.

Why
---
Every result so far scores a model's **attention**. If attention misses the binding site,
two very different things could be true: the attention is a poor explanation of a model
that does know where the drug binds, or the model never learned the site in the first
place. Those have opposite implications -- the first is a criticism of attention as an
explanation, the second of the models -- and no attention measurement can separate them.

Integrated gradients separate them. The attribution is computed from the trained weights
and the prediction, not from any interpretability head: if IG finds the annotated
residues where attention does not, the information was in the model and attention failed
to report it; if neither finds them, the models do not represent the site.

    IG_i = (x_i - x'_i) . mean_alpha grad_x f(x' + alpha (x - x'))      (Sundararajan 2017)

How it is done here
-------------------
Protein input is token ids, which are not differentiable, so the path runs over the
**protein embedding**: the real embedding is the endpoint, and the baseline is the
embedding of the padding token -- the same "no residue here" the faithfulness masking
uses, rather than an arbitrary zero. `PROTEIN_EMBEDDING` names the module per model;
`tests/test_integrated_gradients.py` checks each name still resolves, because a silently
wrong layer would produce attributions over the drug or over nothing.

Per position the attribution is summed over embedding dimensions and its **magnitude** is
taken: attention weights are non-negative and precision@k asks "which residues does this
explanation point at", a question with no sign. `signed=True` keeps the sign for anyone
who wants direction instead.

MolTrans reads sub-word tokens, not residues, so its token attributions are projected
onto residues by `src/evaluation/attention_projection.py` -- the same projection its
attention goes through, so the two explanations are scored in one coordinate frame.

The result is registered as a model in its own right (`coldsite_dti_ig`,
`hyperattentiondti_ig`, `moltrans_ig`), reusing the same checkpoints: every ladder,
faithfulness run, null and audit applies to it unchanged.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from src.evaluation.model_registry import ExplainableDTIModel, _REGISTRY, register

# The protein-side embedding of each model, by dotted attribute path.
PROTEIN_EMBEDDING = {
    "coldsite_dti": "protein_encoder.embedding",
    "hyperattentiondti": "protein_embed",
    "moltrans": "pemb.word_embeddings",
}
DEFAULT_STEPS = 32
PAD_INDEX = 0


def embedding_module(model, path: str):
    """The nn.Embedding named by a dotted path, or a clear error naming what was found."""
    node = model
    for part in path.split("."):
        if not hasattr(node, part):
            raise AttributeError(
                f"{path!r} does not resolve on {type(model).__name__}: no {part!r}. "
                f"Update PROTEIN_EMBEDDING in src/evaluation/integrated_gradients.py")
        node = getattr(node, part)
    if not isinstance(node, torch.nn.Embedding):
        raise TypeError(f"{path!r} is {type(node).__name__}, not an nn.Embedding")
    return node


@contextlib.contextmanager
def _substituted(module):
    """Let the caller replace this embedding's output with a tensor of its own.

    Yields a dict: read `captured` after a forward pass to get the real embedding, and
    set `replacement` to have the next forward pass use that tensor instead. A hook is
    the only way in without editing three vendored models.
    """
    state = {"captured": None, "replacement": None}

    def hook(_module, _inputs, output):
        state["captured"] = output.detach()
        return output if state["replacement"] is None else state["replacement"]

    handle = module.register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def attributions(predict, model, protein_embedding_path: str, steps: int = DEFAULT_STEPS,
                 signed: bool = False, pad_index: int = PAD_INDEX) -> np.ndarray:
    """Integrated gradients over the protein embedding for one pair.

    `predict` is a zero-argument callable running the model's own forward pass and
    returning one scalar tensor -- the quantity being explained, which must be the same
    quantity the model's `predict` reports, or the explanation belongs to another
    question. Returns one attribution per protein token.
    """
    module = embedding_module(model, protein_embedding_path)
    was_training = model.training
    model.eval()
    # cuDNN refuses to backpropagate through an RNN in eval mode ("cudnn RNN backward can
    # only be called in training mode"), which killed every ColdSite-DTI job on Kaggle
    # while passing here, because a CPU has no cuDNN. Disabling cuDNN for the attribution
    # uses PyTorch's own RNN kernels instead: same maths, and eval mode is kept, so
    # dropout stays off and the attributions still belong to the model as it predicts.
    with torch.backends.cudnn.flags(enabled=False), _substituted(module) as state:
        with torch.no_grad():                      # pass 1: the real embedding
            predict()
        real = state["captured"]
        if real is None:
            raise RuntimeError("the protein embedding was never reached by this forward "
                               "pass -- is PROTEIN_EMBEDDING pointing at the drug side?")
        baseline = module.weight[pad_index].detach().to(real.device)
        baseline = baseline.expand_as(real)

        total = torch.zeros_like(real)
        for step in range(steps):
            alpha = (step + 0.5) / steps           # midpoint rule
            point = (baseline + alpha * (real - baseline)).requires_grad_(True)
            state["replacement"] = point
            model.zero_grad(set_to_none=True)
            output = predict()
            if output.numel() != 1:
                output = output.reshape(-1)[0]
            output.backward()
            if point.grad is None:
                raise RuntimeError("no gradient reached the protein embedding: the model "
                                   "may have been called under torch.no_grad()")
            total += point.grad.detach()
        state["replacement"] = None

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    integrated = (real - baseline) * (total / steps)
    per_token = integrated.sum(dim=-1).squeeze(0)
    return per_token.cpu().numpy() if signed else per_token.abs().cpu().numpy()


class IntegratedGradientsAdapter(ExplainableDTIModel):
    """Any registered model, explained by integrated gradients instead of attention.

    `predict` is delegated untouched, so accuracy and faithfulness deltas are the same
    model's; only `explain` changes. That is the point: two explanations of one
    checkpoint, scored by one pipeline.
    """

    base_model: str = ""
    provides_attention = True          # it provides an explanation; not from attention
    steps: int = DEFAULT_STEPS

    def __init__(self, *args, **kwargs):
        self.steps = int(kwargs.pop("ig_steps", self.steps))
        self.base = _REGISTRY[self.base_model](*args, **kwargs)
        self.device = getattr(self.base, "device", "cpu")
        self.model = self.base.model
        self.citation = f"integrated gradients over {self.base_model}"

    @property
    def embedding_path(self) -> str:
        return PROTEIN_EMBEDDING[self.base_model]

    @classmethod
    def encode(cls, *args, **kwargs):
        """The base adapter's own tokenisation: a variant must read the same tensors the
        model was trained on, or its attributions are indexed against another encoding."""
        return _REGISTRY[cls.base_model].encode(*args, **kwargs)

    def predict(self, drug, protein) -> float:
        return self.base.predict(drug, protein)

    def _forward_scalar(self, drug, protein):
        """The model's own scalar output, differentiably -- per model, like predict()."""
        raise NotImplementedError

    def explain(self, drug, protein) -> np.ndarray:
        raw = attributions(lambda: self._forward_scalar(drug, protein), self.model,
                           self.embedding_path, steps=self.steps)
        return self._to_residues(raw, protein)

    def _to_residues(self, weights: np.ndarray, protein) -> np.ndarray:
        """One weight per residue. Token-level models override this."""
        length = int(np.asarray(protein).reshape(-1).shape[0])
        return np.asarray(weights, dtype=float)[:length]


# --------------------------------------------------------------------------
# one subclass per model: the differentiable twin of its own predict()
# --------------------------------------------------------------------------

def _as_batch(tensor):
    tensor = torch.as_tensor(tensor)
    return tensor if tensor.dim() == 2 else tensor.unsqueeze(0)


@register("coldsite_dti_ig")
class ColdSiteDTIIntegratedGradients(IntegratedGradientsAdapter):
    """ColdSite-DTI's own prediction, explained by IG rather than its cross-attention."""

    base_model = "coldsite_dti"

    def _forward_scalar(self, drug, protein):
        pred, _attention = self.model(_as_batch(drug).to(self.device),
                                      _as_batch(protein).to(self.device))
        return pred.reshape(-1)[0]


@register("hyperattentiondti_ig")
class HyperAttentionDTIIntegratedGradients(IntegratedGradientsAdapter):
    """The published model, explained by IG.

    The explained quantity is the log-odds (logit1 - logit0), exactly what
    `HyperAttentionDTIAdapter.predict` returns and what its AUROC is computed on. The
    positive logit alone would let a constant shift count as a change the model never
    made.
    """

    base_model = "hyperattentiondti"

    def _forward_scalar(self, drug, protein):
        logits = self.model(_as_batch(drug).to(self.device),
                            _as_batch(protein).to(self.device)).reshape(-1)
        if logits.numel() != 2:
            raise RuntimeError(f"HyperAttentionDTI returned {logits.numel()} logits for "
                               f"one pair; expected 2 (binary head)")
        return logits[1] - logits[0]

    def _to_residues(self, weights, protein):
        """Its protein tower is a convolution stack, but IG attributes to the embedding,
        which is per residue already -- no projection, unlike its attention."""
        length = int((torch.as_tensor(protein).reshape(-1) != 0).sum())
        return np.asarray(weights, dtype=float)[:length]


@register("moltrans_ig")
class MolTransIntegratedGradients(IntegratedGradientsAdapter):
    """MolTrans, explained by IG over its sub-word token embeddings.

    Tokens are not residues, so the attributions go through the same projection its
    attention does (`attention_projection.project_token_attention`) and the two
    explanations end up in one coordinate frame. `protein_tokens` is therefore required,
    as it is for its attention.
    """

    base_model = "moltrans"

    def _forward_scalar(self, drug, protein, drug_mask=None, protein_mask=None):
        d, p = _as_batch(drug), _as_batch(protein)
        dm = _as_batch(drug_mask) if drug_mask is not None else (d != 0).long()
        pm = _as_batch(protein_mask) if protein_mask is not None else (p != 0).long()
        self.base._fit_batch_size(d.shape[0])
        out = self.model(d.to(self.device), p.to(self.device),
                         dm.to(self.device), pm.to(self.device))
        return out.reshape(-1)[0]

    def explain(self, drug, protein, protein_tokens=None, drug_mask=None,
                protein_mask=None) -> np.ndarray:
        if protein_tokens is None:
            raise ValueError(
                "MolTrans IG needs `protein_tokens`: its input is sub-word tokens, and "
                "without the tokens there is no way to know how many residues each one "
                "covered -- the same requirement its attention has")
        raw = attributions(
            lambda: self._forward_scalar(drug, protein, drug_mask, protein_mask),
            self.model, self.embedding_path, steps=self.steps)
        from src.evaluation.attention_projection import project_token_attention
        return np.asarray(project_token_attention(raw, protein_tokens), dtype=float)
