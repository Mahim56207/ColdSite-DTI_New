"""
More explanation methods for the same checkpoints (T08, amendment family E3).

Attention and integrated gradients are two ways to ask a trained model where it looks.
This module adds three more, each registered as a model in its own right that reads its
base model's checkpoint, so every ladder, null, faithfulness run and audit applies
unchanged (`python -m src.evaluation.run_ladder --model moltrans_occlusion ...`):

    <model>_occlusion     leave a window of residues out and read the drop in the model's
                          own score. Needs nothing but `predict`, so it applies to every
                          model that has residues to mask.
    <model>_attngrad      attention x gradient (Chefer et al. 2021): the readout's own
                          attention tensor, weighted by the positive part of the gradient
                          of the model's score with respect to it, then reduced exactly
                          as the plain readout is reduced. Needs the attention tensor to
                          sit on the path to the score.
    moltrans_rollout      attention rollout (Abnar & Zuidema 2020): the per-layer
                          attention, with the residual connection added, multiplied
                          through the stack. Needs several attention layers with
                          residuals.

Where a method is not defined for a model it is NOT implemented, and the reason is
written in `docs/method_applicability.md`; nothing here fabricates a substitute.

Choices that change the numbers, declared here so they are not tuned afterwards
(amendment addendum A2 seals them before any score is computed):

* occlusion window 5 residues, stride 2 (`WINDOW`, `STRIDE`); a residue's score is the
  mean drop over the windows that cover it; the score is the **magnitude** of the drop,
  as integrated gradients uses magnitude (precision@k asks which residues the explanation
  points at, a question with no sign). `signed=True` keeps the direction.
* occluded residues become `X` through each model's own tokeniser (UNK for ColdSite-DTI),
  the intervention faithfulness already uses (`residue_space.py`).
* MolTrans runs every forward pass under one fixed RNG seed, because its vendored forward
  leaves dropout live at inference (`residue_space.PREDICT_SEED`); without it the drop
  would measure dropout.
* attention x gradient and rollout reduce heads/channels and query rows the way the base
  readout does (mean), so a difference from the plain readout is the method, not the
  reduction.
"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.model_registry import ExplainableDTIModel, _REGISTRY, register

WINDOW = 5
STRIDE = 2
BATCH = 32
MASK_TOKEN = 1            # ColdSite-DTI's UNK (faithfulness.MASK_TOKEN)
PREDICT_SEED = 0          # residue_space.PREDICT_SEED: one dropout draw for every pass


# --------------------------------------------------------------------------
# occlusion, model-independent part
# --------------------------------------------------------------------------

def occlusion_windows(n: int, window: int = WINDOW, stride: int = STRIDE) -> list:
    """(start, end) windows covering residues 0..n-1, the last one flush with the end."""
    if n <= 0:
        raise ValueError("cannot occlude an empty protein")
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if n <= window:
        return [(0, n)]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    return [(s, s + window) for s in starts]


def occlusion_from_scores(n: int, windows: list, base_score: float, masked_scores,
                          signed: bool = False) -> np.ndarray:
    """One weight per residue: the mean drop over the windows that cover it.

    A window whose score is NaN (a masked input the model could not encode) is left out
    of the average; a residue no valid window covers gets 0.
    """
    total = np.zeros(n)
    count = np.zeros(n)
    for (s, e), masked in zip(windows, masked_scores):
        if not np.isfinite(masked):
            continue
        total[s:e] += base_score - float(masked)
        count[s:e] += 1
    out = np.divide(total, count, out=np.zeros(n), where=count > 0)
    return out if signed else np.abs(out)


def occlusion_map(n: int, base_score: float, score_windows, window: int = WINDOW,
                  stride: int = STRIDE, signed: bool = False) -> np.ndarray:
    """`score_windows(list of (s, e))` -> the model's score with each window masked."""
    windows = occlusion_windows(n, window, stride)
    return occlusion_from_scores(n, windows, base_score, score_windows(windows), signed)


def _batchify(tensor):
    tensor = torch.as_tensor(tensor)
    return tensor if tensor.dim() == 2 else tensor.unsqueeze(0)


# --------------------------------------------------------------------------
# a variant that reads its base model's checkpoint
# --------------------------------------------------------------------------

class _MethodVariant(ExplainableDTIModel):
    """The base adapter's weights, tokenisation and prediction; only `explain` differs."""

    base_model: str = ""
    provides_attention = True          # it provides an explanation; not necessarily attention
    method = ""

    def __init__(self, *args, **kwargs):
        self.window = int(kwargs.pop("window", WINDOW))
        self.stride = int(kwargs.pop("stride", STRIDE))
        self.signed = bool(kwargs.pop("signed", False))
        self.base = _REGISTRY[self.base_model](*args, **kwargs)
        self.device = getattr(self.base, "device", "cpu")
        self.model = self.base.model
        self.citation = f"{self.method} over {self.base_model}"

    @classmethod
    def encode(cls, *args, **kwargs):
        return _REGISTRY[cls.base_model].encode(*args, **kwargs)

    def predict(self, *args, **kwargs) -> float:
        return self.base.predict(*args, **kwargs)


class _Occlusion(_MethodVariant):
    method = "occlusion"

    def _masked_scores(self, drug, protein, windows, **extra) -> tuple:
        """(base score, scores with each window occluded), same conditions for all."""
        raise NotImplementedError

    def _length(self, protein, **extra) -> int:
        raise NotImplementedError

    def explain(self, drug, protein, **extra) -> np.ndarray:
        n = self._length(protein, **extra)
        windows = occlusion_windows(n, self.window, self.stride)
        base, masked = self._masked_scores(drug, protein, windows, **extra)
        return occlusion_from_scores(n, windows, base, masked, self.signed)


def _masked_columns(protein_row: torch.Tensor, windows: list, code: int) -> torch.Tensor:
    """(len(windows), L): one copy of the row per window, that window set to `code`."""
    batch = protein_row.reshape(1, -1).repeat(len(windows), 1)
    for i, (s, e) in enumerate(windows):
        batch[i, s:e] = code
    return batch


@register("coldsite_dti_occlusion")
class ColdSiteDTIOcclusion(_Occlusion):
    base_model = "coldsite_dti"

    def _length(self, protein, **extra) -> int:
        from src.model.protein_encoder import real_lengths
        return int(real_lengths(_batchify(protein))[0].item())

    @torch.no_grad()
    def _masked_scores(self, drug, protein, windows, **extra):
        drug_b = _batchify(drug).to(self.device)
        row = _batchify(protein)[0]
        self.model.eval()
        base = float(self.model(drug_b, row.unsqueeze(0).to(self.device))[0].reshape(-1)[0])
        masked = _masked_columns(row, windows, MASK_TOKEN)
        out = []
        for i in range(0, len(windows), BATCH):
            chunk = masked[i:i + BATCH].to(self.device)
            pred, _ = self.model(drug_b.expand(chunk.shape[0], -1), chunk)
            out.extend(pred.reshape(-1).cpu().tolist())
        return base, np.asarray(out, dtype=float)


@register("hyperattentiondti_occlusion")
class HyperAttentionDTIOcclusion(_Occlusion):
    """Occludes to `X`, HyperAttentionDTI's own unknown-residue code."""

    base_model = "hyperattentiondti"

    def _length(self, protein, **extra) -> int:
        return int((_batchify(protein)[0] != 0).sum().item())

    def _x_code(self) -> int:
        return int(self.encode("C", "X")[1][0])

    @torch.no_grad()
    def _masked_scores(self, drug, protein, windows, **extra):
        drug_b = _batchify(drug).to(self.device)
        row = _batchify(protein)[0]
        code = self._x_code()
        self.model.eval()

        def log_odds(protein_batch):
            logits = self.model(drug_b.expand(protein_batch.shape[0], -1),
                                protein_batch.to(self.device))
            return (logits[:, 1] - logits[:, 0]).cpu().tolist()

        base = log_odds(row.unsqueeze(0))[0]
        masked = _masked_columns(row, windows, code)
        out = []
        for i in range(0, len(windows), BATCH):
            out.extend(log_odds(masked[i:i + BATCH]))
        return base, np.asarray(out, dtype=float)


@register("drugban_occlusion")
class DrugBANOcclusion(_Occlusion):
    """Occludes to `X` (index 24 of DrugBAN's protein table). One pass per window: its drug
    side is a DGL graph, so it is not batched here. NOT run on the machine that wrote it
    (no DGL); the generic occlusion machinery it shares is what the tests exercise."""

    base_model = "drugban"

    def _length(self, protein, **extra) -> int:
        from src.evaluation.drugban_adapter import _real_residue_count
        return _real_residue_count(protein)

    def _masked_scores(self, drug, protein, windows, **extra):
        row = _batchify(protein)[0]
        code = int(self.encode("C", "X")[1][0])
        base = float(self.base.predict(drug, row.unsqueeze(0)))
        masked = _masked_columns(row, windows, code)
        return base, np.asarray(
            [float(self.base.predict(drug, masked[i].unsqueeze(0)))
             for i in range(len(windows))], dtype=float)


@register("moltrans_occlusion")
class MolTransOcclusion(_Occlusion):
    """Occludes residues of the sequence, not tokens, and re-tokenises: masking a residue
    re-segments the protein, exactly as in `residue_space`. `protein_tokens` is required
    (it carries the sequence and how many residues the model saw), as for its attention."""

    base_model = "moltrans"

    def _length(self, protein, protein_tokens=None, **extra) -> int:
        if protein_tokens is None:
            raise ValueError("MolTrans occlusion needs `protein_tokens`, as its attention does")
        from src.evaluation.attention_projection import moltrans_covered_residues
        return moltrans_covered_residues(protein_tokens)

    def _masked_scores(self, drug, protein, windows, protein_tokens=None, drug_mask=None,
                       protein_mask=None, **extra):
        sequence = "".join(protein_tokens)
        devices = []
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            devices = [torch.device(self.device).index or 0]

        def score(p, pm):
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(PREDICT_SEED)
                return self.base.predict(drug, p, drug_mask=drug_mask, protein_mask=pm)

        base = score(protein, protein_mask)
        out = []
        for s, e in windows:
            masked = sequence[:s] + "X" * (e - s) + sequence[e:]
            p, pm = type(self.base).encode_protein(masked)
            p, pm = torch.as_tensor(np.asarray(p)), torch.as_tensor(np.asarray(pm))
            # ESPF returns the single token [0] for the whole protein when a subword is
            # missing from its index; that is a prediction for an empty protein.
            if int(pm.sum()) == 1 and int(p[0]) == 0 and len(masked) > 1:
                out.append(float("nan"))
                continue
            out.append(score(p, pm))
        return base, np.asarray(out, dtype=float)


# --------------------------------------------------------------------------
# attention x gradient
# --------------------------------------------------------------------------

def hyperattentiondti_gate_and_grad(model, drug_b, protein_b, perturb=None):
    """(gate, d log-odds / d gate), both (1, channels, positions), for one pair.

    The gate is the second call of the model's `sigmoid` module (the first is the drug's).
    `perturb(gate) -> gate` lets a test nudge one element to check the gradient by finite
    differences; it is never used by the explainer.
    """
    model.eval()
    holder: dict = {}

    def hook(_m, _i, out):
        calls = holder.setdefault("n", [])
        calls.append(out)
        if len(calls) == 2:
            new = out if perturb is None else perturb(out)
            new.retain_grad()
            holder["gate"] = new
            return new
        return None

    handle = model.sigmoid.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(drug_b, protein_b).reshape(-1)
            score = logits[1] - logits[0]
            score.backward()
    finally:
        handle.remove()
    if len(holder.get("n", [])) != 2:
        raise RuntimeError(f"expected two sigmoid calls (drug, protein), saw "
                           f"{len(holder.get('n', []))}: the vendored forward changed")
    gate = holder["gate"]
    if gate.grad is None:
        raise RuntimeError("no gradient reached the attention gate")
    model.zero_grad(set_to_none=True)
    return gate.detach(), gate.grad.detach(), float(score.detach())


@register("hyperattentiondti_attngrad")
class HyperAttentionDTIAttnGrad(_MethodVariant):
    """Its channel gate over convolution positions, weighted by the gradient of the
    log-odds (logit1 - logit0, what `predict` reports), averaged over channels like the
    plain readout. No vendored code is edited."""

    base_model = "hyperattentiondti"
    method = "attention x gradient"

    def explain(self, drug, protein, **extra) -> np.ndarray:
        from src.evaluation.attention_projection import project_conv_attention

        drug_b, protein_b = _batchify(drug).to(self.device), _batchify(protein).to(self.device)
        real_length = int((protein_b[0] != 0).sum().item())
        if real_length == 0:
            raise ValueError("protein tensor is entirely padding")
        gate, grad, _ = hyperattentiondti_gate_and_grad(self.model, drug_b, protein_b)
        weights = torch.relu(gate * grad).mean(dim=1)[0]
        return project_conv_attention(weights.cpu().numpy(), real_length,
                                      mode=self.base.projection_mode)


def _moltrans_forward_scalar(adapter, drug, protein, drug_mask, protein_mask):
    d, p = _batchify(drug), _batchify(protein)
    dm = _batchify(drug_mask) if drug_mask is not None else (d != 0).long()
    pm = _batchify(protein_mask) if protein_mask is not None else (p != 0).long()
    adapter._fit_batch_size(d.shape[0])
    out = adapter.model(d.to(adapter.device), p.to(adapter.device),
                        dm.to(adapter.device), pm.to(adapter.device))
    return out.reshape(-1)[0]


def moltrans_probs_and_grad(adapter, drug, protein, drug_mask=None, protein_mask=None,
                            layer: int = -1, perturb=None):
    """(attention probabilities, d score / d probabilities, score) of one protein-encoder
    layer, each (1, heads, T, T). The probabilities are the input of that layer's own
    dropout module, where they are on the path to the output. `perturb(probs) -> probs`
    is for tests (finite differences) only."""
    dropout = adapter.model.p_encoder.layer[layer].attention.self.dropout
    holder: dict = {}

    def pre(_m, inputs):
        probs = inputs[0] if perturb is None else perturb(inputs[0])
        probs.retain_grad()
        holder["probs"] = probs
        return (probs,) + tuple(inputs[1:])

    adapter.model.eval()
    handle = dropout.register_forward_pre_hook(pre)
    try:
        with torch.enable_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(PREDICT_SEED)
            adapter.model.zero_grad(set_to_none=True)
            score = _moltrans_forward_scalar(adapter, drug, protein, drug_mask, protein_mask)
            score.backward()
    finally:
        handle.remove()
    probs = holder.get("probs")
    if probs is None or probs.grad is None:
        raise RuntimeError("no gradient reached the attention probabilities")
    adapter.model.zero_grad(set_to_none=True)
    return probs.detach(), probs.grad.detach(), float(score.detach())


@register("moltrans_attngrad")
class MolTransAttnGrad(_MethodVariant):
    """The last protein-encoder layer's attention probabilities (the plain readout's layer),
    weighted by the gradient of the model's score with respect to them, heads averaged and
    then queries averaged as the plain readout does."""

    base_model = "moltrans"
    method = "attention x gradient"

    def explain(self, drug, protein, protein_tokens=None, drug_mask=None,
                protein_mask=None, **extra) -> np.ndarray:
        from src.evaluation.attention_projection import project_token_attention
        if protein_tokens is None:
            raise ValueError("MolTrans attention x gradient needs `protein_tokens`")
        probs, grad, _ = moltrans_probs_and_grad(self.base, drug, protein, drug_mask,
                                                 protein_mask)
        weighted = torch.relu(probs * grad)[0].mean(dim=0)          # (T, T)
        return project_token_attention(weighted.mean(dim=0).cpu().numpy(), protein_tokens)


# --------------------------------------------------------------------------
# attention rollout
# --------------------------------------------------------------------------

def rollout(layer_attention: list) -> np.ndarray:
    """Attention rollout of a list of (T, T) head-averaged attention matrices, first layer
    first: each layer contributes `0.5 A + 0.5 I`, rows re-normalised, and the layers are
    multiplied, later layers on the left (Abnar & Zuidema 2020, eq. 2)."""
    if not layer_attention:
        raise ValueError("rollout needs at least one layer")
    total = None
    for attention in layer_attention:
        a = np.asarray(attention, dtype=float)
        a = 0.5 * a + 0.5 * np.eye(a.shape[-1])
        a = a / a.sum(axis=-1, keepdims=True)
        total = a if total is None else a @ total
    return total


@register("moltrans_rollout")
class MolTransRollout(_MethodVariant):
    """Rollout through every protein-encoder layer (MolTrans's encoder is a stack with
    residual connections). Reduced to one weight per token the way the plain readout is:
    mean over query rows."""

    base_model = "moltrans"
    method = "attention rollout"

    def explain(self, drug, protein, protein_tokens=None, drug_mask=None,
                protein_mask=None, **extra) -> np.ndarray:
        from src.evaluation.attention_projection import (capture_moltrans_attention,
                                                         project_token_attention)
        if protein_tokens is None:
            raise ValueError("MolTrans rollout needs `protein_tokens`")
        layers = self.model.p_encoder.layer
        stores, handles = [], []
        for layer in layers:
            store, handle = capture_moltrans_attention(layer.attention.self)
            stores.append(store)
            handles.append(handle)
        try:
            self.model.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                torch.manual_seed(PREDICT_SEED)
                _moltrans_forward_scalar(self.base, drug, protein, drug_mask, protein_mask)
        finally:
            for handle in handles:
                handle.remove()
        if any(s.get("probs") is None for s in stores):
            raise RuntimeError("a layer's attention hook captured nothing")
        per_layer = [s["probs"][0].mean(dim=0).cpu().numpy() for s in stores]   # (T, T)
        token_weights = rollout(per_layer).mean(axis=0)
        return project_token_attention(token_weights, protein_tokens)


# --------------------------------------------------------------------------
# which method applies to which model (mirrored in docs/method_applicability.md)
# --------------------------------------------------------------------------

# (method, base model) -> registered name of the explainer, or None with the reason.
# "attention" and "integrated_gradients" are the existing methods, listed for the
# agreement analysis; the other three are this module's.
APPLICABILITY = {
    ("occlusion", "coldsite_dti"): "coldsite_dti_occlusion",
    ("occlusion", "hyperattentiondti"): "hyperattentiondti_occlusion",
    ("occlusion", "moltrans"): "moltrans_occlusion",
    ("occlusion", "drugban"): "drugban_occlusion",
    ("attention_x_gradient", "hyperattentiondti"): "hyperattentiondti_attngrad",
    ("attention_x_gradient", "moltrans"): "moltrans_attngrad",
    ("attention_x_gradient", "coldsite_dti"): None,
    ("attention_x_gradient", "drugban"): None,
    ("rollout", "moltrans"): "moltrans_rollout",
    ("rollout", "coldsite_dti"): None,
    ("rollout", "hyperattentiondti"): None,
    ("rollout", "drugban"): None,
}
NOT_APPLICABLE_REASON = {
    ("attention_x_gradient", "coldsite_dti"):
        "the cross-attention weights it returns are a head-averaged copy that never feeds "
        "the prediction (nn.MultiheadAttention computes its output from the per-head "
        "probabilities internally), so no gradient with respect to them exists; getting one "
        "means re-implementing the attention forward, which is a different computation",
    ("attention_x_gradient", "drugban"):
        "not implemented: the bilinear map's path to the score needs DGL to inspect, and "
        "DGL is not installed on the machine that wrote this module; deferred, not "
        "judged impossible",
    ("rollout", "coldsite_dti"):
        "rollout composes attention across a stack of layers with residual connections; the "
        "explanation here is one cross-attention read by a single query, and the protein "
        "tower's self-attention is one layer, so there is nothing to compose",
    ("rollout", "hyperattentiondti"):
        "no stacked softmax attention: the attention is a single sigmoid channel gate over "
        "convolution positions",
    ("rollout", "drugban"):
        "one bilinear attention map, no stack of attention layers",
}
