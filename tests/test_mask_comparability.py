"""Are the two arms of the masking test the same size of intervention?

The faithfulness delta subtracts a random-masking control from the explanation's
comprehensiveness, which is only meaningful if both arms change the input by the same
amount. For MolTrans they do not, and the consequence was a negative delta in 11 of 12
cells that would have read as "its attention points at residues that matter less than
arbitrary ones". These tests pin the measurement and the matched control on sequences
whose answer is known.
"""
import numpy as np
import pytest

from src.evaluation.mask_comparability import (TOLERANCE, residue_level_models_are_unaffected,
                                               report, token_change_fraction,
                                               token_matched_control)


def _fake_encode(returns):
    """An encoder whose token sequence depends on the masked sequence in a known way."""
    def encode(_smiles, sequence):
        tokens = returns(sequence)
        arr = np.asarray(tokens, dtype=int)
        mask = np.ones_like(arr)
        return None, None, arr, mask, tokens
    return encode


def test_no_masking_changes_nothing():
    encode = _fake_encode(lambda s: [ord(c) for c in s])
    assert token_change_fraction(encode, "ABCDE", []) == 0.0


def test_one_residue_in_a_residue_level_encoder_is_one_token():
    encode = _fake_encode(lambda s: [ord(c) for c in s])
    assert token_change_fraction(encode, "ABCDE", [2]) == pytest.approx(1 / 5)


def test_a_re_segmentation_that_shortens_the_sequence_is_not_free():
    """A masked residue that collapses two tokens into one must count as a change, or the
    biggest interventions would score as the smallest."""
    def segment(s):
        return [1] * (len(s) // 2) if "X" in s else [1] * len(s)

    encode = _fake_encode(segment)
    assert token_change_fraction(encode, "ABCDEF", [0]) == pytest.approx(0.5)


def test_a_cascading_encoder_shows_the_asymmetry_the_module_exists_to_report():
    """Masking early cascades through everything after it; masking at the end does not."""
    def cascade(s):
        first = s.find("X")
        return [0] * len(s) if first < 0 else [0] * first + [9] * (len(s) - first)

    encode = _fake_encode(cascade)
    at_end = token_change_fraction(encode, "A" * 100, [99])
    at_start = token_change_fraction(encode, "A" * 100, [0])
    assert at_end == pytest.approx(0.01)
    assert at_start == pytest.approx(1.0)
    assert at_start > 50 * at_end


def _cascade_encoder():
    def cascade(s):
        first = s.find("X")
        return [0] * len(s) if first < 0 else [0] * first + [9] * (len(s) - first)
    return _fake_encode(cascade)


def test_the_matched_control_hits_a_reachable_target():
    """The control matches the explanation's intervention size, not its residue count."""
    encode = _cascade_encoder()
    sequence = "A" * 200
    target = token_change_fraction(encode, sequence, [100])          # half the tokens
    positions, fraction, tries = token_matched_control(
        encode, sequence, k=1, target=target, rng=np.random.default_rng(0))
    assert abs(fraction - target) <= TOLERANCE, (fraction, target)
    assert len(positions) == 1 and tries >= 1


def test_the_matched_control_is_closer_to_the_explanation_than_a_blind_draw():
    """The property that matters even when an exact match is unreachable: 3 random
    residues in 200 almost never fall in the last tenth, so the blind arm is a much
    bigger intervention -- which is exactly the artefact this module reports."""
    encode = _cascade_encoder()
    sequence = "A" * 200
    rng = np.random.default_rng(0)
    target = token_change_fraction(encode, sequence, [180, 185, 190])
    _positions, matched, _tries = token_matched_control(
        encode, sequence, k=3, target=target, rng=np.random.default_rng(1))
    blind = float(np.mean([
        token_change_fraction(encode, sequence, rng.choice(200, 3, replace=False))
        for _ in range(20)]))
    assert abs(matched - target) < abs(blind - target)


def test_the_matched_control_reports_when_it_could_not_match():
    """Better a control that says it failed than one that pretends to be matched."""
    encode = _fake_encode(lambda s: [ord(c) for c in s])
    positions, fraction, tries = token_matched_control(
        encode, "ABCDEFGHIJ", k=2, target=0.99,        # unreachable: 2 of 10 tokens
        rng=np.random.default_rng(0), max_tries=15)
    assert tries == 15
    assert abs(fraction - 0.99) > TOLERANCE
    assert len(positions) == 2


def test_residue_level_models_are_unaffected_by_construction():
    out = residue_level_models_are_unaffected("ACDEFGHIKLMNPQ", k=3)
    assert out["same_length"] is True
    assert out["positions_changed"] == 3          # exactly the residues masked


def test_the_report_states_the_ratio_and_the_consequence():
    records = [{"target": "P1", "tokens": 100, "attended": 0.02, "random": 0.95,
                "ratio": 47.5, "matched": 0.03, "matched_tries": 4}]
    text = report(records, "moltrans")
    assert "2.0%" in text and "95.0%" in text and "47.5x" in text
    assert "token-matched control" in text
    assert "not as an anti-faithful explanation" in text


# ---------------------------------------------------------------------------
# protein_token_encoder: the fast path must be the same control, only cheaper
# ---------------------------------------------------------------------------

class _CountingAdapter:
    """An adapter with both paths, counting how often each protein is encoded."""
    calls = []

    @staticmethod
    def _tokens(sequence):
        first = sequence.find("X")                       # the cascading encoder again
        return [0] * len(sequence) if first < 0 else [0] * first + [9] * (len(sequence) - first)

    @staticmethod
    def encode(_smiles, sequence):
        arr = np.asarray(_CountingAdapter._tokens(sequence))
        return "drug", "drug_mask", arr, np.ones_like(arr), ["tok"]

    @staticmethod
    def encode_protein(sequence):
        _CountingAdapter.calls.append(sequence)
        arr = np.asarray(_CountingAdapter._tokens(sequence))
        return arr, np.ones_like(arr)


def test_the_fast_encoder_gives_the_same_control_draw_for_draw():
    """Same fractions, same positions, same number of tries -- from the same seed. That is
    what makes the 2026-09-16 speed-up safe to use on KIBA after DAVIS ran without it."""
    from src.evaluation.mask_comparability import protein_token_encoder
    sequence = "A" * 300
    target = token_change_fraction(_CountingAdapter.encode, sequence, [250, 260, 270])
    slow = token_matched_control(_CountingAdapter.encode, sequence, 3, target,
                                 np.random.default_rng(7), max_tries=50)
    fast = token_matched_control(protein_token_encoder(_CountingAdapter), sequence, 3,
                                 target, np.random.default_rng(7), max_tries=50)
    assert list(slow[0]) == list(fast[0])
    assert slow[1] == fast[1] and slow[2] == fast[2]


def test_the_fast_encoder_encodes_the_unmasked_protein_once():
    from src.evaluation.mask_comparability import protein_token_encoder

    class OnePerResidue(_CountingAdapter):
        @staticmethod
        def _tokens(sequence):                     # 3 masked residues = 1% of 300 tokens,
            return [ord(c) for c in sequence]      # so a target of 99% is unreachable

        @staticmethod
        def encode_protein(sequence):
            _CountingAdapter.calls.append(sequence)
            arr = np.asarray(OnePerResidue._tokens(sequence))
            return arr, np.ones_like(arr)

    _CountingAdapter.calls = []
    sequence = "A" * 300
    _pos, _frac, tries = token_matched_control(protein_token_encoder(OnePerResidue),
                                               sequence, 3, 0.99,
                                               np.random.default_rng(0), max_tries=40)
    assert tries == 40
    assert _CountingAdapter.calls.count(sequence) == 1, "the unmasked protein was re-encoded"
    assert len(_CountingAdapter.calls) == 41                           # 1 + one per try


def test_an_adapter_without_a_protein_path_falls_back_to_its_encode():
    from src.evaluation.mask_comparability import protein_token_encoder

    class Plain:
        @staticmethod
        def encode(_smiles, sequence):
            return None, None, np.zeros(3), np.ones(3), []

    assert protein_token_encoder(Plain) is Plain.encode


def test_moltrans_protein_path_is_its_encode_on_real_sequences():
    """The real tokeniser: encode_protein must return encode's protein arrays exactly,
    masked or not. Skipped where the vendored MolTrans or subword_nmt is absent."""
    pytest.importorskip("subword_nmt")
    from src.evaluation.baseline_adapters import MolTransAdapter
    try:
        MolTransAdapter.encode("C", "MKV")
    except Exception as exc:                                  # vendored repo missing
        pytest.skip(f"MolTrans not available: {exc}")
    sequence = "MSGPRAGFYRQELNKTVWEVPQRLQGLRPVGSGAYGSVCSAYDARLRQKVAVKKLSRPFQSLIHARRTYRELRLLKHLKHENVIGLLDVFTPATSIEDFSEVYLVTTLMGADLNNIVKCQALSDEHVQFLVYQLLRGLKYIHSAGIIHRDLKPSNVAVNEDCELRILDFGLARQADEEMTGYVATRWYRAPEIMLNWMHYNQTVDIWSVGCIMAELLQGKALFPGSDYIDQLKRIMEVVGTPSPEVLAKISSEHARTYIQSLPPMPQKDLSSIFRGANPLAIDLLGRMLVLDSDQRVSAAEALAHAYFSQYHDPEDEPEAEPYDESVEAKERTLEEWKELTYQEVLSFKPPEPPKPPGSLEIEQ"
    masked = list(sequence)
    for i in (3, 40, 41, 150, 299):
        masked[i] = "X"
    for seq in (sequence, "".join(masked)):
        _d, _dm, p, pm, _t = MolTransAdapter.encode("CCO", seq)
        fp, fpm = MolTransAdapter.encode_protein(seq)
        assert np.array_equal(np.asarray(p), np.asarray(fp))
        assert np.array_equal(np.asarray(pm), np.asarray(fpm))
