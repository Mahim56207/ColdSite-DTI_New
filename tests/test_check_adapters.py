"""The adapter gate must actually find the checkpoints the trainers write."""
from src.evaluation.check_adapters import find_checkpoint
from src.model.checkpoint_naming import checkpoint_path


def test_finds_the_names_the_trainers_write(tmp_path):
    written = checkpoint_path(str(tmp_path), "davis", "cold_target", "binary", 2,
                              model="moltrans")
    open(written, "wb").close()
    assert find_checkpoint(str(tmp_path), "moltrans") == written


def test_does_not_mistake_another_models_checkpoint(tmp_path):
    open(checkpoint_path(str(tmp_path), "davis", "random", "binary", 1,
                         model="hyperattentiondti"), "wb").close()
    assert find_checkpoint(str(tmp_path), "moltrans") is None


def test_hand_placed_legacy_name_still_works(tmp_path):
    legacy = tmp_path / "moltrans_best.pt"
    legacy.touch()
    assert find_checkpoint(str(tmp_path), "moltrans") == str(legacy)
