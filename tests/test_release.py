import json

from locket_decider import release
from locket_decider.decider import DeciderConfig


def report(acc):
    m = {"n": 100, "accuracy": acc, "macro_f1": acc - 0.01, "ece": 0.02, "nll": 0.3}
    return {"temperature": 1.4, "overall": {**m, "selective": {"threshold": 0.8, "coverage": 0.9, "accuracy": 0.96}}, "by_source": {"banking77": m}}


def test_card_reports_only_given_numbers(tmp_path):
    model = tmp_path / "model"
    DeciderConfig(
        temperature=1.4,
        min_confidence=0.8,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        extra={"calibration": {"target_accuracy": 0.95, "validation_n": 500, "ece_before": 0.08, "ece_after": 0.02}},
    ).write(str(model))
    (tmp_path / "eval.json").write_text(json.dumps(report(0.91)))
    (tmp_path / "base.json").write_text(json.dumps(report(0.72)))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"data": {"train": {"banking77": 10, "massive_tr": 10}}}))

    release.main([
        "--model", str(model), "--repo", "acme/locket-decider-2-7b", "--eval", str(tmp_path / "eval.json"),
        "--base-eval", str(tmp_path / "base.json"), "--train-config", str(cfg), "--dry-run",
    ])
    card = (model / "README.md").read_text()
    assert card.startswith("---\nlicense: apache-2.0\nbase_model: Qwen/Qwen2.5-7B-Instruct")
    assert "- PolyAI/banking77" in card and "- AmazonScience/massive" in card
    assert "| banking77 | 100 | 0.910 | 0.900 | 0.020 | 0.720 |" in card
    assert "90.0% of decisions are automated at 96.0% accuracy" in card
    assert "from 0.080 to 0.020" in card
    assert 'Decider.load("acme/locket-decider-2-7b")' in card


def test_card_without_eval_says_so(tmp_path):
    card = release.model_card("a/b", DeciderConfig(), None, None, [])
    assert "Not evaluated yet." in card
