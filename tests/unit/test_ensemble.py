"""Tests for low-level classifier loading."""
import pickle

import joblib
import pytest

from llm_firewall.classifiers.ensemble import ClassifierEnsemble
from llm_firewall.classifiers.huggingface import (
    TinyToxicDetectorClassifier,
    TinyTransformerConfig,
    _align_config_with_state_dict,
)
from llm_firewall.classifiers.pickle_classifier import PickleClassifier
from llm_firewall.classifiers.registry import ClassifierSpec
from tests.fake_models import KeywordPipeline


def test_pickle_classifier_supports_pipeline_bundles(tmp_path):
    model_path = tmp_path / "pipeline_guard.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"pipeline": KeywordPipeline(["block me"])}, handle)

    classifier = PickleClassifier(
        ClassifierSpec(name="pipeline_guard", path=model_path)
    )

    clean_result = classifier.evaluate("hello world")
    blocked_result = classifier.evaluate("please block me now")

    assert clean_result.passed is True
    assert blocked_result.passed is False
    assert blocked_result.filter_name == "pipeline_guard"


def test_pickle_classifier_applies_hard_coded_preprocessing(tmp_path):
    model_path = tmp_path / "preprocessed_guard.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"pipeline": KeywordPipeline(["block me"])}, handle)

    classifier = PickleClassifier(
        ClassifierSpec(
            name="preprocessed_guard",
            path=model_path,
            preprocess=lambda text: text.lower().replace("-", " "),
        )
    )

    result = classifier.evaluate("Please BLOCK-ME now")

    assert result.passed is False
    assert result.filter_name == "preprocessed_guard"


def test_pickle_classifier_supports_joblib_serialized_bundles(tmp_path):
    model_path = tmp_path / "joblib_guard.pkl"
    joblib.dump({"pipeline": KeywordPipeline(["block me"])}, model_path)

    classifier = PickleClassifier(
        ClassifierSpec(name="joblib_guard", path=model_path)
    )

    result = classifier.evaluate("please block me now")

    assert result.passed is False
    assert result.filter_name == "joblib_guard"


def test_tiny_toxic_detector_classifier_blocks_toxic_text(monkeypatch):
    class FakeTensor:
        def __init__(self, value):
            self._value = value

        def squeeze(self):
            return self

        def item(self):
            return self._value

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return FakeInputs(
                {
                    "input_ids": [1, 2, 3],
                    "attention_mask": [1, 1, 1],
                    "token_type_ids": [0, 0, 0],
                }
            )

    class FakeModel:
        def __call__(self, **_kwargs):
            return {"logits": FakeTensor(0.91)}

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTorch:
        @staticmethod
        def no_grad():
            return FakeNoGrad()

    def fake_load_runtime(self):
        return FakeTorch(), "cpu", FakeTokenizer(), FakeModel()

    monkeypatch.setattr(
        TinyToxicDetectorClassifier,
        "_load_runtime",
        fake_load_runtime,
    )

    classifier = TinyToxicDetectorClassifier(
        ClassifierSpec(
            name="tiny_toxic_detector",
            display_name="Tiny-Toxic-Detector",
            backend="huggingface_tiny_toxic_detector",
            model_id="AssistantsLab/Tiny-Toxic-Detector",
        )
    )

    result = classifier.evaluate("You are an idiot.")

    assert result.passed is False
    assert result.filter_name == "Tiny-Toxic-Detector"
    assert result.confidence == 0.91


@pytest.mark.asyncio
async def test_classifier_ensemble_validate_async_runs_pickle_backend(tmp_path):
    """Smoke test the async path against the pickle backend.

    Covers the parallel-execution branch (`run_in_executor`) that's not
    exercised by the synchronous path. We use a pickle classifier rather
    than the HF runtime to avoid network/torch overhead in the unit test —
    the executor doesn't care which classifier is on the other side.
    """
    model_path = tmp_path / "async_guard.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"pipeline": KeywordPipeline(["block me"])}, handle)

    spec = ClassifierSpec(name="async_guard", path=model_path)
    ensemble = ClassifierEnsemble([spec])
    try:
        passed = await ensemble.validate_async("hello world")
        blocked = await ensemble.validate_async("please block me now")
    finally:
        ensemble.close()

    assert passed.passed is True
    assert blocked.passed is False
    assert [r.filter_name for r in blocked.results] == ["async_guard"]


def test_classifier_ensemble_close_is_idempotent(tmp_path):
    """`close()` may be called multiple times; second call is a no-op."""
    model_path = tmp_path / "close_idempotent.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"pipeline": KeywordPipeline(["x"])}, handle)
    ensemble = ClassifierEnsemble([ClassifierSpec(name="x", path=model_path)])
    ensemble.close()
    ensemble.close()  # must not raise


def test_classifier_ensemble_works_as_context_manager(tmp_path):
    """`with ClassifierEnsemble(...)` shuts down the executor on exit."""
    model_path = tmp_path / "ctx_guard.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"pipeline": KeywordPipeline(["block me"])}, handle)
    spec = ClassifierSpec(name="ctx_guard", path=model_path)

    with ClassifierEnsemble([spec]) as ensemble:
        result = ensemble.validate("benign text")
        assert result.passed is True
    # After the with block the executor is closed; the bool is private but
    # we surface enough through the context manager that future regressions
    # become obvious.
    assert ensemble._closed is True


def test_tiny_toxic_detector_uses_checkpoint_position_shape():
    class FakeTensor:
        shape = (1, 512, 64)

    config = TinyTransformerConfig(
        vocab_size=30522,
        embed_dim=64,
        num_heads=2,
        ff_dim=128,
        num_layers=4,
        max_position_embeddings=4096,
    )

    aligned = _align_config_with_state_dict(
        config,
        {"transformer.pos_encoding": FakeTensor()},
    )

    assert aligned.max_position_embeddings == 512
    assert aligned.embed_dim == 64
