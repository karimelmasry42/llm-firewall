"""
Hugging Face-backed toxicity classifier for firewall output checks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from time import perf_counter

from llm_firewall.classifiers.registry import ClassifierSpec
from llm_firewall.filters import FilterResult


@dataclass(frozen=True)
class TinyTransformerConfig:
    """Minimal config for the Tiny-Toxic-Detector architecture."""

    vocab_size: int = 30522
    embed_dim: int = 64
    num_heads: int = 2
    ff_dim: int = 128
    num_layers: int = 4
    max_position_embeddings: int = 512

    @classmethod
    def from_huggingface(cls, model_id: str):
        """Load model config directly from the Hugging Face repo."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ValueError(
                "Tiny-Toxic-Detector requires 'huggingface-hub'. "
                "Install it before enabling output classifiers."
            ) from exc

        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            raw_config = json.load(handle)

        return cls(
            vocab_size=raw_config.get("vocab_size", cls.vocab_size),
            embed_dim=raw_config.get("embed_dim", cls.embed_dim),
            num_heads=raw_config.get("num_heads", cls.num_heads),
            ff_dim=raw_config.get("ff_dim", cls.ff_dim),
            num_layers=raw_config.get("num_layers", cls.num_layers),
            max_position_embeddings=raw_config.get(
                "max_position_embeddings",
                cls.max_position_embeddings,
            ),
        )


def _align_config_with_state_dict(
    config: TinyTransformerConfig,
    state_dict,
) -> TinyTransformerConfig:
    """Prefer the checkpoint tensor shape when metadata and weights disagree."""
    pos_encoding = state_dict.get("transformer.pos_encoding")
    if pos_encoding is None or len(getattr(pos_encoding, "shape", ())) != 3:
        return config

    checkpoint_positions = int(pos_encoding.shape[1])
    checkpoint_embed_dim = int(pos_encoding.shape[2])
    if (
        checkpoint_positions == config.max_position_embeddings
        and checkpoint_embed_dim == config.embed_dim
    ):
        return config

    return replace(
        config,
        max_position_embeddings=checkpoint_positions,
        embed_dim=checkpoint_embed_dim,
    )


def _load_tiny_toxic_detector_runtime(model_id: str):
    """Load the Tiny-Toxic-Detector model, tokenizer, and runtime dependencies."""
    try:
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "Tiny-Toxic-Detector requires 'torch', 'transformers', 'huggingface-hub', "
            "and 'safetensors'. "
            "Install them before enabling output classifiers."
        ) from exc

    class TinyTransformer(nn.Module):
        def __init__(
            self,
            vocab_size,
            embed_dim,
            num_heads,
            ff_dim,
            num_layers,
            max_position_embeddings,
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoding = nn.Parameter(
                torch.zeros(1, max_position_embeddings, embed_dim)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.fc = nn.Linear(embed_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
            x = self.transformer(x)
            x = x.mean(dim=1)
            x = self.fc(x)
            return self.sigmoid(x)

    class TinyTransformerForSequenceClassification(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.transformer = TinyTransformer(
                config.vocab_size,
                config.embed_dim,
                config.num_heads,
                config.ff_dim,
                config.num_layers,
                config.max_position_embeddings,
            )

        def forward(self, input_ids, attention_mask=None):
            return {"logits": self.transformer(input_ids)}

    device = torch.device("cpu")
    weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    state_dict = load_file(weights_path, device="cpu")
    config = _align_config_with_state_dict(
        TinyTransformerConfig.from_huggingface(model_id),
        state_dict,
    )
    model = TinyTransformerForSequenceClassification(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return torch, device, tokenizer, model


class TinyToxicDetectorClassifier:
    """Runs AssistantsLab/Tiny-Toxic-Detector as a firewall output classifier."""

    def __init__(self, spec: ClassifierSpec):
        if not spec.model_id:
            raise ValueError("Tiny-Toxic-Detector classifier requires a model_id")

        self.name = spec.display_name or spec.name
        self.model_id = spec.model_id
        self._preprocess = spec.preprocess
        self._threshold = spec.threshold
        self._max_length = spec.max_length
        self._torch, self._device, self._tokenizer, self._model = self._load_runtime()

    def _load_runtime(self):
        return _load_tiny_toxic_detector_runtime(self.model_id)

    def evaluate(self, text: str) -> FilterResult:
        """Classify text and return a firewall-friendly result."""
        started_at = perf_counter()
        normalized_text = self._preprocess(text)
        inputs = self._tokenizer(
            normalized_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
        ).to(self._device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        blocking_score = float(outputs["logits"].squeeze().item())
        blocking_score = max(0.0, min(1.0, blocking_score))
        blocked = blocking_score > self._threshold
        latency_ms = round((perf_counter() - started_at) * 1000, 3)

        if blocked:
            return FilterResult(
                passed=False,
                filter_name=self.name,
                confidence=blocking_score,
                latency_ms=latency_ms,
                detail=f"[BLOCKED] {self.name} | Confidence: {blocking_score:.1%}",
            )

        return FilterResult(
            passed=True,
            filter_name=self.name,
            confidence=blocking_score,
            latency_ms=latency_ms,
            detail=f"[PASSED] {self.name} | Confidence: {1 - blocking_score:.1%}",
        )
