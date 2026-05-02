"""
Hugging Face-backed classifiers used by the firewall.

Two runtimes live here:

* `TinyToxicDetectorClassifier` — bespoke loader for AssistantsLab's
  `Tiny-Toxic-Detector`, an output-side toxicity filter built on a custom
  `TinyTransformer` architecture.
* `HFSequenceClassifier` — generic wrapper around any
  `AutoModelForSequenceClassification` checkpoint on the Hub. Intended for
  off-the-shelf prompt-injection classifiers (e.g.
  `meta-llama/Llama-Prompt-Guard-2-86M`).

Both implement the same firewall-internal contract: take a `ClassifierSpec`,
expose `evaluate(text) -> FilterResult`, and let `ClassifierEnsemble`
dispatch to them via the `spec.backend` field.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from time import perf_counter

from llm_firewall.classifiers.registry import ClassifierSpec
from llm_firewall.filters import FilterResult

logger = logging.getLogger(__name__)


def _select_torch_device():
    """Pick the best available torch device: mps -> cuda -> cpu.

    Centralized here so both runtimes can opt in. The existing
    `TinyToxicDetectorClassifier` historically pinned to CPU; new code paths
    use this helper. Honor `LLM_FIREWALL_FORCE_CPU=1` for tests / debugging.
    """
    import os

    import torch

    if os.environ.get("LLM_FIREWALL_FORCE_CPU") == "1":
        return torch.device("cpu")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and getattr(mps, "is_available", lambda: False)():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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


class HFSequenceClassifier:
    """Generic `AutoModelForSequenceClassification` runtime.

    Loads any HuggingFace sequence-classification checkpoint and exposes the
    same `evaluate(text) -> FilterResult` contract every other firewall
    classifier uses, so it slots into `ClassifierEnsemble` unchanged. The
    role (input vs output classifier) is decided at registry level by where
    the spec lives, not by this class.

    The spec controls which label is treated as "blocking":

    * `injection_label_id`: integer index into `model.config.id2label`. Use
      this when you know the exact index (e.g., 1 for typical binary
      checkpoints).
    * `injection_label_name`: case-insensitive label string. Looked up in
      `id2label`. Useful for multi-label models like Llama-Prompt-Guard-2 where
      one of several labels means "block."

    If neither is set we fall back to: index 1 for 2-class models, otherwise
    the first label whose lowercased name matches a known blocking keyword
    (`injection`, `jailbreak`, `unsafe`, `harmful`, `malicious`).
    """

    _BLOCKING_KEYWORDS = {
        "injection",
        "jailbreak",
        "unsafe",
        "harmful",
        "malicious",
        "adversarial",
        "attack",
        "block",
        "blocked",
    }

    def __init__(self, spec: ClassifierSpec):
        if not spec.model_id:
            raise ValueError("HFSequenceClassifier requires a model_id")

        self.name = spec.display_name or spec.name
        self.model_id = spec.model_id
        self._preprocess = spec.preprocess
        self._threshold = spec.threshold
        self._max_length = spec.max_length
        self._injection_label_id = getattr(spec, "injection_label_id", None)
        self._injection_label_name = getattr(spec, "injection_label_name", None)
        self._torch, self._device, self._tokenizer, self._model, self._block_idx = (
            self._load_runtime()
        )

    def _load_runtime(self):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ValueError(
                "HFSequenceClassifier requires 'torch' and 'transformers'. "
                "Install them before enabling HuggingFace input classifiers."
            ) from exc

        device = _select_torch_device()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=False
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, trust_remote_code=False
        )
        # Resolve the blocking-label index BEFORE moving weights to the
        # device, so a configuration error doesn't leave a 100s-of-MB model
        # stranded in GPU memory until the next GC cycle.
        block_idx = self._resolve_blocking_index(model)
        model = model.to(device)
        model.eval()

        logger.info(
            "loaded %s on %s (blocking label index=%d, label=%r)",
            self.model_id,
            device,
            block_idx,
            getattr(model.config, "id2label", {}).get(block_idx),
        )
        return torch, device, tokenizer, model, block_idx

    def _resolve_blocking_index(self, model) -> int:
        if self._injection_label_id is not None:
            return int(self._injection_label_id)

        id2label: dict[int, str] = getattr(model.config, "id2label", {}) or {}

        if self._injection_label_name is not None:
            target = str(self._injection_label_name).lower()
            for idx, name in id2label.items():
                if str(name).lower() == target:
                    return int(idx)
            raise ValueError(
                f"injection_label_name={self._injection_label_name!r} not found in "
                f"model.config.id2label={id2label!r}"
            )

        if len(id2label) == 2:
            return 1

        for idx, name in id2label.items():
            lname = str(name).lower()
            if any(kw in lname for kw in self._BLOCKING_KEYWORDS):
                return int(idx)

        raise ValueError(
            f"Cannot infer blocking-label index for {self.model_id}. "
            f"Set injection_label_id or injection_label_name. "
            f"id2label={id2label!r}"
        )

    # Overlap (in tokens) between adjacent windows when a long prompt is
    # chunked. Set so an injection payload spanning a window boundary still
    # appears wholly in at least one window. 64 tokens ≈ a sentence or two
    # — enough for typical "ignore previous instructions" payloads.
    _CHUNK_OVERLAP_TOKENS = 64

    def evaluate(self, text: str) -> FilterResult:
        started_at = perf_counter()
        normalized_text = self._preprocess(text)
        # Tokenize WITHOUT truncation first so we know the real length and
        # can run a sliding window if the prompt would otherwise be cut.
        # The classifier model still has a fixed max_length; we just refuse
        # to silently drop tokens past it.
        full = self._tokenizer(normalized_text, return_tensors="pt", truncation=False)
        full_ids = full["input_ids"]
        full_mask = full.get("attention_mask")
        full_len = int(full_ids.shape[-1])

        # Fast path: short-enough prompts go through a single forward pass,
        # behaviour identical to the pre-chunking implementation.
        if full_len <= self._max_length:
            blocking_score = self._score_window(full_ids, full_mask)
        else:
            # Sliding window: stride = max_length - overlap. Take max
            # P(injection) across all windows so an injection anywhere in
            # the prompt — head, tail, or middle — is caught instead of
            # being hidden behind a benign preamble that fills the model's
            # context window.
            stride = max(1, self._max_length - self._CHUNK_OVERLAP_TOKENS)
            scores: list[float] = []
            offset = 0
            while offset < full_len:
                end = min(offset + self._max_length, full_len)
                window_ids = full_ids[:, offset:end]
                window_mask = full_mask[:, offset:end] if full_mask is not None else None
                scores.append(self._score_window(window_ids, window_mask))
                if end == full_len:
                    break
                offset += stride
            blocking_score = max(scores) if scores else 0.0
            logger.warning(
                "%s: input was %d tokens (> max_length=%d); scored as "
                "%d overlapping windows (overlap=%d), max P(injection)=%.4f",
                self.name,
                full_len,
                self._max_length,
                len(scores),
                self._CHUNK_OVERLAP_TOKENS,
                blocking_score,
            )

        blocked = blocking_score > self._threshold
        latency_ms = round((perf_counter() - started_at) * 1000, 3)

        verdict = "BLOCKED" if blocked else "PASSED"
        return FilterResult(
            passed=not blocked,
            filter_name=self.name,
            confidence=blocking_score,
            latency_ms=latency_ms,
            detail=f"[{verdict}] {self.name} | P(injection)={blocking_score:.1%}",
        )

    def _score_window(self, input_ids, attention_mask) -> float:
        """Run one forward pass on a (batch=1) tensor of token ids and
        return P(injection). Caller guarantees `input_ids.shape[-1]` is
        ≤ max_length."""
        kwargs = {"input_ids": input_ids.to(self._device)}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(self._device)
        # Some BERT-style configs (Llama-Prompt-Guard-2 included) don't
        # need token_type_ids; suppress them if the tokenizer emitted any.
        with self._torch.no_grad():
            logits = self._model(**kwargs).logits
        # Softmax across labels; pull out the configured blocking-label prob.
        # `confidence` in the FilterResult is always P(injection) regardless
        # of pass/fail so downstream metrics (ROC-AUC, PR-AUC) and the detail
        # string both speak the same number.
        probs = self._torch.softmax(logits, dim=-1).squeeze(0).tolist()
        return float(probs[self._block_idx])
