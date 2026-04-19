"""
Model-agnostic backbone wrapper with Engram module injection.

Supports:
  - google/gemma-4-E4B-it (4-bit quantized via bitsandbytes, ~5 GB VRAM)
  - gpt2 / gpt2-medium / gpt2-large (full precision, for development)
  - Any AutoModelForCausalLM-compatible decoder-only model

Architecture auto-detection:
  - Transformer layers: model.model.layers (Gemma/LLaMA) or model.transformer.h (GPT-2)
  - d_model: config.text_config.hidden_size → config.hidden_size → config.n_embd
  - Token embedding: model.model.embed_tokens (Gemma) or model.transformer.wte (GPT-2)

Canonical ID mapping (paper §2.2 "Tokenizer Compression"):
  NFKC + lowercase normalisation collapses the effective vocabulary by ~20%.
"""

from __future__ import annotations

import unicodedata
from typing import Optional

import torch
import torch.nn as nn

from engram_module import EngramModule


# ------------------------------------------------------------------
# Architecture introspection helpers
# ------------------------------------------------------------------

def _get_d_model(config) -> int:
    """Extract hidden dimension from any HuggingFace model config."""
    for cfg in (getattr(config, "text_config", None), config):
        if cfg is None:
            continue
        for attr in ("hidden_size", "n_embd"):
            v = getattr(cfg, attr, None)
            if v is not None:
                return int(v)
    raise ValueError(f"Cannot determine d_model from config: {config}")


def _get_transformer_layers(model: nn.Module):
    """Return the nn.ModuleList of transformer blocks."""
    # Gemma 4 multimodal: model.model.language_model.layers
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")):
        return model.model.language_model.layers
    # Standard decoder: model.model.layers (Gemma 3, LLaMA, Mistral, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(
        "Cannot locate transformer layers. "
        "Expected model.model.language_model.layers, model.model.layers, or model.transformer.h."
    )


def _get_embed_tokens(model: nn.Module) -> nn.Module:
    """Return the token embedding layer (used to track input_ids in generate)."""
    # Gemma 4 multimodal
    if (hasattr(model, "model") and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "embed_tokens")):
        return model.model.language_model.embed_tokens
    # Standard decoder
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    # GPT-2
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    raise ValueError("Cannot locate token embedding layer.")


# ------------------------------------------------------------------
# Canonical map
# ------------------------------------------------------------------

def build_canonical_map(tokenizer) -> torch.Tensor:
    """Build a [vocab_size] int64 tensor mapping raw token ID → canonical ID.

    Tokens whose decoded text normalises to the same NFKC-lowercased string
    are collapsed to the same canonical ID, reducing the effective vocabulary
    by roughly 20% (matches the paper's "23% reduction" figure).
    """
    vocab = tokenizer.get_vocab()
    mapping = torch.arange(len(vocab), dtype=torch.long)
    canonical: dict[str, int] = {}

    for token, raw_id in sorted(vocab.items(), key=lambda x: x[1]):
        try:
            decoded = tokenizer.convert_tokens_to_string([token])
        except Exception:
            decoded = token
        norm = unicodedata.normalize("NFKC", decoded).lower().strip()
        if norm not in canonical:
            canonical[norm] = raw_id
        mapping[raw_id] = canonical[norm]

    return mapping


# ------------------------------------------------------------------
# EngramBackbone
# ------------------------------------------------------------------

class EngramBackbone(nn.Module):
    """Any HuggingFace decoder backbone with an Engram module injected at one layer.

    The backbone is kept frozen when loaded in 4-bit (quantized) mode.
    Only Engram parameters are trained.
    """

    def __init__(
        self,
        backbone: nn.Module,
        engram_layer: int = 1,
        max_ngram: int = 3,
        num_hash_heads: int = 4,
        num_buckets: int = 16381,
        conv_kernel_size: int = 4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.engram_layer = engram_layer

        d_model = _get_d_model(backbone.config)
        self.engram = EngramModule(
            d_model=d_model,
            max_ngram=max_ngram,
            num_hash_heads=num_hash_heads,
            num_buckets=num_buckets,
            conv_kernel_size=conv_kernel_size,
        )

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

        # Move Engram to the same device and dtype as the backbone
        try:
            ref = next(backbone.parameters())
            if ref.device.type != "cpu":
                # For quantized models the param dtype is uint8/float16; use float16 for Engram
                engram_dtype = ref.dtype if ref.dtype in (torch.float16, torch.bfloat16) else torch.float32
                self.engram = self.engram.to(device=ref.device, dtype=engram_dtype)
        except StopIteration:
            pass

        # Runtime state
        self._current_canonical_ids: Optional[torch.Tensor] = None
        self._current_hygiene_mask: Optional[torch.Tensor] = None
        self.engram_enabled: bool = True   # set False to bypass hook for baseline eval

        # Hook the injection layer
        layers = _get_transformer_layers(backbone)
        self._hook_handle = layers[engram_layer].register_forward_hook(self._engram_hook)

        # Canonical map buffer
        self.register_buffer("canonical_map", None, persistent=False)

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _engram_hook(self, module: nn.Module, inputs: tuple, outputs):
        """Intercept block output and blend in Engram's retrieval signal."""
        if self._current_canonical_ids is None or not self.engram_enabled:
            return outputs

        if isinstance(outputs, torch.Tensor):
            hidden = outputs
            is_tuple = False
        else:
            hidden = outputs[0]
            is_tuple = True

        T_hidden = hidden.shape[1]
        T_ids = self._current_canonical_ids.shape[1]

        if T_ids >= T_hidden:
            canonical = self._current_canonical_ids[:, -T_hidden:]
        else:
            pad = torch.zeros(
                hidden.shape[0], T_hidden - T_ids,
                dtype=torch.long, device=hidden.device,
            )
            canonical = torch.cat([pad, self._current_canonical_ids], dim=1)

        # Cast canonical to same device as hidden (important for 4-bit models)
        canonical = canonical.to(hidden.device)

        engram_out = self.engram(
            hidden,
            canonical,
            hygiene_mask=self._current_hygiene_mask,
        )
        # Match the backbone's dtype (fp16 for quantized models, fp32 for GPT-2)
        engram_out = engram_out.to(hidden.dtype)

        return (engram_out,) + outputs[1:] if is_tuple else engram_out

    # ------------------------------------------------------------------
    # Canonical map
    # ------------------------------------------------------------------

    def set_canonical_map(self, canonical_map: torch.Tensor) -> None:
        self.canonical_map = canonical_map

    def _to_canonical(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.canonical_map is None:
            return input_ids
        return self.canonical_map.to(input_ids.device)[input_ids]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        hygiene_mask: Optional[torch.Tensor] = None,
    ):
        self._current_canonical_ids = self._to_canonical(input_ids)
        self._current_hygiene_mask = hygiene_mask
        try:
            return self.backbone(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
        finally:
            self._current_canonical_ids = None
            self._current_hygiene_mask = None

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Delegate to backbone.generate; refresh canonical IDs at each decode step."""
        self._current_canonical_ids = self._to_canonical(input_ids)

        def _token_hook(module, inputs, outputs):
            if inputs and isinstance(inputs[0], torch.Tensor) and inputs[0].dtype == torch.long:
                self._current_canonical_ids = self._to_canonical(inputs[0])

        embed = _get_embed_tokens(self.backbone)
        handle = embed.register_forward_hook(_token_hook)
        try:
            return self.backbone.generate(input_ids=input_ids, **kwargs)
        finally:
            handle.remove()
            self._current_canonical_ids = None

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def engram_param_count(self) -> int:
        return sum(p.numel() for p in self.engram.parameters())

    def backbone_param_count(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_gpt2(
        cls,
        model_name: str = "gpt2-medium",
        engram_layer: int = 1,
        **engram_kwargs,
    ) -> "EngramBackbone":
        from transformers import GPT2LMHeadModel
        backbone = GPT2LMHeadModel.from_pretrained(model_name)
        return cls(backbone, engram_layer=engram_layer, freeze_backbone=False, **engram_kwargs)

    @classmethod
    def from_pretrained_quantized(
        cls,
        model_name: str = "google/gemma-4-E4B-it",
        engram_layer: int = 1,
        load_in_4bit: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        **engram_kwargs,
    ) -> "EngramBackbone":
        """Load a quantized backbone (4-bit via bitsandbytes) with frozen weights.

        Tries to place the full model on GPU first.  If it doesn't fit (OOM or
        bitsandbytes dispatching error), retries with 8-bit quantization which
        has lower peak-load memory.
        """
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        def _load(load_4bit: bool):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=not load_4bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            bits = "4-bit" if load_4bit else "8-bit"
            print(f"Loading {model_name} in {bits} ...")
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map={"": 0},   # force everything to GPU 0 — fails fast if OOM
                dtype=compute_dtype,
                trust_remote_code=True,
            )

        try:
            backbone = _load(load_4bit=load_in_4bit)
        except (RuntimeError, ValueError) as e:
            if load_in_4bit and ("memory" in str(e).lower() or "dispatch" in str(e).lower()):
                print(f"  4-bit OOM, retrying in 8-bit ...")
                backbone = _load(load_4bit=False)
            else:
                raise

        print("  Backbone loaded.")
        return cls(backbone, engram_layer=engram_layer, freeze_backbone=True, **engram_kwargs)


# ------------------------------------------------------------------
# Backwards-compat alias (used by existing tests and run_engram_training.py)
# ------------------------------------------------------------------

class GPT2WithEngram(EngramBackbone):
    """Thin alias kept so existing tests don't break."""

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "gpt2-medium",
        engram_layer: int = 1,
        **engram_kwargs,
    ) -> "GPT2WithEngram":
        return cls.from_pretrained_gpt2(model_name, engram_layer=engram_layer, **engram_kwargs)
