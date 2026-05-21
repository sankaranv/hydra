"""
HookedModel factory: auto-detects model family from config and returns the
appropriate hooked wrapper.

Dispatch table
--------------
    model_type="gpt2"     → HookedGPT2
    model_type="gpt_neox" → HookedGPTNeoX
    model_type="llama"    → HookedLlama
    model_type="qwen2"    → HookedLlama (same architecture, shared wrapper)
"""

import torch.nn as nn
from transformers import AutoConfig

from transformer.lib.models.gpt2 import HookedGPT2
from transformer.lib.models.gpt_neox import HookedGPTNeoX
from transformer.lib.models.llama import HookedLlama

# Maps config.model_type → wrapper class
_MODEL_TYPE_TO_CLASS: dict[str, type[nn.Module]] = {
    "gpt2": HookedGPT2,
    "gpt_neox": HookedGPTNeoX,
    "llama": HookedLlama,
    "qwen2": HookedLlama,
}


class HookedModel:
    """Factory for constructing hooked model wrappers from a path or name.

    Use HookedModel.from_pretrained rather than instantiating this class directly.
    """

    @staticmethod
    def from_pretrained(path_or_name: str, **kwargs) -> nn.Module:
        """Auto-detect model family from config and return appropriate hooked wrapper.

        Reads the model config at path_or_name (using local_files_only=True) to
        determine model_type, then constructs and returns the matching wrapper.
        Any extra kwargs are forwarded to the wrapper constructor.

        Supports:
            gpt2     → HookedGPT2
            gpt_neox → HookedGPTNeoX  (Pythia, GPT-NeoX)
            llama    → HookedLlama    (Llama-3.1, Llama-2, etc.)
            qwen2    → HookedLlama    (Qwen2.5, Qwen2, etc.)

        Args:
            path_or_name: Local path or HuggingFace model identifier.
            **kwargs: Forwarded to the wrapper's __init__.

        Returns:
            An nn.Module subclass (HookedGPT2 | HookedGPTNeoX | HookedLlama)
            with pyro.deterministic sites registered for all activations.

        Raises:
            ValueError: If the model_type is not in the supported dispatch table.
        """
        config = AutoConfig.from_pretrained(path_or_name, local_files_only=True)
        model_type = config.model_type

        wrapper_class = _MODEL_TYPE_TO_CLASS.get(model_type)
        if wrapper_class is None:
            supported = sorted(_MODEL_TYPE_TO_CLASS)
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"HookedModel supports: {supported}"
            )

        return wrapper_class(path_or_name, **kwargs)
