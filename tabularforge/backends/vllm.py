"""vLLM backend (optional).

Serves a tabular foundation model behind an OpenAI-compatible endpoint running
on vLLM. When the endpoint is unreachable it falls back to the CPU backend so
the framework remains runnable without a GPU.
"""
import os

from .base import Backend, register_backend
from .openai import OpenAIBackend


@register_backend("vllm")
class VllmBackend(OpenAIBackend):
    name = "vllm"

    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.api_key:
            self.base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
            self.model = os.environ.get("VLLM_MODEL", "tabular-foundation-model")
