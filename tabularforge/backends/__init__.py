"""Backend registry. Explicit imports register every backend."""
from .base import Backend, get_backend, list_backends, register_backend
from . import mock
from . import cpu
from . import openai
from . import vllm
from . import transformers

__all__ = ["Backend", "get_backend", "list_backends", "register_backend"]
