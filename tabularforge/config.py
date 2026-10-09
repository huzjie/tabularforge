"""Configuration: dataclass + YAML/JSON loading with a zero-dependency fallback."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False


@dataclass
class ModelConfig:
    dim: int = 64
    mlp_hidden: int = 128
    n_layers: int = 2
    max_columns: int = 64
    max_rows: int = 512
    vocab_size: int = 1024


@dataclass
class ScmConfig:
    min_columns: int = 4
    max_columns: int = 12
    edge_prob: float = 0.35
    n_rows: int = 256
    seed: int = 42


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 0.05
    mask_prob: float = 0.15
    batch_size: int = 64


@dataclass
class Config:
    name: str = "tabularforge"
    backend: str = "mock"
    model: ModelConfig = field(default_factory=ModelConfig)
    scm: ScmConfig = field(default_factory=ScmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d):
        d = d or {}

        def _sub(prefix, cls_):
            sub = d.get(prefix, {}) or {}
            return cls_(**{k: v for k, v in sub.items() if k in cls_.__dataclass_fields__})

        return cls(
            name=d.get("name", "tabularforge"),
            backend=d.get("backend", "mock"),
            model=_sub("model", ModelConfig),
            scm=_sub("scm", ScmConfig),
            train=_sub("train", TrainConfig),
        )

    def to_dict(self):
        return asdict(self)


def _find_default():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "configs" / "default.yaml",
        here.parent / "configs" / "default.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_config(path=None):
    """Load config from a path (yaml/json) or fall back to defaults.

    The built-in default config lives in ``tabularforge/configs/default.yaml``.
    """
    if path is None:
        path = _find_default()
    if path is not None:
        p = Path(path)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            if p.suffix in (".yaml", ".yml"):
                if _HAS_YAML:
                    data = yaml.safe_load(text)
                else:
                    from .utils.yamlish import parse
                    data = parse(text)
            else:
                import json
                data = json.loads(text)
            return Config.from_dict(data)
    return Config()
