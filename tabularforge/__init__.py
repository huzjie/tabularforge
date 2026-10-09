"""tabularforge: a tabular foundation model framework.

A zero-dependency, fully runnable reference implementation of the engineering
ideas behind NVIDIA's Kumo Tabular (announced 2026-09-29): foundation models
for TABULAR data -- give the model a table with some cells filled in and it
predicts the rest, with no per-dataset training.

Kumo Tabular is pretrained entirely on *artificial* tables sampled from
structural causal models (SCMs), so it never saw customer or benchmark data,
ranks first on the TabArena benchmark, and predicts ~17x faster than the
previous SOTA (Tsinghua's LimiX-2).

This repo distills its most reusable engineering ideas into one framework:

1. Masked Cell Modeling (MCM) -- the "BERT for tables": mask random cells and
   train the model to reconstruct them from row context + column statistics.
2. SCM synthetic data engine -- sample a DAG of causal relations, then generate
   infinite labeled tables where every column value is a function of its
   causal parents + noise. No real data required, no privacy leakage.
3. Column-type-aware encoding -- numeric / categorical / datetime / text
   columns get different encoders but collapse into ONE cell embedding space.
4. Zero-shot imputation -- a pretrained model fills NaN cells without training.
5. Five backends -- deterministic trainable mock / cpu / openai / vllm /
   transformers -- plus an OpenAI-compatible stdlib HTTP server.
6. TabArena-style benchmark -- imputation / classification / regression /
   generation / inference-speed tasks with known ground truth.

Highlights
----------
- Zero-dependency pure-Python core (SCM engine + TabFM model + benchmark)
- Deterministic trainable mock backend (imputation accuracy 0.5 -> 1.0)
- Structural causal model synthesis + optional causal parent discovery
- CLI / Docker / K8s / Helm / CI + prompt library + recipes + docs
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
