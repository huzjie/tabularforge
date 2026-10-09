"""Doctor: smoke-check the whole framework (backends + benchmark + train)."""
from ..backends import get_backend, list_backends
from ..config import Config


def run_doctor(cfg: Config, quiet=False):
    """Instantiate every backend, run the benchmark and a training demo."""
    lines = []
    lines.append(f"tabularforge {__import__('tabularforge.version', fromlist=['__version__']).__version__}")
    lines.append(f"config: backend={cfg.backend} dim={cfg.model.dim} "
                 f"max_columns={cfg.model.max_columns} n_rows={cfg.scm.n_rows}")
    lines.append(f"backends: {', '.join(list_backends())}")

    from ..bench import run_all, summarize
    for name in list_backends():
        backend = get_backend(name, cfg)
        results = run_all(backend, cfg)
        lines.append(f"-- backend: {name}")
        lines.append(summarize(results))

    # training demo on the mock backend
    from ..bench.imputation import run_imputation
    mock = get_backend("mock", cfg)
    before = run_imputation(mock, cfg)["accuracy"]
    for _ in range(10):
        mock.train_step()
    after = run_imputation(mock, cfg)["accuracy"]
    lines.append(f"train demo (mock): imputation accuracy {before:.3f} -> {after:.3f}")

    if not quiet:
        for ln in lines:
            print(ln, flush=True)
    return lines
