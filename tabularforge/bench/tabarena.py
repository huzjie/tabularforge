"""TabArena: run every benchmark task and aggregate a report."""
from .classification import run_classification
from .generation import run_generation
from .imputation import run_imputation
from .regression import run_regression
from .speed import run_speed


def run_all(backend, cfg):
    """Run all tasks. Returns an ordered list of result dicts."""
    results = [
        run_imputation(backend, cfg),
        run_classification(backend, cfg),
        run_regression(backend, cfg),
        run_generation(backend, cfg),
        run_speed(backend, cfg),
    ]
    return results


def summarize(results):
    """Render a human-readable multi-line summary string."""
    lines = []
    for r in results:
        task = r.get("task")
        if task == "imputation":
            lines.append(f"  imputation      accuracy = {r['accuracy']:.4f}  ({r['n_correct']}/{r['n_cells']})")
        elif task == "classification":
            lines.append(f"  classification  accuracy = {r['accuracy']:.4f}" if r.get("accuracy") is not None
                         else "  classification  (no categorical column)")
        elif task == "regression":
            lines.append(f"  regression      rmse     = {r['rmse']:.4f}" if r.get("rmse") is not None
                         else "  regression      (no numeric column)")
        elif task == "generation":
            lines.append(f"  generation      validity = {r['validity']:.4f}  shape_ok={r['shape_ok']}")
        elif task == "speed":
            lines.append(f"  speed           {r['speedup']}x  (batched vs naive cell prediction)")
    return "\n".join(lines)
