"""CLI entry point."""
import argparse

from ..config import load_config
from .doctor import run_doctor


def _add_backend(parser):
    parser.add_argument("--backend", default=None, help="backend name (mock/cpu/openai/vllm/transformers)")


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="path to config.yaml")
    common.add_argument("--backend", default=None, help="backend name")

    parser = argparse.ArgumentParser(prog="tabularforge", parents=[common])
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("doctor", parents=[common], help="smoke-check the framework")

    p_impute = sub.add_parser("impute", parents=[common], help="impute a synthetic table")
    p_impute.add_argument("--n-rows", type=int, default=64)
    p_impute.add_argument("--n-cols", type=int, default=8)
    p_impute.add_argument("--seed", type=int, default=42)

    p_gen = sub.add_parser("generate", parents=[common], help="generate a table from a schema")
    p_gen.add_argument("--n-rows", type=int, default=32)
    p_gen.add_argument("--n-cols", type=int, default=8)
    p_gen.add_argument("--seed", type=int, default=42)

    p_train = sub.add_parser("train", parents=[common], help="run a training demo")
    p_train.add_argument("--epochs", type=int, default=10)

    p_bench = sub.add_parser("bench", parents=[common], help="run the TabArena benchmark")

    p_serve = sub.add_parser("serve", parents=[common], help="start the HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.backend:
        cfg.backend = args.backend

    if args.cmd is None or args.cmd == "doctor":
        run_doctor(cfg)
        return 0

    from ..engine import TabularEngine
    engine = TabularEngine(cfg)

    if args.cmd == "impute":
        from ..scm.synth import synthesize_table
        table, _, _ = synthesize_table(seed=args.seed, n_columns=args.n_cols, n_rows=args.n_rows)
        masked, masked_cells = table.mask(mask_prob=cfg.train.mask_prob, seed=0)
        filled, _ = engine.impute(masked)
        print(f"imputed {len(masked_cells)} cells")
        print(filled.to_records()[:3])
        return 0

    if args.cmd == "generate":
        from ..scm.synth import synthesize_table
        table, _, _ = synthesize_table(seed=args.seed, n_columns=args.n_cols, n_rows=1)
        rows = engine.generate(table.columns, args.n_rows)
        for r in rows[:5]:
            print(r)
        return 0

    if args.cmd == "train":
        from ..bench.imputation import run_imputation
        before = run_imputation(engine.backend, cfg)["accuracy"]
        for _ in range(args.epochs):
            engine.train_step()
        after = run_imputation(engine.backend, cfg)["accuracy"]
        print(f"imputation accuracy: {before:.3f} -> {after:.3f}")
        return 0

    if args.cmd == "bench":
        from ..bench import run_all, summarize
        print(summarize(run_all(engine.backend, cfg)))
        return 0

    if args.cmd == "serve":
        from ..serving import serve
        srv = serve(cfg, args.host, args.port)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
