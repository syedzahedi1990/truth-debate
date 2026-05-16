from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, set_seed, write_resolved_config
from .data import build_datasets
from .evaluate import run_evaluation
from .preflight import download_model, run_preflight
from .report import build_report
from .rescore import rescore_run
from .train_rl import run_rl_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Truth-seeking multi-agent debate experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["run", "make-data", "eval", "train", "preflight"]:
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/quick.yaml")
        p.add_argument("--output", default="runs/debug")
    p_download = sub.add_parser("download-model")
    p_download.add_argument("--config", default="configs/quick.yaml")
    p_download.add_argument("--local-dir", default=None)
    p_report = sub.add_parser("report")
    p_report.add_argument("--output", default="runs/debug")
    p_rescore = sub.add_parser("rescore")
    p_rescore.add_argument("--source", required=True, help="Run directory or zip archive to rescore")
    p_rescore.add_argument("--output", default=None, help="Directory for rescored artifacts")

    args = parser.parse_args()

    if args.cmd == "report":
        report_path = build_report(args.output)
        print(f"Wrote {report_path}")
        return

    if args.cmd == "rescore":
        rescore_run(args.source, args.output)
        return

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    if args.cmd == "preflight":
        run_preflight(cfg)
        return

    if args.cmd == "download-model":
        download_model(cfg, local_dir=args.local_dir)
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_resolved_config(cfg, output)

    if args.cmd == "make-data":
        train_path, eval_path = build_datasets(cfg, output)
        print(f"Wrote {train_path}")
        print(f"Wrote {eval_path}")
        return

    if args.cmd == "eval":
        run_evaluation(cfg, output, label="baseline")
        report_path = build_report(output)
        print(f"Wrote {report_path}")
        return

    if args.cmd == "train":
        adapter = run_rl_training(cfg, output)
        print(f"Wrote adapter to {adapter}")
        return

    if args.cmd == "run":
        build_datasets(cfg, output)
        run_evaluation(cfg, output, label="baseline")
        adapter = None
        if bool(cfg["training"].get("enabled", True)):
            adapter = run_rl_training(cfg, output)
            run_evaluation(cfg, output, label="trained", adapter_path=adapter)
        report_path = build_report(output)
        print(f"Wrote {report_path}")
        return

    raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
