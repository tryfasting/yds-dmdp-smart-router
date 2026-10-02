"""
SmartRouter command-line entry point.

    smartrouter audit-data   --raw-dir DIR [--v3 FILE]
    smartrouter evaluate     --csv test_results_routed.csv [--thresholds 0.2 0.25 0.5]
    smartrouter audit-labels --csv dataset_master.csv
    smartrouter judge-report --csv evaluation_results.csv
    smartrouter predict      --text "..." --intensity STRONG [--field NONE]
    smartrouter reinfer      --csv test_results_routed.csv [--limit 16]

No command calls a paid API. Input sentences are never printed or written to reports.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from smartrouter.core import config


def _provenance(inputs: dict[str, Path]) -> dict[str, Any]:
    from smartrouter.data import sha256_file

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "inputs": {name: {"file": Path(p).name, "sha256": sha256_file(Path(p))} for name, p in inputs.items()},
    }


def _emit(payload: dict[str, Any], out: Optional[Path]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=float)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[saved] {out}", file=sys.stderr)
    print(text)


def cmd_audit_data(args: argparse.Namespace) -> int:
    from smartrouter.data import CLIENT_FILE, EVENT_FILE, USAGE_FILE, check_v3, load_raw, merge_logs

    usage, clients, events = load_raw(args.raw_dir)
    _, report = merge_logs(usage, clients, events)
    payload: dict[str, Any] = {"merge": report.as_dict()}
    inputs = {name: args.raw_dir / f for name, f in (("usage", USAGE_FILE), ("clients", CLIENT_FILE), ("events", EVENT_FILE))}
    if args.v3:
        payload["stored_v3"] = check_v3(args.v3)
    payload["provenance"] = _provenance(inputs)
    _emit(payload, args.out)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from smartrouter.evaluation import evaluate_routing

    df = pd.read_csv(args.csv)
    payload = evaluate_routing(df, thresholds=args.thresholds, reference_threshold=args.reference_threshold)
    payload["provenance"] = _provenance({"predictions": args.csv})
    _emit(payload, args.out)
    return 0


def cmd_audit_labels(args: argparse.Namespace) -> int:
    from smartrouter.evaluation import label_distribution

    df = pd.read_csv(args.csv)
    payload = label_distribution(df)
    payload["provenance"] = _provenance({"dataset": args.csv})
    _emit(payload, args.out)
    return 0


def cmd_judge_report(args: argparse.Namespace) -> int:
    from smartrouter.judge import summarize_judge

    df = pd.read_csv(args.csv)
    payload = summarize_judge(df, n_boot=args.n_boot, seed=args.seed)
    payload["provenance"] = _provenance({"judge_results": args.csv})
    _emit(payload, args.out)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from smartrouter.models.classifier import RoBERTaClassifier
    from smartrouter.router.dynamic import IntensityRuleRouter, RoBERTaDynamicRouter

    model_router = RoBERTaDynamicRouter(
        classifier=RoBERTaClassifier(model_path=args.model_dir), threshold=args.threshold
    )
    payload = {
        "roberta": model_router.predict(args.text, args.intensity, args.field),
        "intensity_rule": IntensityRuleRouter().predict(args.text, args.intensity, args.field),
    }
    _emit(payload, None)
    return 0


def cmd_reinfer(args: argparse.Namespace) -> int:
    """Re-run the local checkpoint on stored inputs and compare with stored probabilities."""
    from smartrouter.models.classifier import RoBERTaClassifier

    df = pd.read_csv(args.csv).head(args.limit)
    classifier = RoBERTaClassifier(model_path=args.model_dir)
    probs = pd.Series(classifier.predict_proba(df["full_input_text"].astype(str).tolist()), index=df.index)
    diff = (probs - df["hard_prob"]).abs()
    new_dec = probs >= args.threshold
    old_dec = df["hard_prob"] >= args.threshold
    payload = {
        "rows": int(len(df)),
        "threshold": args.threshold,
        "decision_agreement": int((new_dec == old_dec).sum()),
        "max_abs_prob_diff": float(diff.max()),
        "mean_abs_prob_diff": float(diff.mean()),
        "note": "Stored full_input_text may be tokenizer-decoded text; probability identity is not expected.",
        "provenance": _provenance({"predictions": args.csv}),
    }
    _emit(payload, args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smartrouter", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit-data", help="Merge raw service logs with cardinality checks")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--v3", type=Path, help="Stored full_merged_v3.csv to hash-check")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_audit_data)

    p = sub.add_parser("evaluate", help="Routing metrics + intensity-rule baseline from stored probabilities")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.20, 0.25, 0.50])
    p.add_argument("--reference-threshold", type=float, default=config.ROUTER_THRESHOLD)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("audit-labels", help="Hard-label rate per intensity/field tag in the training dataset")
    p.add_argument("--csv", type=Path, required=True, help="dataset_master.csv")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_audit_labels)

    p = sub.add_parser("judge-report", help="Summarize stored LLM-as-a-Judge results with bootstrap CIs")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_judge_report)

    p = sub.add_parser("predict", help="Route one request with the local checkpoint and the baseline")
    p.add_argument("--text", required=True)
    p.add_argument("--intensity", choices=config.INTENSITIES, default=config.DEFAULT_INTENSITY)
    p.add_argument("--field", choices=config.FIELDS, default=config.DEFAULT_FIELD)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--threshold", type=float, default=config.ROUTER_THRESHOLD)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("reinfer", help="Compare local checkpoint inference with stored probabilities")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--threshold", type=float, default=config.ROUTER_THRESHOLD)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_reinfer)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
