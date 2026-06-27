import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
 
from dotenv import load_dotenv
load_dotenv()
 
from evaluation.evaluator import run_evaluation
 
#  Metric thresholds 
#
# Minimum acceptable score for each RAGAS metric.
# If any score falls below its threshold the CI gate fails and the PR is blocked.

THRESHOLDS: dict[str, float] = {
    "faithfulness":      0.80,
    "answer_relevancy":  0.75,
    "context_precision": 0.70,
}
 

CI_SAMPLE_SIZE = 20 # we check only on 20 samples, to avoid much llm api costing
CI_SEED        = 42
 
 

def evaluate_thresholds(
    scores: dict[str, float],
) -> tuple[bool, list[str], list[str]]:
    """
    Compare scores against thresholds.
 
    Returns:
        passed:   True if all metrics meet their threshold
        failures: List of failure strings
        passes:   List of pass strings
    """
    failures, passes = [], []
 
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric)
 
        if score is None:
            failures.append(
                f" {metric:<22} — score not available (evaluation may have errored)"
            )
            continue
 
        if score >= threshold:
            margin = score - threshold
            passes.append(
                f"  passed  {metric:<22}  {score:.4f}  "
                f"(threshold: {threshold:.2f}, margin: +{margin:.4f})"
            )
        else:
            gap = threshold - score
            failures.append(
                f"  failed  {metric:<22}  {score:.4f}  "
                f"(threshold: {threshold:.2f}, gap: -{gap:.4f})"
            )
 
    return len(failures) == 0, failures, passes



# Saving the CI gate result to a JSON file for artifact upload.
def save_gate_result(
    scores: dict,
    passed: bool,
    failures: list[str],
    sample_size: int,
) -> None:
    results_dir = Path("evaluation/results")
    results_dir.mkdir(exist_ok=True)
 
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"ci_gate_{timestamp}.json"
 
    output = {
        "timestamp":   timestamp,
        "passed":      passed,
        "sample_size": sample_size,
        "scores":      scores,
        "thresholds":  THRESHOLDS,
        "failures":    failures,
        "margins": {
            metric: round(scores.get(metric, 0) - threshold, 4)
            for metric, threshold in THRESHOLDS.items()
        },
    }
 
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
 
    print(f"\n  Gate result saved to: {output_path}")



# Print a clearly visible PASS or FAIL banner 
def print_gate_banner(passed: bool) -> None:
    width = 60
 
    if passed:
        print(f"\n{'═' * width}")
        print(f"  {'  CI GATE PASSED'}")
        print(f"  {'All metrics meet their thresholds.'}")
        print(f"{'═' * width}")
    else:
        print(f"\n{'═' * width}")
        print(f"  {'  CI GATE FAILED'}")
        print(f"  {'One or more metrics are below threshold.'}")
        print(f"  {'Review the failing metrics before merging.'}")
        print(f"{'═' * width}")




#  Main 
 
def main():
    parser = argparse.ArgumentParser(
        description="DocuMind CI evaluation gate. Exits 0 on pass, 1 on fail."
    )
    parser.add_argument(
        "--n", type=int, default=CI_SAMPLE_SIZE,
        help=f"Number of samples to evaluate (default: {CI_SAMPLE_SIZE})"
    )
    parser.add_argument(
        "--seed", type=int, default=CI_SEED,
        help=f"Random seed for reproducible sampling (default: {CI_SEED})"
    )
    # This allows you to test if GitHub Actions server is booting up correctly without actually spending LLM API credits to run evaluations.
    parser.add_argument(
        "--dry-run", action="store_true", #this is use for flags
        help="Print configuration and exit without running evaluation"
    )
    args = parser.parse_args()
 
    print("=" * 60)
    print("  DocuMind CI Gate")
    print("=" * 60)
    print(f"\n  Sample size : {args.n}")
    print(f"  Seed        : {args.seed}")
    print(f"\n  Thresholds:")
    for metric, threshold in THRESHOLDS.items():
        print(f"    {metric:<22}  >=  {threshold:.2f}")
 
    if args.dry_run:
        print("\n  [Dry run — exiting without evaluation]")
        sys.exit(0)
 
    if not os.environ.get("GROQ_API_KEY"):
        print("\n  ERROR: API KEY not set.")
        print("     Add it to your .env file or GitHub Actions secrets.")
        sys.exit(1)
 
    # Run evaluation
    print(f"\n{'─' * 60}")
    scores = run_evaluation(
        sample_size=args.n,
        seed=args.seed,
        thresholds=THRESHOLDS,
    )
 
    # Check thresholds
    passed, failures, passes = evaluate_thresholds(scores)
 
    print(f"\n  Threshold Check")
    print(f"  {'─' * 48}")
    for line in passes:
        print(line)
    for line in failures:
        print(line)
 
    # Save gate result JSON
    save_gate_result(scores, passed, failures, args.n)
 
    # Final banner and exit code
    print_gate_banner(passed)
    sys.exit(0 if passed else 1)
 
 
if __name__ == "__main__":
    main()
 
 