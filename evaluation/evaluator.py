import json
import random
import sys
import time
import argparse
import math
from datetime import datetime
from pathlib import Path
 
from dotenv import load_dotenv
load_dotenv()
 
#  RAGAS imports 
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
 
# Pipeline imports
from src.retrieval.vector_store import load_vector_store
from src.retrieval.bm25_index   import load_bm_25_idx
from src.retrieval.hybrid       import hybrid_search
from src.retrieval.reranker     import load_reranker, rerank
from src.generation.generator   import generate_answer
 
# Paths
GOLDEN_DATASET_PATH = "evaluation/golden_dataset.json"
RESULTS_DIR         = Path("evaluation/results")
RESULTS_DIR.mkdir(exist_ok=True)
 
 
#  Dataset loading - loading the golden dataset from json
 
def load_golden_dataset(path: str = GOLDEN_DATASET_PATH) -> list[dict]:

    with open(path, "r") as f:
        data = json.load(f)
 
    samples = data["samples"]
 
    return samples



# Pipeline runner for single sample
 
def run_pipeline_for_sample(
    sample: dict,
    vector_store,
    bm25_index,
    bm25_chunks,
    reranker,
    top_k: int = 5,
) -> dict:
    """ 
    Returns a dict with:
        question     — original question
        answer       — generated answer
        contexts     — list of retrieved chunk texts (strings, not Documents)
        ground_truth — the reference answer from the golden dataset
        metadata     — timing, company, category
    """
    question     = sample["question"]
    ground_truth = sample["ground_truth"]
 
    t_start = time.perf_counter()
 
    # Hybrid retrieval
    candidates = hybrid_search(vector_store, bm25_index, bm25_chunks, question, k=20)
 
    # Cross-encoder reranking
    top_chunks = rerank(reranker, question, candidates, top_n=top_k)
 
    # Answer generation
    generation = generate_answer(question, top_chunks)
 
    elapsed_ms = (time.perf_counter() - t_start) * 1000
 
    # RAGAS needs contexts as a flat list of strings, not Document objects
    contexts = [doc.page_content for doc, _ in top_chunks]
 
    return {
        "question":     question,
        "answer":       generation["answer"],
        "contexts":     contexts,
        "ground_truth": ground_truth,
        "metadata": {
            "id":           sample["id"],
            "company":      sample["company"],
            "category":     sample["category"],
            "difficulty":   sample.get("difficulty", "medium"),
            "citations":    generation["citations_used"],
            "latency_ms":   round(elapsed_ms, 1),
            "token_usage":  generation["token_usage"],
        },
    }


#  RAGAS evaluation 
 
 
def run_ragas_evaluation(eval_samples: list[dict]) -> tuple[dict, object]:
    """
    Run RAGAS evaluation on a list of collected pipeline outputs.
 
    RAGAS internally uses an LLM (OpenAI by default) to score each metric, so this step also incurs API costs.
 
    Args:
        eval_samples: List of dicts with question, answer, contexts, ground_truth
 
    Returns:
        scores_dict: Dict mapping metric name => average score (0.0-1.0)
        ragas_df:    Pandas DataFrame with per-sample scores
    """
    print(f"\n   Running RAGAS evaluation on {len(eval_samples)} samples ...")
    print("   This uses OpenAI key and takes 1-3 minutes.\n")
 
    dataset = Dataset.from_dict({
        "question":     [s["question"]     for s in eval_samples],
        "answer":       [s["answer"]       for s in eval_samples],
        "contexts":     [s["contexts"]     for s in eval_samples],
        "ground_truth": [s["ground_truth"] for s in eval_samples],
    })
 
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        raise_exceptions=False,  # continue even if an individual sample fails
    )
 
    scores = {
        "faithfulness":      round(float(result["faithfulness"]),      4),
        "answer_relevancy":  round(float(result["answer_relevancy"]),  4),
        "context_precision": round(float(result["context_precision"]), 4),
    }
 
    ragas_df = result.to_pandas()
 
    return scores, ragas_df



# Per-company breakdown 
 
def compute_company_breakdown(
    eval_samples: list[dict],
    ragas_df,
) -> dict[str, dict]:
    """
    Compute average RAGAS scores grouped by company.
    Useful for identifying which company's documents the pipeline
    retrieves most / least accurately.
    """
 
#  creating the copy ae we dont want to change in the original one.
    ragas_df = ragas_df.copy()
    ragas_df["company"] = [s["metadata"]["company"] for s in eval_samples]
 
    breakdown = {}
    for company, group in ragas_df.groupby("company"):
        breakdown[company] = {
            "faithfulness":      round(group["faithfulness"].mean(),      4),
            "answer_relevancy":  round(group["answer_relevancy"].mean(),  4),
            "context_precision": round(group["context_precision"].mean(), 4),
            "sample_count":      len(group),
        }
 
    return breakdown
 


# Result saving 
# Saving full evaluation results to a timestamped JSON file in evaluation/results/.
def save_results(
    scores: dict,
    eval_samples: list[dict],
    ragas_df,
    company_breakdown: dict,
    sample_size: int,
) -> Path:

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"
 
    estimated_cost = round(sample_size * 0.008, 3)
 
    per_sample_data = []
    for i, s in enumerate(eval_samples):
        per_sample_data.append({
            "id":                s["metadata"]["id"],
            "company":           s["metadata"]["company"],
            "category":          s["metadata"]["category"],
            "difficulty":        s["metadata"]["difficulty"],
            "question":          s["question"],
            "answer_preview":    s["answer"][:300],
            "citations":         s["metadata"]["citations"],
            "latency_ms":        s["metadata"]["latency_ms"],
            "faithfulness":      round(float(ragas_df["faithfulness"].iloc[i]),      4) if i < len(ragas_df) else None,
            "answer_relevancy":  round(float(ragas_df["answer_relevancy"].iloc[i]),  4) if i < len(ragas_df) else None,
            "context_precision": round(float(ragas_df["context_precision"].iloc[i]), 4) if i < len(ragas_df) else None,
        })
 
    output = {
        "run_metadata": {
            "timestamp":          timestamp,
            "sample_size":        sample_size,
            "total_available":    len(eval_samples),
            "estimated_cost_usd": estimated_cost,
        },
        "scores":            scores,
        "company_breakdown": company_breakdown,
        "per_sample":        per_sample_data,
    }
 
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
 
    print(f"\n   Results saved to: {output_path}")
    return output_path



# Printing report 
 
def print_report(scores: dict, company_breakdown: dict, thresholds: dict | None = None):

    print(f"\n{'=' * 60}")
    print("  RAGAS Evaluation Report")
    print(f"{'=' * 60}")
 
    print(f"\n  Aggregate Scores")
    print(f"  {'─' * 48}")

    for metric, score in scores.items(): #metric is the string key
        threshold = (thresholds or {}).get(metric)

        if math.isnan(score):
            print(f"  {metric:<22}  N/A     [API Error] FAIL")
            continue

        bar    = "||" * int(score * 20)
        status = ""
        if threshold is not None:
            status = "PASS" if score >= threshold else f"FAIL (threshold: {threshold})"
        print(f"  {metric:<22}  {score:.4f}  [{bar:<20}]{status}")
 
    print(f"\n  Per-Company Breakdown")
    print(f"  {'─' * 48}")
    print(f"  {'Company':<18}  {'Faith':>6}  {'Relev':>6}  {'Prec':>6}  {'n':>4}")
    print(f"  {'─' * 48}")
    for company, s in sorted(company_breakdown.items()):
        print(
            f"  {company:<18}  {s['faithfulness']:>6.3f}  "
            f"{s['answer_relevancy']:>6.3f}  {s['context_precision']:>6.3f}  "
            f"{s['sample_count']:>4}"
        )
 
    print(f"\n{'=' * 60}")



#  Main entry point 

def run_evaluation(
    sample_size: int | None = None,
    seed: int = 42,
    thresholds: dict | None = None,
) -> dict:
    """
    Run the full evaluation pipeline.
 
    Args:
        sample_size: Number of samples to evaluate. None = all ready samples.
        seed:        Random seed for reproducible sampling.
        thresholds:  Optional dict of metric => min_score for pass/fail display.
 
    Returns:
        Dict of metric → average score.
    """
    print("=" * 60)
    print("  DocuMind — RAGAS Evaluation")
    print("=" * 60)
 
    # 1. Load pipeline resources
    print("\n[1/4] Loading pipeline resources ...")
    vector_store            = load_vector_store()
    bm25_index, bm25_chunks = load_bm_25_idx()
    reranker                = load_reranker()
 
    # 2. Load golden dataset
    print("\n[2/4] Loading golden dataset ...")
    ready_samples = load_golden_dataset()
 
    if not ready_samples:
        print("\nNo samples are ready for evaluation.")
        sys.exit(1) # it immediately kills the Python script with an error code to prevent it from crashing further down.
 
    # Sample if requested
    if sample_size and sample_size < len(ready_samples):
        random.seed(seed)
        samples_to_eval = random.sample(ready_samples, sample_size)
        print(f"   Randomly sampled {sample_size} of {len(ready_samples)} ready samples (seed={seed})")
    else:
        samples_to_eval = ready_samples
        print(f"Evaluating all {len(samples_to_eval)} ready samples")
 
    # 3. Run pipeline for each sample
    print(f"\n[3/4] Running pipeline for {len(samples_to_eval)} samples ...")
    eval_samples = []
 
    for i, sample in enumerate(samples_to_eval, 1):
        print(f"   [{i:02d}/{len(samples_to_eval):02d}] {sample['id']:<12} — {sample['question'][:55]}...")
        try:
            result = run_pipeline_for_sample(
                sample, vector_store, bm25_index, bm25_chunks, reranker
            )
            eval_samples.append(result)
        except Exception as e:
            print(f"Pipeline error: {e}")
 
    print(f"\n   Pipeline completed for {len(eval_samples)}/{len(samples_to_eval)} samples")
 
    # 4. RAGAS scoring
    print("\n[4/4] Running RAGAS scoring ...")
    scores, ragas_df = run_ragas_evaluation(eval_samples)
 
    # Breakdown + report
    company_breakdown = compute_company_breakdown(eval_samples, ragas_df)
    print_report(scores, company_breakdown, thresholds)
 
    # Save results
    save_results(scores, eval_samples, ragas_df, company_breakdown, len(samples_to_eval))
 
    return scores


# command line interface
 
if __name__ == "__main__":


# argparse is a module and ArgumentParser is the class, and we are creating the 'parser' object of this class type. 
# And then we are calling the methods of this class as per our requirement, like add_argument and parse_args

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on DocuMind")

    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of samples to evaluate (default: all ready samples)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    
    args = parser.parse_args()
 
    run_evaluation(sample_size=args.n, seed=args.seed)
 
 