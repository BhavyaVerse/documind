# Verifies the Phase 3 monitoring layer.
#
# What it does:
#   1. Runs 2 queries through the full pipeline (with tracing enabled)
#   2. Prints the Langfuse trace ID and direct dashboard URL for each
#   3. Waits 3 seconds for events to be processed by Langfuse cloud(it is working langfuse is taking more time)
#   4. Fetches and validates each trace via the Langfuse API(as langfuse is taking more time (more than 3 sec that we have set) so no fetching occured and no display of data on terminal)
#   5. Prints a latency breakdown per step across all test queries


import time
import os
from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_store import load_vector_store
from src.retrieval.bm25_index   import load_bm_25_idx
from src.retrieval.hybrid       import hybrid_search
from src.retrieval.reranker     import load_reranker, rerank
from src.generation.generator   import generate_answer
from src.generation.validator   import validate_answer
from src.monitoring.tracer      import DocuMindTrace, get_langfuse

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── Test queries — change these to match your documents ───────────────────────
TEST_QUERIES = [
    "What specific risks does Amazon mention regarding its fulfillment centers and supply chain?",
    "What were the total net sales for the iPhone segment, and did they increase or decrease?",
]


# ── Single traced query ────────────────────────────────────────────────────────

def run_traced_query(
    question: str,
    vector_store,
    bm25_index,
    bm25_chunks,
    reranker,
) -> dict:
    """
    Run the full Phase 2 pipeline for one query with full Phase 3 tracing.
    Returns timing data and the Langfuse trace ID.
    """
    timings = {}
    tracer  = DocuMindTrace()
    tracer.start(question, session_id="phase3-test")

    t_total = time.perf_counter()

    # 1. Hybrid search
    t = time.perf_counter()
    candidates = hybrid_search(vector_store, bm25_index, bm25_chunks, question, k=20)
    timings["hybrid_ms"] = (time.perf_counter() - t) * 1000

    tracer.record_hybrid_search(
        elapsed_ms   = timings["hybrid_ms"],
        query        = question,
        vector_count = len(candidates),
        bm25_count   = len(candidates),
        fused_count  = len(candidates),
    )

    # 2. Rerank
    t = time.perf_counter()
    top_chunks = rerank(reranker, question, candidates, top_n=5)
    timings["rerank_ms"] = (time.perf_counter() - t) * 1000

    tracer.record_rerank(
        elapsed_ms   = timings["rerank_ms"],
        input_count  = len(candidates),
        output_count = len(top_chunks),
        top_score    = top_chunks[0][1]  if top_chunks else 0.0,
        bottom_score = top_chunks[-1][1] if top_chunks else 0.0,
    )

    # 3. Generate
    t = time.perf_counter()
    generation = generate_answer(question, top_chunks)
    timings["gen_ms"] = (time.perf_counter() - t) * 1000

    tu = generation["token_usage"]
    tracer.record_generation(
        elapsed_ms        = timings["gen_ms"],
        question          = question,
        answer            = generation["answer"],
        model             = tu["model"],
        prompt_tokens     = tu["prompt_tokens"],
        completion_tokens = tu["completion_tokens"],
        citations_used    = generation["citations_used"],
        context_str       = generation["context_str"],
    )

    # 4. Validate
    t = time.perf_counter()
    validation = validate_answer(
        answer      = generation["answer"],
        sources     = generation["sources"],
        context_str = generation["context_str"],
    )
    timings["val_ms"] = (time.perf_counter() - t) * 1000

    tracer.record_validation(timings["val_ms"], validation)
    tracer.record_scores(validation, generation)

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000
    tracer.finish(generation["answer"], timings["total_ms"])

    return {
        "question":      question,
        "answer":        generation["answer"],
        "citations":     generation["citations_used"],
        "token_usage":   generation["token_usage"],
        "validation":    validation,
        "timings":       timings,
        "trace_id":      tracer.trace_id,
    }


# ── Langfuse verification ──────────────────────────────────────────────────────

def verify_trace_in_langfuse(trace_id: str) -> bool:
    """
    Fetch a trace from the Langfuse API and confirm it was received.
    Returns True if the trace exists and has the expected name.
    """
    try:
        lf = get_langfuse()
        trace = lf.fetch_trace(trace_id)
        return trace is not None and trace.name == "documind-query"
    except Exception as e:
        print(f" Could not verify trace via API: {e}")
        return False


# ── Latency summary ────────────────────────────────────────────────────────────

def print_latency_summary(all_results: list[dict]) -> None:
    """Print a table of step latencies across all test queries."""
    steps = ["hybrid_ms", "rerank_ms", "gen_ms", "val_ms", "total_ms"]
    labels = {
        "hybrid_ms": "Hybrid search",
        "rerank_ms": "Reranking    ",
        "gen_ms":    "Generation   ",
        "val_ms":    "Validation   ",
        "total_ms":  "TOTAL        ",
    }

    print(f"\n{'=' * 60}")
    print("  Step Latency Summary (ms)")
    print(f"{'=' * 60}")
    print(f"  {'Step':<18}  {'Query 1':>8}  {'Query 2':>8}  {'Query 3':>8}  {'Avg':>8}")
    print(f"  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for step in steps:
        values = [r["timings"].get(step, 0) for r in all_results]
        avg    = sum(values) / len(values)
        row    = f"  {labels[step]:<18}"
        for v in values:
            row += f"  {v:>7.0f}ms"
        row += f"  {avg:>7.0f}ms"
        print(row)

    print(f"{'=' * 60}")

    # Token and cost summary
    print("\n  Token & Cost Summary")
    print(f"  {'─'*40}")
    for i, r in enumerate(all_results, 1):
        tu   = r["token_usage"]
        cost = (tu["prompt_tokens"] * 0.15 + tu["completion_tokens"] * 0.60) / 1_000_000
        print(
            f"  Query {i}: {tu['prompt_tokens']}in / {tu['completion_tokens']}out  "
            f"= ${cost:.6f}"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DocuMind — Phase 3 Monitoring Test")
    print("=" * 60)

    # Validate env vars early
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        print("\n⚠  WARNING: LANGFUSE_PUBLIC_KEY not set in .env")
        print("   Traces will not be sent. Add your keys and retry.\n")

    # Load all resources once
    print("\n[Loading resources ...]")
    vector_store            = load_vector_store()
    bm25_index, bm25_chunks = load_bm_25_idx()
    reranker                = load_reranker()

    all_results = []

    # Run each test query
    for i, question in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 60}")
        print(f"  Query {i}/{len(TEST_QUERIES)}: {question}")
        print(f"{'─' * 60}")

        result = run_traced_query(
            question, vector_store, bm25_index, bm25_chunks, reranker
        )
        all_results.append(result)

        # Print answer preview
        answer_preview = result["answer"][:200].replace("\n", " ")
        print(f"\n  Answer  : {answer_preview}...")
        print(f"  Citations used : {result['citations']}")
        print(f"  Validation     : {'✓ passed' if result['validation']['passed'] else '✗ failed'}")
        print(f"  Total latency  : {result['timings']['total_ms']:.0f} ms")
        print(f"\n  Langfuse trace : {result['trace_id']}")
        if result['trace_id']:
            print(f"  Dashboard URL  : {LANGFUSE_HOST}/trace/{result['trace_id']}")

    # Wait for Langfuse to process events before verifying
    print(f"\n\n[Waiting 3s for Langfuse to process events ...]")
    time.sleep(3)

    # Verify traces landed in Langfuse
    print("\n[Verifying traces in Langfuse ...]")
    verified = 0
    for i, r in enumerate(all_results, 1):
        tid = r["trace_id"]
        if not tid:
            print(f"  Query {i}:  no trace ID (Langfuse keys missing?)")
            continue
        ok = verify_trace_in_langfuse(tid)
        status = " found" if ok else " not found yet (may still be processing)"
        print(f"  Query {i}: {status}  - {tid}")
        if ok:
            verified += 1

    # Latency breakdown table
    print_latency_summary(all_results)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  Phase 3 test complete.")
    print(f"  {verified}/{len(all_results)} traces verified in Langfuse.")
    print(f"\n  View your dashboard:")
    print(f"  {LANGFUSE_HOST}/project")
    print(f"\n  What to check in the dashboard:")
    print(f"    Traces   =>  each query appears as one row")
    print(f"    Spans    =>  click a trace to see the step timeline")
    print(f"    Scores   =>  citation_coverage, faithfulness, validation_passed")
    print(f"    Costs    => token usage and USD cost per generation")
    print(f"{'=' * 60}")