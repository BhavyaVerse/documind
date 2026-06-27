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

TEST_QUERIES = [
    "What were the total net sales for the iPhone segment, and did they increase or decrease?",
]

def test_hybrid_search(vector_store, bm25_idx, bm25_chunks, query: str) :
    """Verify hybrid retrieval returns relevant candidates."""
    print(f"\n{'─' * 55}")
    print("TEST 1: Hybrid Search")
    print(f"Query: {query}")

    t = time.perf_counter()
    results = hybrid_search(vector_store, bm25_idx, bm25_chunks, query,k=20 )
    elapsed = (time.perf_counter() -t) * 1000

    print(f"\nReturned {len(results)} candidates in {elapsed:.0f} ms")
    print("\nTop 3 candidates:")
    for i, (doc, score) in enumerate(results[:3] , 1) :
        source = os.path.basename(doc.metadata.get("source", "?"))
        page = doc.metadata.get("page" , "?")
        print(f" [{i}] RRF score: {score:.5f} | {source} | Page {page}")
        print(f" {doc.page_content[:120].strip()} ...")

    assert len(results) > 0, "Hybrid search results return no searches."
    print("Hybrid search passed")
    return results

def test_reranker(reranker, query: str, candidates):
    """Verify the cross-encoder reranks and trims to top-5."""
    print(f"\n{'─' * 55}")
    print("TEST 2: Cross-Encoder Reranking")
 
    t = time.perf_counter()
    top_chunks = rerank(reranker, query, candidates, top_n=5)
    elapsed = (time.perf_counter() - t) * 1000
 
    print(f"\nReranked {len(candidates)} => {len(top_chunks)} in {elapsed:.0f} ms")
    print("\nTop 3 after reranking:")
    for i, (doc, score) in enumerate(top_chunks[:3], 1):
        source = os.path.basename(doc.metadata.get("source", "?"))
        page   = doc.metadata.get("page", "?")
        print(f"  [{i}] CE score: {score:.4f} | {source} | Page {page}")
 
    assert len(top_chunks) > 0, "Reranker returned no results!"
    print("\n Reranking passed")
    return top_chunks


def test_generation(query: str, top_chunks):
    """Verify the LLM generates a cited answer."""
    print(f"\n{'─' * 55}")
    print("TEST 3: Answer Generation")
 
    t = time.perf_counter()
    result = generate_answer(query, top_chunks)
    elapsed = (time.perf_counter() - t) * 1000
 
    answer         = result["answer"]
    citations_used = result["citations_used"]
    sources        = result["sources"]
 
    print(f"\nGenerated in {elapsed:.0f} ms")
    print(f"Citations used: {citations_used}")
    print(f"\nAnswer:\n{answer}")
    print(f"\nsources:\n{sources}")
 
    assert answer, "Generator returned an empty answer!"
    print("\n Generation passed")
    return result

def test_validation(generation_result: dict):
    """Verify the validator runs citation and faithfulness checks."""
    print(f"\n{'─' * 55}")
    print("TEST 4: Validation")
 
    t = time.perf_counter()
    validation = validate_answer(
        answer      = generation_result["answer"],
        sources     = generation_result["sources"],
        context_str = generation_result["context_str"],
        run_faithfulness_check=True,
    )
    elapsed = (time.perf_counter() - t) * 1000
 
    print(f"\nValidation completed in {elapsed:.0f} ms")
    print(f"Overall passed      : {validation['passed']}")
 
    citation_check = validation["citation_check"]
    print(f"\nCitation check:")
    print(f"  Has citations      : {citation_check['has_citation']}")
    print(f"  All valid          : {citation_check['all_citations_valid']}")
    print(f"  Cited numbers      : {citation_check['cited_source_numbers']}")
    print(f"  Coverage ratio     : {citation_check['coverage_ratio']:.0%}")
 
    faith = validation["faithfulness_check"]
    if faith:
        print(f"\nFaithfulness check:")
        print(f"  Is faithful        : {faith.get('is_faithful')}")
        print(f"  Score              : {faith.get('faithfulness_score'):.2f}")
        unsupported = faith.get("unsupported_claims", [])
        if unsupported:
            print(f"  Unsupported claims : {unsupported}")
 
    if validation["warnings"]:
        print(f"\nWarnings:")
        for w in validation["warnings"]:
            print(f" {w}")
 
    print("\n Validation passed")
    return validation


#  Full pipeline run 
def run_full_pipeline(query: str):
    """Run all Phase 2 tests for a single query end-to-end."""
    print(f"\n{'=' * 55}")
    print(f"  Running full pipeline for query:")
    print(f"  \"{query}\"")
    print(f"{'=' * 55}")
 
    t_total = time.perf_counter()
 
    # Loading resources
    vector_store              = load_vector_store()
    bm25_index, bm25_chunks   = load_bm_25_idx()
    reranker                  = load_reranker()
 
    # Running pipeline steps
    candidates      = test_hybrid_search(vector_store, bm25_index, bm25_chunks, query)
    top_chunks      = test_reranker(reranker, query, candidates)
    generation      = test_generation(query, top_chunks)
    validation      = test_validation(generation)
 
    total_ms = (time.perf_counter() - t_total) * 1000
 
    print(f"\n{'=' * 55}")
    print(f"  Pipeline complete in {total_ms:.0f} ms")
    print(f"  Answer: {generation['answer'][:80]}...")
    print(f"  Validation: {'passed' if validation['passed'] else 'failed'}")
    print(f"{'=' * 55}")

if __name__ == "__main__":
    print("=" * 55)
    print("  DocuMind — Phase 2 Test Suite")
    print("=" * 55)
    print(f"\nRunning test with query: '{TEST_QUERIES[0]}'")
 
    run_full_pipeline(TEST_QUERIES[0])









