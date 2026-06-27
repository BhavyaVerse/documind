import os
from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_store import(
    load_vector_store,
    vector_search,
    print_search_results,
)
from langchain_openai import ChatOpenAI

TEST_QUERIES = [
    "What specific risks does Amazon mention regarding its fulfillment centers and supply chain?",
    "What were the total net sales for the iPhone segment, and did they increase or decrease?",
]

# although there is no need of building this function (as we also do the retrieval work in the test_generation function), we are creating this to just see if this system is giving the relevant chunks.
def test_retrieval(query : str, k : int = 3) -> list:
    print(f"\n{'─'*55}")
    print(f"RETRIEVAL TEST")
    print(f"Query: {query}")

    store = load_vector_store()
    results = vector_search(store,query,k=k)
    print_search_results(results)

    return results

def test_generation(query : str) -> str:

    print(f"\n{'─'*55}")
    print(f"GENERATION TEST")
    print(f"Query: {query}")

    store = load_vector_store()
    results = vector_search(store, query, k=5)

    # Build a simple context string
    context = "\n\n".join(chunk.page_content for chunk,_ in results)
    # By using an underscore _ for the second variable, we are telling Python to take only chunk text,and ignore the mathematical score for now.

    llm = ChatOpenAI(model = "gpt-5.4-mini", temperature=0) # 0 temp helps to prevent hallucinations.

    prompt = (
        f"Answer the following question using only the context provided below.\n"
        f"If the context does not contain the answer, say 'I cannot find this in the documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
    )

    response = llm.invoke(prompt)
    print(response.content)

    return response.content

if __name__ == "__main__":
    print("=" * 55)
    print("  DocuMind — Phase 1 Test")
    print("=" * 55)

    query = TEST_QUERIES[0]

    # Test 1: Does retrieval return relevant chunks?
    test_retrieval(query, k=3)

    # Test 2: calling ai for the answer
    test_generation(query)
