from sentence_transformers import CrossEncoder
from langchain.schema import Document

# free and local model by microsoft
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# module-level singleton - avoids reloading the model on every API call
_reranker_instane : CrossEncoder | None = None

def load_reranker() -> CrossEncoder :

    # using the module-level singleton so the module is loaded once per process. The first call takes few seconds and then every subsequent call return immediately.

    global _reranker_instane

    if _reranker_instane is None :

        print(f"Loading reranker model: {RERANKER_MODEL_NAME} ...")
        _reranker_instane = CrossEncoder(
            RERANKER_MODEL_NAME,
            max_length=512,
        )
        print("Reranker model ready.")

    return _reranker_instane


def rerank(
        reranker : CrossEncoder,
        query : str,
        candidates : list[tuple[Document,float]],
        top_n : int = 5,
) -> list[tuple[Document,float]] :

# creating the list of tuple for feeding it to the our reranker model
    pairs = [(query, doc.page_content) for doc, _ in candidates]

# it will return the numpy array of score for each chunk
    score = reranker.predict(pairs)

# creating the list of tuple for every chunk with their score given by our reranker model
    scored = list(zip( [doc for doc, _ in candidates], score.tolist()))

    scored.sort(key= lambda x : x[1], reverse=True)

    top = scored[:top_n]

    best_score  = top[0][1]
    worst_score = top[-1][1]
    print(f"  Reranking done. Score range: {worst_score:.3f} – {best_score:.3f}")

    return top