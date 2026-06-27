import os
import pickle

from pathlib import Path

from rank_bm25 import BM25Okapi
from langchain.schema import Document

BM25_INDEX_PATH = "data/bm25_index.pkl"

def _tokenize(text : str) -> list[str] :
    # simple whitespace tokenizer
    return text.lower().split()

def build_bm25_idx(chunks : list) -> BM25Okapi:

    '''BM25 scores documents based on:
        - Term frequency (TF):  how often query words appear in the chunk
        - Inverse document frequency (IDF): how rare the word is across all chunks
        - Document length normalization: shorter documents get a slight boost '''
    
    print(f"Building BM25 index from {len(chunks)} chunks ...")

    tokenize_corpus = [ _tokenize(chunk.page_content) for chunk in chunks]
    index = BM25Okapi(tokenize_corpus)

    print("BM25 index built successfully.")
    return index

def save_bm25_idx(
        index : BM25Okapi,
        chunks : list,
        path : str = BM25_INDEX_PATH,
) -> None:
    
    '''We must save chunks alongside the index because BM25 only returns
    integer indices into the corpus — we need the original Document
    objects to retrieve the actual text and metadata.'''

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    payload = {"index" : index, "chunks" : chunks}

    with open(path , "wb") as f:
        pickle.dump(payload, f)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"BM25 index saved to '{path}' ({size_mb:.1f} MB).")

def load_bm_25_idx(
        path : str = BM25_INDEX_PATH,
) -> tuple[BM25Okapi, list] :
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No BM25 index found at '{path}'."
        )
    
    with open(path , "rb") as f:
        payload = pickle.load(f)

    index = payload["index"]
    chunks = payload["chunks"]

    print(f"BM25 index loaded. Contains {len(chunks)} chunks.")
    return index, chunks

def bm25_search(
        index : BM25Okapi,
        chunks : list,
        query : str,
        k : int = 20,
) -> list[tuple[Document , float]] :
    
    # returning the list of (document, score) tuple for the top k chunks based on the score(high to low).
    
    tokenize_query = _tokenize(query)
    score = index.get_scores(tokenize_query)

    top_indices = sorted(
        range(len(score)),
        key=lambda i:score[i],
        reverse=True,
    )[:k]

    results = [(chunks[i], float(score[i])) for i in top_indices ]

    return results


