import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PERSIST_PATH = "data/chroma_db"
CHROMA_COLLECTION_NAME = "documind"

EMBEDDING_MODEL_NAME = "all-miniLM-L6-v2"

def get_embedder() -> HuggingFaceEmbeddings:
    embedder = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL_NAME,
        model_kwargs = {"device" : "cpu"},
        encode_kwargs = {"normalize_embeddings" : True},
    )

    return embedder

def bulid_vector_store(chunks : list) -> Chroma :

    embedder = get_embedder()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_PERSIST_PATH,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    stored_count = vector_store._collection.count()
    print(f"\nVector store built successfully.")
    print(f"Vectors stored : {stored_count}")
    print(f"Saved to       : {CHROMA_PERSIST_PATH}/")

    return vector_store

def load_vector_store() -> Chroma:

    if not os.path.exists(CHROMA_PERSIST_PATH):
        raise FileNotFoundError(
            f"No vector store found at '{CHROMA_PERSIST_PATH}'.\n"
            f"Run 'python ingest.py' first to build it."
        )
    
    print(f"Loading vector store from '{CHROMA_PERSIST_PATH}' ...")

    embedder = get_embedder()

    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_PATH,
        embedding_function=embedder,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    
    print(f"vector store is loaded.")

    return vector_store

def vector_search(
        vector_store : Chroma,
        query : str,
        k : int = 20,
) -> list :
    
    results = vector_store.similarity_search_with_score(query,k=k)

    return results

def print_search_results(results: list, preview_chars : int = 200) -> None:

    print(f"\n{'='*55}")
    print(f"  Search Results ({len(results)} chunks)")
    print(f"{'='*55}")

    for i,(chunk,score) in enumerate(results):
        source = os.path.basename(chunk.metadata.get("source","unknown"))
        page = chunk.metadata.get("page","?")
        print(f"\n[{i+1}] Score: {score:.4f} | File: {source} | Page: {page}")
        print(f"     {chunk.page_content[:preview_chars].strip()} ...")


