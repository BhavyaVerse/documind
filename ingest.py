import os
from dotenv import load_dotenv
load_dotenv()

from src.ingestion.loader import load_all_documents
from src.ingestion.chunker import chunk_documents,preview_chunks
from src.retrieval.vector_store import bulid_vector_store
from src.retrieval.bm25_index import build_bm25_idx, save_bm25_idx

def main():
    print("=" * 55)
    print("  DocuMind — Ingestion Pipeline")
    print("=" * 55)

#step 1 - Load
    print("\n[1/4] Loading documents...")
    documemts = load_all_documents("data/documents")

    if not documemts:
        print("\nNo documents loaded. Exiting.")
        print("Add some PDF files and try again.")
        return
    
#step 2 - Chunk
    print("\n[2/4] Splitting documents into chunks ...")
    chunks = chunk_documents(documemts)
    preview_chunks(chunks)

#step 3 - vector store
    print("\n[3/4] Embedding chunks and saving to ChromaDB ...")
    bulid_vector_store(chunks)

# step 4 - bm25 keyword index
    print("\n[4/4] Building BM25 keyword index ...")
    bm25_idx = build_bm25_idx(chunks)
    save_bm25_idx(bm25_idx,chunks)

# printing details
    print("\n" + "=" * 55)
    print("  Ingestion complete!")
    print(f"  Pages loaded   : {len(documemts)}")
    print(f"  Chunks created : {len(chunks)}")
    print(f"  Vectors Stored at      : data/chroma_db/")
    print(f"  Bm25_index : data/bm25_idx.pkl")
    print("=" * 55)

if __name__ ==  "__main__": 
    main()