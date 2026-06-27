import os
import sys
import subprocess
from pathlib import Path
 
 
def indexes_exist() -> bool:
    """
    Return True when both the ChromaDB vector store and BM25 index are present.
    A non-empty chroma_db directory + the pickle file means ingest has run.
    """
    chroma_path = Path("data/chroma_db")
    bm25_path   = Path("data/bm25_index.pkl")
 
    chroma_ok = chroma_path.exists() and any(chroma_path.iterdir())
    bm25_ok   = bm25_path.exists() and bm25_path.stat().st_size > 0
 
    return chroma_ok and bm25_ok
 
 
def run_ingest() -> None:
    """
    Run ingest.py as a subprocess.
    Exits the container if ingest fails so Railway surfaces the error
    rather than silently starting a broken API.
    """
    print("\n[start.py] Running ingest.py to build document indexes ...")
    result = subprocess.run(
        [sys.executable, "ingest.py"],
        check=False,
    )
    if result.returncode != 0:
        print("\n[start.py] ERROR: ingest.py exited with a non-zero code.")
        print("           Check that your PDF files are in data/documents/")
        print("           and that API key is set correctly.")
        sys.exit(1)
 
    print("[start.py] Ingest complete.\n")
 
 
def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
 
    port = int(os.environ.get("PORT", 8000))
 
    print("=" * 55)
    print("  DocuMind — Container startup")
    print("=" * 55)
 
    #  Check for PDF source documents 
    doc_dir = Path("data/documents")
    pdfs    = list(doc_dir.glob("**/*.pdf")) if doc_dir.exists() else []
 
    if pdfs:
        print(f"\n  Source documents  : {len(pdfs)} PDF file(s) found")
        for pdf in pdfs:
            print(f"    - {pdf.name}")
    else:
        print("\n   WARNING: No PDF files found in data/documents/")
        print("     The API will start but queries will return no results.")
        print("     Add PDFs and restart to fix.\n")
 
    #  Check / build indexes 
    if indexes_exist():
        print("\n  Indexes           : found — skipping ingest")
    elif pdfs:
        print("\n  Indexes           : not found — building now ...")
        run_ingest()
    else:
        print("\n  Indexes           : not found, but no PDFs to ingest")
        print("                      starting API without document index\n")
 
    #  Environment check 
    if not os.environ.get("GROQ_API_KEY"):
        print("\n  WARNING: GROQ_API_KEY is not set.")
        print("     Answer generation will fail. Set it in Railway => Variables.\n")
 
    #  Start uvicorn 
    print(f"\n  Starting API on port {port} ...")
    print(f"  Docs available at : http://localhost:{port}/docs\n")
 
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )
 
 
if __name__ == "__main__":
    main()