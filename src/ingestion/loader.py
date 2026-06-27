import os
from pathlib import Path

from langchain_community.document_loaders import(
    DirectoryLoader,
    PyPDFLoader,
    WebBaseLoader
)

def load_pdfs(directory : str = "data/documents") -> list:
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(
            f"Directory '{directory}' not exist."
            f"Look at it."
        )
    
    pdf_files = list(directory_path.glob("**/*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No pdf files are found in '{directory}'."
        )
    
    print(f"Found {len(pdf_files)} pdf in '{directory}'.")

    for f in pdf_files:
        print(f" - {f.name}")

    loader = DirectoryLoader(
        str(directory_path),
        glob= "**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    documents = loader.load()

    print(f"\nLoaded {len(documents)} pages total from {len(pdf_files)} pdf files")

    return documents

def load_single_pdf(file_path : str) -> list : 
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: '{file_path}'"
        )
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from '{file_path}'")
    return documents

def load_webpage(url : str) ->list:

    loader = WebBaseLoader(url);
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from '{url}'")
    return documents

def load_all_documents(directory : str = "data/documents") ->list :
    
    all_docs = []
    
    all_docs.extend(load_pdfs(directory))

    print(f"\nTotal pages loaded across all documents: {len(all_docs)}")
    return all_docs

