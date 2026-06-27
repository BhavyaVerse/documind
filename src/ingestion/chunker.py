from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(
        documents : list,
        chunk_size : int = 400, #in tokens
        chunk_overlap : int = 50, 
) -> list:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size*4, #converting tokens into characters
        chunk_overlap = chunk_overlap*4,
        separators=["\n\n" , "\n", ".", " ", ""],
        length_function  = len,
    )

    chunks = splitter.split_documents(documents)

    for i,chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    total_chars = sum(len(c.page_content) for c in chunks)
    avg_size = total_chars // max(len(chunks),1)

    print(f"Created {len(chunks)} chunks from {len(documents)} pages.")
    print(f"Average chunk size : {avg_size} characters (~{avg_size // 4} tokens)")
    print(f"Smallest chunk     : {min(len(c.page_content) for c in chunks)} characters")
    print(f"Largest chunk      : {max(len(c.page_content) for c in chunks)} characters")

    return chunks

def preview_chunks(chunks : list, n:int = 3) ->None:
    print(f"\n{'='*55}")
    print(f" chunks preview (first {n} chunks)")
    print(f"\n{'='*55}")

    for i in range(min(n,len(chunks))):
        c =chunks[i]
        source = c.metadata.get("source", "unknown")
        print(f"\nchunk[{i}]")
        print(f" source  :  {source}")
        print(f" page    :  {c.metadata.get('page', 'unknown')}")
        print(f" size    :  {c.metadata['chunk_size']}")
        print(f"preview  :  {c.page_content[:180].strip()}...")

    if(len(chunks) >= 2):
        print(f"\n{'='*55}")
        print("  Overlap Verification")
        print(f"{'='*55}")
        print(f"chunk[0] tail : ...{chunks[0].page_content[-120:].strip()}")
        print(f"chunk[1] head : {chunks[1].page_content[:120].strip()}...")
        print("  (The text above should visibly overlap)")

