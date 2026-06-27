from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import vector_search
from src.retrieval.bm25_index import bm25_search

def reciprocal_rank_fusion(
        results_lists : list[list[tuple[Document,float]]], #this contains 2 list, one list from vector based searching and one from the keyword based searching

        k : int =60, #this is the number that is used to calculate the rrf score, 60 is the standard, as it prevents the rank1 chunk to completely dominating over the #2 and #3 chunks

)-> list[tuple[Document,float]] :
    
    fused : dict[int , tuple[Document,float]] = {}
    # key of this dictionary contains the chunk id , it helps us to know whether any chunk has come already or not. And value contains the tuple of chunk and the rrf score.

    for results in results_lists:

        for rank,(doc , _score) in enumerate(results,start=1):

            chunk_id = doc.metadata.get("chunk_id")

            if chunk_id is None :
                #chunk without id will not be processed by our pipeline so we skipped them rather than crashing.
                continue

            rrf_contribution = 1 / (k + rank)

            if chunk_id in fused:
                existing_doc , existing_score = fused[chunk_id]
                fused[chunk_id] = (existing_doc , existing_score + rrf_contribution)
            else:
                fused[chunk_id] = (doc, rrf_contribution)

    return sorted(fused.values() , key= lambda x : x[1], reverse=True)


#  Retrieve the top-k most relevant chunks using hybrid search.
def hybrid_search(
        vector_store : Chroma,
        bm25_index : BM25Okapi,
        chunks : list,
        query : str,
        k : int =20,
) -> list[tuple[Document,float]]:
    

    fetch_k = k*2
    # here we are over fething the chunks as it will give the more candidates to fuse together
    # for example if k is 20 but there is the chunk at #22 at ChromaDB and #21 at BM25index then this is good hybrid match, but if fetched 20 from each then RRF algo would not be able to see these.

    vector_results = vector_search(vector_store, query, k= fetch_k)
    print(f"  Vector search  => {len(vector_results)} results")

    bm25_results = bm25_search(bm25_index, chunks, query, k=fetch_k)
    print(f"  BM25 search    => {len(bm25_results)} results")

    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    return fused[:k]
