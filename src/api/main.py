import time

# contextlib is the library for managing the resources
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
# FastApi is the actual engine that convert our computes into live web server
# HTTPexecution - use for sending error meassages back to the user

from fastapi.middleware.cors import CORSMiddleware
# cors stands for cross origin resource sharing

from pydantic import BaseModel, Field
# pydantics is data-checking library, when the user send any query to server in json then it is use to check whether user sent the string only, not a malicious code

from src.retrieval.vector_store import load_vector_store
from src.retrieval.bm25_index import load_bm_25_idx
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import load_reranker, rerank
from src.generation.generator import generate_answer
from src.generation.validator import validate_answer
from src.monitoring.tracer import DocuMindTrace

# All loaded models/indexes live here for the process lifetime.
_resources : dict = {}

@asynccontextmanager
async def lifespan(app : FastAPI):
    """Load all models at startup; release on shutdown."""
    print("\n" + "=" * 55)
    print("  DocuMind API — Loading resources ...")
    print("=" * 55)

    _resources["vector_store"] = load_vector_store()

    bm_25_index, bm_chunks = load_bm_25_idx()
    _resources["bm25_idx"] = bm_25_index
    _resources["bm_chunks"] = bm_chunks

    _resources["reranker"] = load_reranker()

    print("\nAll resources loaded. API is ready.\n")


# yield tells the python to pause the function here, now FastAPI open the website and allow user to use it. when we want to close the server then it unpauses and then clear all the resources that it have stored in _resources so far.
    yield

    _resources.clear()
    print("Resources released.")


# app definition
app = FastAPI(
    title = "documind API",
    description=(
        "Hybrid RAG pipeline: BM25 + vector search fused with RRF, "
        "cross-encoder reranking, and cited answer generation. "
    ),
    version= "beta",
    lifespan=lifespan,
)

# middleware is something that sits between the client browser and the FastAPI backend server
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], #it check from which website does the query came from and allowed only those which are mentiones under the allowed_origins, here now we have "*", which means any website can send the request to our api
    allow_methods = ["GET", "POST"], # the request can be of 2 types, to ask for data(GET) and to send data(POST)
    allow_headers = ["*"], #when the website talks to API, it send the hidden metadata called headers, this "*" tells to accept all type of metadata headers
)


class QueryRequest(BaseModel) :
    question : str = Field(
        ...,  #tells that this particular field is absolutely required, do not accept the request without it.
        min_length=5,
        max_length=500,
        example="What was Amazon's total revenue in fiscal year 2025?",
    )

    # for the number of chunks to retrieve, 5 is default and ranges from 1 to 10.
    top_k : int = Field(
        default=5,
        ge = 1,
        le = 10,
        description="Number of source passages to retrieve and cite (1-10).",
    )

    run_validation : bool = Field(
        default=True,
        description=(
            "Run faithfulness validation after generation. "
            "Set False to reduce latency."
        ),
    )

    session_id: str | None = Field(
        default=None,
        description="Optional session ID to group related queries in Langfuse.",
    )


    # this block automatically populates the text boxes with perfectly formatted example question
    class Config :
        json_schema_extra = {
            "example" : {
                "question" : "What were the total net sales for the iPhone segment, and did they increase or decrease?",
                "top_k" : 5,
                "run_validation" : True,
                "session_id" : None,
            }
        }

class SourceInfo(BaseModel) :
    citation_number: int
    file:            str
    page:            int | str
    relevance_score: float
    text_preview:    str

class QueryResponse(BaseModel):
    question:         str
    answer:           str
    citations_used:   list[int]
    sources:          list[SourceInfo]
    validation:       dict | None
    latency_ms:       float
    trace_id:         str | None


@app.get("/health", tags=["System"])
def health_check() :
    """
    Liveness check — confirms all models are loaded and the API is ready.
    Returns 200 OK when healthy.
    """
    return {
        "status":               "ok",
        "vector_store_loaded":  "vector_store" in _resources,
        "bm25_loaded":          "bm25_idx"   in _resources,
        "reranker_loaded":      "reranker"     in _resources,
    }

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request : QueryRequest) :

    """  
    main  query endpoint
    steps 
    1. hybrid search
    2. cross encoder reranking
    3. llm generation
    4. validation

    """
    t_total = time.perf_counter()
    tracer = DocuMindTrace()
    tracer.start(request.question, session_id=request.session_id)

    generation = None
    validation = None


    try :
        # hybrid retrieval
        t = time.perf_counter()
        candidates = hybrid_search(
            vector_store =  _resources["vector_store"],
            bm25_index =  _resources["bm25_idx"],
            chunks =  _resources["bm_chunks"],
            query =  request.question,
            k= 20,
        )
        hybrid_ms = (time.perf_counter() - t) * 1000

        if not candidates :
            raise HTTPException(
                status_code=404,
                detail="No relevant passages found. Try rephrasing your question.",
            )
        
        tracer.record_hybrid_search(
            elapsed_ms   = hybrid_ms,
            query        = request.question,
            vector_count = len(candidates), 
            bm25_count   = len(candidates),   
            fused_count  = len(candidates),
        )

        # rerank
        t = time.perf_counter()
        top_chunks = rerank(
            reranker = _resources["reranker"],
            query= request.question,
            candidates= candidates,
            top_n=  request.top_k,
        )

        rerank_ms = (time.perf_counter() - t) * 1000
        tracer.record_rerank(
            elapsed_ms   = rerank_ms,
            input_count  = len(candidates),
            output_count = len(top_chunks),
            top_score    = top_chunks[0][1]  if top_chunks else 0.0,
            bottom_score = top_chunks[-1][1] if top_chunks else 0.0,
        )


        # generate cited answer
        t = time.perf_counter()
        generation = generate_answer(
            query=  request.question,
            chunks= top_chunks,
        )
        gen_ms = (time.perf_counter() - t) * 1000

        tu = generation["token_usage"]
        tracer.record_generation(
            elapsed_ms        = gen_ms,
            question          = request.question,
            answer            = generation["answer"],
            model             = tu["model"],
            prompt_tokens     = tu["prompt_tokens"],
            completion_tokens = tu["completion_tokens"],
            citations_used    = generation["citations_used"],
            context_str       = generation["context_str"],
        )

        # validate
        if request.run_validation :
            t = time.perf_counter()
            validation = validate_answer(
                answer= generation["answer"],
                sources=  generation["sources"],
                context_str=  generation["context_str"],
            )
            val_ms =  (time.perf_counter() - t) * 1000

            tracer.record_validation(val_ms, validation)
            tracer.record_scores(validation, generation)

    except HTTPException as e:
        tracer.record_error(e, "hybrid-search")
        tracer.finish("", (time.perf_counter() - t_total) * 1000, error="404 no candidates")
        raise # re-raise our own 404 as-is

    except FileNotFoundError as e:
        tracer.record_error(e, "resource")
        tracer.finish("", (time.perf_counter() - t_total) * 1000, error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"Required resource not found: {e}.",
        )
    except Exception as e:
        tracer.record_error(e, "pipeline")
        tracer.finish("", (time.perf_counter() - t_total) * 1000, error=str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Pipeline error : {str(e)}"
        )
    
    total_ms = (time.perf_counter() - t_total)*1000
    tracer.finish(generation["answer"], total_ms)

    return QueryResponse(
        question = request.question,
        answer= generation["answer"],
        citations_used = generation["citations_used"],
        sources= [SourceInfo(**s) for s in generation["sources"]],
        validation= validation,
        latency_ms = round(total_ms, 1),
        trace_id= tracer.trace_id,
    )

