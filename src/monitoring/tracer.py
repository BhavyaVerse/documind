import os
from functools import lru_cache
from langfuse import Langfuse
 
 
# OpenAI pricing (USD per 1 million tokens)

# Langfuse can also auto-calculate costs for known models but these are a fallback.
# although i have used groq free api for my project, but in case if anyone use paid one
_COST_PER_1M: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile" : {"input" : 0.59, "output" : 0.79},
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.60},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-5.4-nano": {"input": 0.20,"output": 1.25},
    "gpt-5.4-mini": {"input": 0.75,"output": 4.50},
    "gpt-5.4": {"input": 2.50,"output": 15.00},
    "gpt-5.5-pro": {"input": 30.00,"output": 180.00},
}
 
 
def _calculate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:

    pricing = _COST_PER_1M.get(model, {"input": 0.15, "output": 0.60}) #for the default fallback, used gpt-4o-mini
    cost = (prompt_tokens     / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]
    
    return round(cost, 8)

#  Singleton Langfuse client 
# we are creating only one connection to langfuse server for every process. if we run the process and user make lets say 100 query then for all those queries only this one connection is gonna be used.
# we ain't creating the different connection for every query as this will help to reduce the latency. 
@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    
    lf = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"), #this is for fallback
    )
    return lf


# per request trace

class DocuMindTrace:
    """
    Manages one Langfuse trace for a single end-to-end query.
 
    We will make our trace object of this class type and use it for tracing.
    
    Every method is safe to call when Langfuse is unavailable exceptions are caught and printed as warnings, never raised.

    Not raising the error and only printing because then it will be Fail - open system, If we raise the error then it would be Fail - close system, means if langfuse is down then our project will not do even its primary task that is generate answer for user's query.

    But now the pipeline will always complete regardless of monitoring state.
    """

    # Constructor
    def __init__(self):
        self._lf:     Langfuse | None = None
        self._trace:  object | None = None
        self.trace_id: str | None = None

    # this is the function which we run at the starting when we make a object of this class type
    # as we are creating the Fail open system so we wrap everything in the "try except" block throughout all the methods
    # providing session id is optional, provide it or it will take it as none

    def start(self, question: str, session_id: str | None = None) -> None:
        
        try:
            self._lf    = get_langfuse()
            self._trace = self._lf.trace(  #this .trace() is method from langfuse sdk
                name="documind-query",
                input={"question": question},
                session_id=session_id,
                tags=["monitoring"],
                metadata={"pipeline_version": "1.0.0"},
            )
            # grabing the newly generated unique id by langfuse
            self.trace_id = self._trace.id

            print(f"[Tracer] Trace started: {self.trace_id}")
        except Exception as e:
            print(f"[Tracer] Warning — could not start trace: {e}")


    # we will call this function at the last when we are done with tracing to wrap all the things and flush out the remaining data to lanfuse server.
    # flush is critical, without it events may be lost on process shutdown
    def finish(
        self,
        answer: str,
        total_latency_ms: float,
        error: str | None = None,
    ) -> None:
        
        # just for safety check, if the start function is not able to process completely and it don't get the languse object then self._lf will be None, so in that situation it would be better to return, otherwise running the update method on None will give the error.
        # and we will be performing this safety check in all the upcoming functions
        if not self._trace:
            return
        try:
            self._trace.update(
                output={"answer": answer[:500]},   # preview in Langfuse UI
                metadata={
                    "total_latency_ms": round(total_latency_ms, 1),
                    "error": error,
                },
            )
            self._lf.flush()

            print(f"[Tracer] Trace finished and flushed: {self.trace_id}")

        except Exception as e:
            print(f"[Tracer] Warning  could not finish trace: {e}")

    # this function is taking two args, one is the actual python exception that we got and another is the step at which we got it
    def record_error(self, error: Exception, step: str) -> None:

        if not self._trace:
            return
        try:
            self._trace.update(
                metadata={
                    "error_step":  step,
                    "error_message": str(error)[:500],
                    "status": "ERROR",
                },
                level="ERROR",
            )
            self._lf.flush()
        except Exception:
            pass



# logging the hybrid retrieval step as span
    def record_hybrid_search(
        self,
        elapsed_ms: float,
        query: str,
        vector_count: int,
        bm25_count: int,
        fused_count: int,  # no. of chunks after RRF fusion(basically top_k)
    ) -> None:
        
        if not self._trace:
            return
        try:
            span = self._trace.span(
                name="hybrid-search",
                input={"query": query},
            )
            span.end(
                output={
                    "vector_candidates": vector_count,
                    "bm25_candidates":   bm25_count,
                    "fused_candidates":  fused_count,
                },
                metadata={"latency_ms": round(elapsed_ms, 1)},
            )
        except Exception as e:
            print(f"[Tracer] Warning — could not record hybrid-search span: {e}")



# logging the cross-encoder reranking step as span
    def record_rerank(
        self,
        elapsed_ms: float,
        input_count: int,
        output_count: int,
        top_score: float,
        bottom_score: float,
    ) -> None:
       
        if not self._trace:
            return
        try:
            span = self._trace.span(name="rerank")
            span.end(
                output={
                    "input_candidates":  input_count,
                    "output_chunks":     output_count,
                    "top_ce_score":      round(float(top_score), 4),
                    "bottom_ce_score":   round(float(bottom_score), 4),
                    "score_spread":      round(float(top_score - bottom_score), 4),
                    # model confidency is dirctly proportional to wider score spread gap
                },
                metadata={"latency_ms": round(elapsed_ms, 1)},
            )
        except Exception as e:
            print(f"[Tracer] Warning — could not record rerank span: {e}")
 


#  logging the generation step as Lanfuse generation (special span type)
    def record_generation(
        self,
        elapsed_ms: float,
        question: str,
        answer: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        citations_used: list[int],
        context_str: str,
    ) -> None:
 
        # The context_str is truncated to 3000 chars — enough to review in the UI without hitting Langfuse event size limits.

        if not self._trace:
            return
        try:
            cost_usd = _calculate_cost_usd(model, prompt_tokens, completion_tokens)
 
            # using the generation() span type — not span() so Langfuse renders it with token/cost UI and includes it in cost analytics.
            gen = self._trace.generation(
                name="llm-generation",
                model=model,
                model_parameters={"temperature": 0},
                input=context_str[:3000],   # full prompt context (truncated)
                usage={
                    "input":  prompt_tokens,
                    "output": completion_tokens,
                    "total":  prompt_tokens + completion_tokens,
                    "unit":   "TOKENS",
                },
            )
            gen.end(
                input=question,
                output=answer,
                metadata={
                    "latency_ms":     round(elapsed_ms, 1),
                    "citations_used": citations_used,
                    "cost_usd":       cost_usd,
                },
            )
        except Exception as e:
            print(f"[Tracer] Warning — could not record generation span: {e}")



# logging the citation and faihfulness validation step
    def record_validation(
        self,
        elapsed_ms: float,
        validation: dict,
    ) -> None:

        if not self._trace:
            return
        try:
            faith = validation.get("faithfulness_check") or {}
            span = self._trace.span(name="validation")
            span.end(
                output={
                    "passed":            validation.get("passed"),
                    "has_citations":     validation["citation_check"]["has_citation"],
                    "coverage_ratio":    validation["citation_check"]["coverage_ratio"],
                    "faithfulness":      faith.get("faithfulness_score"),
                    "warnings":          validation.get("warnings", []),
                },
                metadata={"latency_ms": round(elapsed_ms, 1)},
            )
        except Exception as e:
            print(f"[Tracer] Warning — could not record validation span: {e}")

    
    # calculating scores 
 
    def record_scores(
        self,
        validation: dict,
        generation: dict,
    ) -> None:
        """      
        Scores recorded:
            citation_coverage   — fraction of retrieved sources cited (0-1)
            validation_passed   — binary pass/fail (0 or 1)
            faithfulness        — LLM-judged faithfulness score (0-1)
        """
        if not self._trace or not self._lf or not self.trace_id:
            return
        try:
            citation_check = validation.get("citation_check", {})
            faith_check    = validation.get("faithfulness_check")
 
            self._lf.score(
                trace_id=self.trace_id,
                name="citation_coverage",
                value=citation_check.get("coverage_ratio", 0.0),
                comment=f"Citations used: {generation.get('citations_used', [])}",
            )
 
            self._lf.score(
                trace_id=self.trace_id,
                name="validation_passed",
                value=1.0 if validation.get("passed") else 0.0,
            )
 
            if faith_check:
                self._lf.score(
                    trace_id=self.trace_id,
                    name="faithfulness",
                    value=faith_check.get("faithfulness_score", 0.0),
                    comment=(
                        "Unsupported: " + str(faith_check.get("unsupported_claims", [])[:1])
                        if not faith_check.get("is_faithful")
                        else "All claims supported"
                    ),
                )
        except Exception as e:
            print(f"[Tracer] Warning — could not record scores: {e}")
 
 