import os
import re
import yaml

# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.schema.messages import SystemMessage, HumanMessage

PROMPTS_PATH = "config/prompts.yaml"
# DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

def load_prompts(path : str = PROMPTS_PATH) -> dict:

# Called on every generate_answer() so that prompt edits take effect without restarting the server.
    with open(path , "r") as f:
        return yaml.safe_load(f)
    

# below functions return a tuple of string and list of dictionary
# the string is combined context of all chunks with their assigned filename, page no, and citations. Ans this will be given to llm
# and the list of dicitonary contains the information of every chunks we are providing
def format_context_with_citations(
        chunks : list[tuple[Document, float]],
) -> tuple[str , list[dict]] :
    
    context_parts = []
    sources = []
    
    for i, (doc, score) in enumerate(chunks, start=1):

        source_file = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id" , "?")

        passage = (
            f"[{i}] Source: {source_file} | Page: {page}\n"
            f"{doc.page_content.strip()}"
        )

        context_parts.append(passage)

        sources.append({
            "citation_number": i,
            "file":            source_file,
            "page":            page,
            "chunk_id":        chunk_id,
            "relevance_score": round(float(score), 4),
            "text_preview":    doc.page_content[:250].strip(),
        })

    separators = "\n\n" + ("-"*50) + "\n\n"
    context_str = "\n\n" + separators.join(context_parts) + "\n\n"

    return context_str, sources

# this function parse all the [n] citation markers from the generated answer from llm.
def extract_citation_from_answer(answer : str) -> list[int] :

    matches = re.findall(r'\[(\d+)\]', answer)
    return sorted(set(int(m) for m in matches))


# this function generate our final answer by invoking the llm here with the prompts, user query and the context string (that we took out from the format_context_with_citations function) and then give this response to extract_citation_from_answer function to know info about the citations and at the end return the dictionary that contain answer, citations used, sources(from 2nd function only), and context string(from 2nd function only).
def generate_answer(
        query : str,
        chunks : list[tuple[Document,float]],
        model : str = DEFAULT_MODEL,
        temperature : float = 0,
) -> dict :
    
    if not chunks : 
        return{
            "answer":         "No relevant passages were retrieved for this query.",
            "citations_used": [],
            "sources":        [],
            "context_str":    "",
            "token_usage":    {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    
    context_str , sources = format_context_with_citations(chunks)

    prompts = load_prompts()
    system_prompt = prompts["rag"]["system"]
    user_template = prompts["rag"]["user"]

    user_meassage = user_template.format(
        question = query,
        context = context_str,
    )

    # llm = ChatOpenAI(model=model, temperature=temperature)
    llm = ChatGroq(model=model, temperature=temperature)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_meassage),
    ]


    print(f"\n  Generating answer with {model} ...")
    response = llm.invoke(messages)
    answer = response.content.strip()

    citations_used = extract_citation_from_answer(answer)

#  Extract token usage from LangChain response metadata.
    raw_usage = response.response_metadata.get("token_usage" , {})
    model_name = response.response_metadata.get("model_name", model)

    token_usage = {
        "model" : model_name,
        "prompt_tokens" : raw_usage.get("prompt_tokens", 0),
        "completion_tokens" : raw_usage.get("completion_tokens", 0),
        "total_tokens" : raw_usage.get("total_tokens" , 0),
    }

    print(f"  Answer generated. Citations used: {citations_used}"
        f"Tokens: {token_usage['prompt_tokens']}in / {token_usage['completion_tokens']}out")

    return{
        "answer" : answer,
        "citations_used" : citations_used,
        "sources" : sources,
        "context_str" : context_str,
        "token_usage" : token_usage,
    }

