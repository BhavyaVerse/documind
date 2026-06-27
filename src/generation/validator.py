# we are doing following types of validations

# first validation - in this, we check if all the citations which are there in the generated answer are correct or they are hallucinated by llm. Here we will return the dictionary that tells us the following five things :
# is answer even have any citations (T/F), 
# is it have any invalid citations(T/F), 
# if it has invalid citations then what are they all,
#  what all citations does the answer contain, and 
# what is the coverage ratio, it tells the ratio of how many chunks llm used among all the given chunks

# second validation - by llm to verify that every claim in the generated answer is grounded in the given source (no hallucination), it return the dictionary ehich contain the following
# is the generated answer faithful[T/F]
# if not, what are the unsupported claims
# faithfulness score( from 0 to 1)


import json
import yaml
import re

from langchain.schema.messages import SystemMessage,HumanMessage
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

PROMPTS_PATH = "config/prompts.yaml"

FAITHFULNESS_THRESHOLD = 0.7 # minimum score to mark the answer as passed

def _load_prompts(path : str = PROMPTS_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# validation for the citations
def check_citation_coverage(
        answer : str,
        sources : list[dict],
) -> dict :
     
    #  taking all the citations numbers that are present in the generated answer
    cited_numbers = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))

    # taking all the citation numbers of all the sources that we have provided to llm
    valid_numbers = set(s["citation_number"] for s in sources)

    # subtracting the above two sets, if cited_numbers contain any invalid citation then this set have some elements otherwise it would be empty
    invalid_citation = sorted(cited_numbers - valid_numbers)

    coverage_ratio = (
        len(cited_numbers & valid_numbers) / len(valid_numbers)
        if valid_numbers else 0.0
    )

    return{
        "has_citation" : len(cited_numbers) >0,
        "all_citations_valid" : len(invalid_citation) == 0,
        "invalid_citations" : invalid_citation,
        "cited_source_numbers" : sorted(cited_numbers),
        "coverage_ratio" : round(coverage_ratio,3),
    }

def check_faithfulness(
        answer : str,
        context_str : str,
        model : str = "llama-3.3-70b-versatile"
) -> dict :
    
    prompts = _load_prompts()

    system_prompt = prompts["faithfulness_check"]["system"]
    user_template = prompts["faithfulness_check"]["user"]

    user_message = user_template.format(
        answer = answer,
        context = context_str,
    )

    # llm = ChatOpenAI(model=model, temperature=0)
    llm = ChatGroq(model=model, temperature=0)

    

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # llm may give the answer that start with " ``` json " and ends with " ``` " with some whitespace so here we are basically removing these, so that json library can load them without any error.
    raw = re.sub(r'^```(?:json)?\s*', '',raw)
    raw = re.sub(r'\s*```$','',raw).strip()


    # By wrapping the code in a `try / except` block, we catch that crash before it happens.
    try:
        result = json.loads(raw)
        # Ensure all expected keys are present, if not then default value occupy the palce
        result.setdefault("is_faithful" , False)
        result.setdefault("unsupported_claims" , [])
        result.setdefault("faithfulness_score" , 0.0)
        return result
    
    except json.JSONDecodeError :

        # if we get the error while parsing the json file then we consider that answer is not safe to provide to user so we return the following dictionary that give false to is_faithful with 0 faithfulness score and with the parse error
        return {
            "is_faithful" : False,
            "unsupported_claims" : ["Faithfulness check response could not be parsed."],
            "faithfulness_score" : 0.0,
            "parse_error" : raw[:300],
        }

# full validation suite
# return the dictionary of following type 
'''
    passed             bool   — overall pass/fail
    citation_check     dict   — result of check_citation_coverage()
    faithfulness_check dict   — result of check_faithfulness() or None
    warnings           list   — warning messages'''

def validate_answer(
        answer : str,
        sources : list[dict],
        context_str : str,
        run_faithfulness_check : bool = True,
) -> dict :
    
    warnings = []

# check 1 : citation
    citation_result = check_citation_coverage(answer, sources)
    
    if not citation_result["has_citation"] : 
        warnings.append("Answer contains no [n] citation markers.")

    if not citation_result["all_citations_valid"] :
        bad = citation_result["invalid_citations"]
        warnings.append(f"Answer cites non-existent source numbers: {bad}")


# check 2 : faithfulness score
    faith_result = None
    if run_faithfulness_check : 
        print("Running Faithfulness check")
        faith_result = check_faithfulness(answer, context_str)
        score = faith_result.get("faithfulness_score", 1.0)

        if not faith_result.get("is_faithful", True):
            unsupported = faith_result.get("unsupported_claims", [])
            if unsupported : 
                warnings.append(
                     f"Potentially unsupported claims detected: {unsupported[:2]}"
                    # Showing at most 2 examples to keep warnings concise
                )

        print(f"  Faithfulness score: {score:.2f}")

    faith_score = (
        faith_result.get("faithfulness_score", 1.0)
        if faith_result is not None
        else 1.0
    )


# give true if our answer passed both the validations
    passed = (
        citation_result["has_citation"] and citation_result["all_citations_valid"] and faith_score >= FAITHFULNESS_THRESHOLD
    )

    return {
        "passed" : passed,
        "citation_check" : citation_result,
        "faithfulness_check" : faith_result,
        "warnings" : warnings,
    }