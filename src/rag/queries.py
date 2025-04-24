# Prompt Builder functions

def ai_query(sub_target: str, extracted_passage: str):
    """
    Build a prompt/query to check if the extracted passage mentions AI and uses AI towards the specific sub_target
    
    Args:
        sub_target: The sub target definition
        extracted_passage: Obtained via the retriever, load from results/retrieve/company/year/passages.json

    Returns:
        ai_q: The built AI Query to be sent to the LLM(OpenAI) client
    """

    ai_q = f"""

    You are given a PASSAGE from an annual report of a company. You are also given a SUB-TARGET, which is a sub part of one of 17 Sustainable Development Goals.
    Your task is to check if the extracted PASSAGE mentions artificial intelligence or related-technological keywords such as machine learning, data science, computer vision and if they are used towards the Sustainable Goal or similar.
    If these keywords are mentioned then respond with YES and nothing else.
    If not then respond with NO and nothing else.

    So respond only with either YES or NO depending on the above situation.

    Here is the PASSAGE - {extracted_passage}

    Here is the SUB-TARGET - {sub_target}
    """

    return ai_q


def sdg_query(sub_target: str, extracted_passage: str):
    """
    Build a prompt/query to check if the extracted passage is towards the specific sub_target or not
    As a verification for Retrieved passages
    
    Args:
        sub_target: The sub target definition
        extracted_passage: Obtained via the retriever, load from results/retrieve/company/year/passages.json

    Returns:
        sdg_q: The built SDG Query to be sent to the LLM(OpenAI) client
    """

    sdg_q = f"""

    You are given a PASSAGE from an annual report of a company. You are also given a SUB-TARGET, which is a sub part of one of 17 Sustainable Development Goals.
    Your task is to check if the extracted PASSAGE actually relates to the SUB-TARGET.
    If the extracted PASSAGE is related to the SUB-TARGET then respond with YES and nothing else.
    If not then respond with NO and nothing else.

    So respond only with either YES or NO depending §on the above situation.

    Here is the PASSAGE - {extracted_passage}

    Here is the SUB-TARGET - {sub_target}
    """

    return sdg_q