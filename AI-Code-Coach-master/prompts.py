from langchain.prompts import PromptTemplate

DEBUG_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a senior software engineer.

User issue:
{query}

Relevant code from repository:
{context}

Debug the problem. Identify root cause and suggest fixes.
"""
)

TRANSLATE_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""
Translate the following code as requested by the user.

User request:
{query}

Code:
{context}

Provide clean translated code.
"""
)

EXPLAIN_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""
Explain the algorithm and design used in the following code.

User request:
{query}

Code:
{context}

Explain clearly.
"""
)