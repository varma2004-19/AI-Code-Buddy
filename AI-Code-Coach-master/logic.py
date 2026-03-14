from retriever import get_retriever
from llm import get_llm
from prompts import DEBUG_PROMPT, TRANSLATE_PROMPT, EXPLAIN_PROMPT
from langchain.chains import LLMChain

def process_query(choice, query):
    """
    Processes a code-related query using RAG.
    choice: "1" (Debug), "2" (Translate), "3" (Explain)
    query: The user's question or problem description.
    """
    retriever = get_retriever()
    llm = get_llm()

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)
    
    # Format context with metadata (filename)
    context_parts = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        context_parts.append(f"--- File: {source} ---\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)

    # Select prompt based on choice
    if choice == "1":
        prompt = DEBUG_PROMPT
    elif choice == "2":
        prompt = TRANSLATE_PROMPT
    else:
        prompt = EXPLAIN_PROMPT

    # Run the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(context=context, query=query)
    
    # Check for suggested fixes
    fix_info = parse_fix(result)
    
    return result, docs, fix_info

def parse_fix(text):
    """
    Parses the LLM output for a suggested file fix.
    Looks for:
    FILE: <path>
    ```python
    content
    ```
    """
    import re
    import os
    from config import BASE_DIR
    
    # Flexible pattern for FILE: followed by a code block
    pattern = r"(?i)FILE:\s*(.*?)\s*\n+```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        path = match.group(1).strip()
        new_content = match.group(2)
        
        # Resolve path if relative
        if not os.path.isabs(path):
            abs_path = os.path.join(BASE_DIR, path)
            if os.path.exists(abs_path):
                path = abs_path
            else:
                # Try relative to codebase
                cb_path = os.path.join(BASE_DIR, "codebase", path)
                if os.path.exists(cb_path):
                    path = cb_path
                    
        return {"file_path": path, "new_content": new_content}
    return None

def apply_fix(file_path, new_content):
    """Writes the new content to the specified file."""
    try:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True, f"Successfully updated {file_path}"
    except Exception as e:
        return False, str(e)
