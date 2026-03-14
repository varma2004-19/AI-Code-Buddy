from langchain_groq import ChatGroq
from config import GROQ_API_KEY
MODEL = "llama-3.1-8b-instant"
def get_llm():
    return ChatGroq(
        model=MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
    )