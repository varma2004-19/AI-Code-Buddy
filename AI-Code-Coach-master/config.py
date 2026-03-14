import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CODEBASE_DIR = os.path.join(BASE_DIR, "codebase")
INDEX_DIR = os.path.join(BASE_DIR, "vector_index")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")