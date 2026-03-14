# AI Code Coach

AI Code Coach is a Retrieval-Augmented Generation (RAG) based tool designed to help developers debug, translate, and understand their codebase. This release introduces **Semantic Intelligence** for a more powerful developer experience.

## 🚀 Features

- **Interactive Web UI**: Modern chat-based interface built with Streamlit.
- **Semantic Intelligence 🧠**: Uses syntax-aware splitting (Python AST) to ensure code chunks are logically complete.
- **Debug & Explain**: Get clear, context-aware explanations and root-cause analysis.
- **Local Indexing**: Rapidly searches your codebase using FAISS and HuggingFace embeddings.
- **Advanced Context**: Automatically includes file metadata and structured headers for better LLM performance.

## 🛠️ Prerequisites

- Python 3.10+
- A Groq API Key (get one at [console.groq.com](https://console.groq.com/))

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Druva4444/AI-Code-Coach.git
   cd AI-Code-Coach
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   ```

## 📖 Usage

### 1. Index your Codebase
Place the code you want to analyze in the `codebase/` directory. Then, run the ingestion script to build the semantic vector index:
```bash
python3 ingest.py
```

### 2. Run the Web Interface (Recommended)
Start the interactive Streamlit application:
```bash
streamlit run streamlit_app.py
```
This will open a browser window where you can choose a task and chat with the AI about your code.

### 3. Run the CLI Version (Optional)
Alternatively, you can use the command-line tool:
```bash
python3 app.py
```

## 📁 Project Structure

- `streamlit_app.py`: Main entry point for the Web application.
- `app.py`: CLI entry point for the application.
- `logic.py`: Shared core logic for query processing and RAG execution.
- `ingest.py`: Script to build the semantic FAISS index using Python-specific splitting.
- `retriever.py`: Handles loading the vector index and retrieving relevant code snippets.
- `llm.py`: Configuration for the Groq LLM.
- `prompts.py`: Task prompt templates for debugging, translation, and explanations.
- `codebase/`: Put the source files you want the AI to "read" here.
- `vector_index/`: Stores the generated FAISS index files.

## 🧰 Technologies Used

- [LangChain](https://github.com/langchain-ai/langchain): Framework for building LLM applications.
- [Streamlit](https://streamlit.io/): Framework for building interactive web apps.
- [FAISS](https://github.com/facebookresearch/faiss): Efficient similarity search and clustering of dense vectors.
- [Groq](https://groq.com/): High-performance LLM inference.
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): For generating code vector representations.
