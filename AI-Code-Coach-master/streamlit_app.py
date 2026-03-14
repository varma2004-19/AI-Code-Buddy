import streamlit as st
from logic import process_query
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Code Coach", page_icon="🤖", layout="wide")

st.title("🤖 AI Code Coach")
st.markdown("RAG-powered assistant for debugging, translating, and explaining your code.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    task_type = st.radio(
        "Choose Task:",
        ["Debug Code", "Translate Code", "Explain Algorithm"],
        index=0
    )
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mapping task names to choice strings
task_map = {
    "Debug Code": "1",
    "Translate Code": "2",
    "Explain Algorithm": "3"
}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about your code..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            choice = task_map[task_type]
            # We still call the same logic but ignore the fix_info return
            result, docs, _ = process_query(choice, prompt)
            st.markdown(result)
            
            with st.expander("View Context Sources"):
                for doc in docs:
                    st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    st.code(doc.page_content, language='python')

    st.session_state.messages.append({"role": "assistant", "content": result})
