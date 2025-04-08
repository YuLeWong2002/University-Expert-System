import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import subprocess
import streamlit as st
from pydantic import Field
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter
from utils import switch_page
from config import BASE_URL
import PyPDF2

# ---------------- STREAMING Ollama CLI-based LLM ----------------
class StreamingOllamaCLI(LLM):
    model: str = Field(...)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @property
    def _llm_type(self) -> str:
        return "ollama_cli"

    def _call(self, prompt: str, stop=None):
        """
        Non-streaming fallback method if something calls ._call() directly.
        """
        return "".join(list(self.generate_stream(prompt, stop=stop))).strip()

    def generate_stream(self, prompt: str, stop=None):
        """
        Runs Ollama in a subprocess, yielding output line-by-line as it arrives.
        """
        command = ["ollama", "run", self.model, prompt]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read Ollama output line-by-line
        for line in process.stdout:
            # Yield the partial line so you can display it
            yield line

        # Make sure the process has finished
        process.wait()
        _, stderr = process.communicate()
        if stderr:
            print("stderr:", stderr)


# ---------------- Document & Vector Store Building ----------------
def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_content = data.get("content", "")

                if not raw_content:
                    continue

                if isinstance(raw_content, list):
                    content_str = "\n".join(
                        item.get("p", "") if isinstance(item, dict) else str(item)
                        for item in raw_content
                    )
                elif isinstance(raw_content, dict):
                    content_str = str(raw_content)
                else:
                    content_str = str(raw_content)

                if not content_str.strip():
                    continue

                doc = Document(
                    page_content=content_str,
                    metadata={
                        "source": filename,
                        "url": data.get("url", ""),
                        "pdf_files": data.get("pdf_files", []),
                        "keywords": data.get("keywords", [])
                    }
                )
                docs.append(doc)
    return docs

def load_pdfs(folder_path: str) -> list[Document]:
    """
    Loads all PDF files in the given folder, extracting text from each page,
    and returning them as a list of Document objects.
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)

                    # Iterate through all the pages
                    for page_idx in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_idx]
                        page_text = page.extract_text()

                        # Skip empty text
                        if not page_text or not page_text.strip():
                            continue

                        # Create a Document with metadata
                        doc = Document(
                            page_content=page_text,
                            metadata={
                                "source": filename,
                                "pdf_page": page_idx + 1,  # 1-based index
                                # You can include more metadata as needed
                                "file_path": file_path
                            }
                        )
                        docs.append(doc)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return docs

def split_document_into_chunks(doc, chunk_size=1000, chunk_overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(doc.page_content)
    return [
        Document(
            page_content=chunk,
            metadata={**doc.metadata, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

def build_or_load_vector_store(folder_path, pdf_path, faiss_dir="faiss_index"):
    import os
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.isdir(faiss_dir):
        print("Loading FAISS index from disk...")
        # Add the keyword argument allow_dangerous_deserialization=True
        vector_store = FAISS.load_local(
            faiss_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Building FAISS index from scratch...")
        # (A) Load JSON + PDF documents
        json_docs = load_documents(folder_path)
        pdf_docs = load_pdfs(pdf_path)
        all_docs = json_docs + pdf_docs

        # (B) Chunk them
        chunked_docs = []
        for d in all_docs:
            chunked_docs.extend(split_document_into_chunks(d))

        # (C) Create new FAISS vector store
        vector_store = FAISS.from_documents(chunked_docs, embeddings)

        # (D) Save to disk
        vector_store.save_local(faiss_dir)
        print(f"FAISS index saved to {faiss_dir}/")

    return vector_store


def build_chat_prompt_with_context(messages, query, retriever, top_k=3):
    # 1. Get relevant docs from vector store
    docs = retriever.get_relevant_documents(query)[:top_k]
    context_texts = "\n\n".join([doc.page_content for doc in docs])

    # 2. Build chat history string
    chat_history = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            chat_history += f"User: {content}\n"
        elif role == "assistant":
            chat_history += f"Assistant: {content}\n"

    # 3. Final prompt with memory + retrieved context
    final_prompt = f"""You are a helpful assistant. Use the following context to answer the user query.

Context:
{context_texts}

Conversation history:
{chat_history}
User: {query}
Assistant:"""

    return final_prompt

# ---------------- The chatbot() function ----------------
def chatbot():
    if not st.session_state.logged_in:
        st.error("Please log in first.")
        st.session_state.page = "Login"
        st.rerun()

    col1, col2 = st.columns([4,2])
    with col1:
        # Use a smaller heading so it doesn't add huge vertical spacing
        st.markdown("## Chatbot")

    with col2:
        # Add a blank line or two to push the button down to match the heading
        st.write("")
        # Now place the button
        if st.button("Profile"):
            switch_page("Profile")


    # 1) Build vector store once
    # if "vector_store" not in st.session_state:
    #     folder_path = "f13_json"
    #     pdf_path = "pdf"
    #     st.session_state.vector_store = build_or_load_vector_store(folder_path, pdf_path)

    # # 2) Initialize streaming LLM once
    # if "llm" not in st.session_state:
    #     st.session_state.llm = StreamingOllamaCLI(model="deepseek-r1:7b")

    # 3) Initialize chat history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you with the RAG system?"}
        ]

    # 4) Display existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 5) Chat input
    if user_input := st.chat_input("Type your message..."):
        # Add user message to session history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ----- 1) Retrieve documents and build prompt manually -----
        vector_store = st.session_state.vector_store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        final_prompt = build_chat_prompt_with_context(
            messages=st.session_state.messages[:-1],  # exclude current user msg to avoid duplication
            query=user_input,
            retriever=retriever
        )

        # ----- 2) Stream the LLM’s response in real time -----
        with st.chat_message("assistant"):
            partial_placeholder = st.empty()   # a placeholder for partial tokens
            partial_text = ""                  # accumulates partial output

            try:
                # We'll call the new generator method we added
                for token in st.session_state.llm.generate_stream(final_prompt):
                    partial_text += token
                    partial_placeholder.markdown(partial_text)
                
                final_answer = partial_text.strip()
            except Exception as e:
                final_answer = f"Error generating answer: {e}"
                partial_placeholder.markdown(final_answer)

        # ----- 3) Save the final answer in session state -----
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        # ----- 4) Show references (just like before) -----
        with st.expander("Referenced Documents"):
            source_docs = retriever.get_relevant_documents(user_input)
            for doc in source_docs:
                source = doc.metadata.get("source", "Unknown source")
                snippet = doc.page_content[:200]
                pdf_page = doc.metadata.get("pdf_page", None)
                if pdf_page is not None:
                    st.write(f"**Source**: {source} | **Page**: {pdf_page} | Snippet: {snippet}...")
                else:
                    st.write(f"**Source**: {source} | Snippet: {snippet}...")

                page_url = doc.metadata.get("url", "")
                if page_url:
                    st.write(f"**Page**: {page_url}")
                    st.markdown(f"[Visit page]({page_url})")

