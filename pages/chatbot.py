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
    docs = retriever.get_relevant_documents(query)[:top_k]
    context_texts = "\n\n".join([doc.page_content for doc in docs])

    # Build the conversation
    chat_history = ""
    for msg in messages:
        if msg["role"] == "user":
            chat_history += f"User: {msg['content']}\n"
        else:
            chat_history += f"Assistant: {msg['content']}\n"

    final_prompt = f"""You are a helpful assistant. Use the following context to answer the user query.

Context:
{context_texts}

Conversation history:
{chat_history}
User: {query}
Assistant:"""
    return final_prompt

# ---------------- Resource / DB Helpers ----------------
def fetch_chat_history_from_db():
    """Fetch all chat records for the current user from the server DB."""
    resp = st.session_state.session.get(
        f"{BASE_URL}/get_chat_records",
        params={"username": st.session_state.username}
    )
    print("Successfully fetch chat history from database")
    if resp.status_code == 200:
        chat_data = resp.json()  # list of {user_message, assistant_response, reference_docs}
        messages = []
        for record in chat_data:
            # Parse references from JSON
            refs = []
            if record["reference_docs"]:
                try:
                    refs = json.loads(record["reference_docs"])
                except:
                    refs = []
            # user turn
            messages.append({"role": "user", "content": record["user_message"]})
            # assistant turn
            messages.append({
                "role": "assistant",
                "content": record["assistant_response"],
                "refs": refs
            })
        return messages
    else:
        st.error("Failed to load chat history: " + resp.text)
        return []
    
def store_chat_record_in_db(user_input, assistant_answer, reference_docs=None):
    """Store the user->assistant turn in DB via /store_chat_record."""
    payload = {
        "username": st.session_state.username,
        "user_message": user_input,
        "assistant_response": assistant_answer,
        "reference_docs": reference_docs
    }
    resp = st.session_state.session.post(
        f"{BASE_URL}/store_chat_record",
        json=payload
    )
    if resp.status_code != 201:
        st.error("Failed to store chat record: " + resp.text)

# ---------------- The chatbot() function ----------------
def chatbot():
    if not st.session_state.logged_in:
        st.error("Please log in first.")
        st.session_state.page = "Login"
        st.rerun()
    # log debug message
    print(st.session_state.logged_in)
    print(st.session_state.username)

    col1, col2 = st.columns([4, 2])
    with col1:
        st.markdown("## Chatbot")
    with col2:
        st.write("")
        if st.button("Profile"):
            switch_page("Profile")

    # 1) Load from DB if no local messages
    if "messages" not in st.session_state:
        print("DEBUG: messages is not in session_state, fetching from DB")
        loaded_history = fetch_chat_history_from_db()
        if loaded_history:
            st.session_state.messages = loaded_history
        else:
            print("DEBUG: messages is ALREADY in session_state, skipping fetch")
            # No existing DB history => start with a greeting
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! How can I help you with the RAG system?"}
            ]

    # 2) Display the conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If assistant references exist, expand them
            if msg["role"] == "assistant" and msg.get("refs"):
                with st.expander("Referenced Documents"):
                    for ref_item in msg["refs"]:
                        source = ref_item.get("source", "Unknown source")
                        snippet = ref_item.get("snippet", "")
                        page = ref_item.get("page", None)
                        if page is not None:
                            st.write(f"**Source**: {source} | **Page**: {page} | Snippet: {snippet}...")
                        else:
                            st.write(f"**Source**: {source} | Snippet: {snippet}...")

    # 3) Chat input
    if user_input := st.chat_input("Type your message..."):
        # Add user turn
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Create final prompt & retrieve docs
        vector_store = st.session_state.vector_store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        final_prompt = build_chat_prompt_with_context(
            messages=st.session_state.messages[:-1],
            query=user_input,
            retriever=retriever
        )

        # 4) Generate assistant response
        with st.chat_message("assistant"):
            partial_placeholder = st.empty()
            partial_text = ""
            try:
                for token in st.session_state.llm.generate_stream(final_prompt):
                    partial_text += token
                    partial_placeholder.markdown(partial_text)
                final_answer = partial_text.strip()
            except Exception as e:
                final_answer = f"Error generating answer: {e}"
                partial_placeholder.markdown(final_answer)

        # Save assistant turn locally
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        # Gather references
        source_docs = retriever.get_relevant_documents(user_input)
        ref_list = []
        with st.expander("Referenced Documents"):
            for doc in source_docs:
                source = doc.metadata.get("source", "Unknown source")
                snippet = doc.page_content[:200]
                pdf_page = doc.metadata.get("pdf_page", None)
                snippet_info = {
                    "source": source,
                    "snippet": snippet,
                    "page": pdf_page
                }
                ref_list.append(snippet_info)

                if pdf_page is not None:
                    st.write(f"**Source**: {source} | **Page**: {pdf_page} | Snippet: {snippet}...")
                else:
                    st.write(f"**Source**: {source} | Snippet: {snippet}...")
                page_url = doc.metadata.get("url", "")
                if page_url:
                    st.write(f"**Page**: {page_url}")
                    st.markdown(f"[Visit page]({page_url})")

        # 5) Store new turn in DB
        reference_docs_json = json.dumps(ref_list)
        store_chat_record_in_db(
            user_input,
            final_answer,
            reference_docs_json
        )