import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import subprocess
import streamlit as st
from pydantic import Field
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.llms.base import LLM
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Optional
from symspellpy import SymSpell, Verbosity
from tavily_config import TavilyClient
from utils import switch_page
from config import BASE_URL
import re

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

# Read pre‑chunked PDF text from a JSONL file
def load_jsonl_chunks(jsonl_path: str) -> list[Document]:
    """
    Each line in jsonl must contain:
    {
      "id": "...",
      "source": "...pdf",
      "page": 3,
      "chunk_index": 17,
      "text": "clean chunk text..."
    }
    """
    docs = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj["text"],
                    metadata={
                        "source": obj["source"],
                        "pdf_page": obj.get("page"),
                        "chunk_index": obj.get("chunk_index"),
                        # keep url key for UI consistency (may be empty)
                        "url": obj.get("url", "")
                    }
                )
            )
    return docs

def build_or_load_vector_store(
        json_folder: str,
        pdf_jsonl: str,
        faiss_dir="faiss_index"
):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.embeddings = embeddings

    if os.path.isdir(faiss_dir):
        print("Loading FAISS index from disk...")
        dense_store = FAISS.load_local(
            faiss_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Building FAISS index from scratch...")
        json_docs = load_documents(json_folder)
        pdf_docs = load_jsonl_chunks(pdf_jsonl)
        all_docs = json_docs + pdf_docs

        chunked_docs = []
        for d in all_docs:
            if "chunk_index" in d.metadata:
                chunked_docs.append(d)
            else:
                chunked_docs.extend(split_document_into_chunks(d))

        dense_store = FAISS.from_documents(chunked_docs, embeddings)
        dense_store.save_local(faiss_dir)
        print(f"FAISS index saved to {faiss_dir}/")

    # --- BM25Retriever: build from same documents used in FAISS ---
    all_docs = list(dense_store.docstore._dict.values())
    sparse_retriever = BM25Retriever.from_documents(all_docs)
    sparse_retriever.k = 5  # Top-K sparse docs

    # Store both retrievers in session state
    st.session_state.vector_store = dense_store  # for backward compatibility
    st.session_state.dense_store = dense_store
    st.session_state.sparse_retriever = sparse_retriever

    return dense_store  # still return for legacy calls

def hybrid_retrieve(query: str, k_dense=5, k_sparse=5, top_k_final=5):
    dense_store = st.session_state.dense_store
    sparse_retriever = st.session_state.sparse_retriever

    # Dense results
    dense_results = dense_store.similarity_search_with_score(query, k=k_dense)
    dense_docs = [(doc, score, "dense") for doc, score in dense_results]

    # Sparse results (BM25Retriever returns Documents, not scores)
    sparse_docs = [(doc, None, "sparse") for doc in sparse_retriever.get_relevant_documents(query)]

    # Combine and deduplicate (based on source + chunk_index)
    seen = set()
    merged = []

    for doc, score, source in dense_docs + sparse_docs:
        doc_id = (doc.metadata.get("source"), doc.metadata.get("chunk_index"))
        if doc_id not in seen:
            merged.append((doc, score, source))
            seen.add(doc_id)

    # Sort by dense score if available
    merged_sorted = sorted(merged, key=lambda x: x[1] if x[1] is not None else float('inf'))

    return merged_sorted[:top_k_final]

# Load once globally (e.g., at the top of your script)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query: str, docs: List[Tuple[Document, Optional[float], str]], top_k=5):
    """
    Reranks documents using a CrossEncoder based on query relevance.
    Input: List of (Document, score, source)
    Output: List of (Document, rerank_score, source)
    """
    pairs = [(query, doc.page_content) for doc, _, _ in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(
        zip([doc for doc, _, src in docs], scores, [src for _, _, src in docs]),
        key=lambda x: x[1],
        reverse=True
    )
    return reranked[:top_k]

def build_chat_prompt_with_context(messages, query, retriever, top_k=5):
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

# Resource / DB Helpers
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

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"  # Adjust if needed
if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
    print("Dictionary file not found!")

def correct_query(query: str) -> str:
    """
    Use SymSpell to correct spelling in the user query.
    Returns corrected text if any corrections are found,
    otherwise returns original query.
    """
    suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
    if suggestions:
        corrected = suggestions[0].term
        # Optional: Print corrections for debugging
        if corrected.lower() != query.lower():
            print(f"Corrected query: '{query}' -> '{corrected}'")
        return corrected
    return query

# Synonym table & query‑expander 
SYNONYM_MAP = {
    # academics
    "exchange programme": ["study abroad", "mobility program", "student exchange", "international study"],
    "assessment"        : ["exam", "test", "evaluation"],
    "lecture"           : ["class session", "teaching session"],
    "module"            : ["course", "subject"],

    # campus life
    "alcohol"           : ["liquor", "drink", "beer", "wine", "alcoholic beverages"],
    "accommodation"     : ["housing", "dormitory", "residence hall"],
    "canteen"           : ["cafeteria", "food court", "dining hall"],
    "library"           : ["learning resource centre"],

    # admin / travel
    "visa"              : ["student pass", "immigration clearance", "entry permit"],
    "registration"      : ["enrolment", "enrollment"],
    "scholarship"       : ["bursary", "financial aid"]
}

def expand_query_with_synonyms(query: str) -> list[str]:
    """Return [original, synonym‑substituted…] – all lower‑cased."""
    expanded = {query.lower()}
    for key, syns in SYNONYM_MAP.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, query, flags=re.I):
            for s in syns:
                expanded.add(re.sub(pattern, s, query, flags=re.I))
    return list(expanded)

def expanded_similarity_search(vdb: FAISS, query: str, k_per=3) -> list[tuple[Document,float]]:
    """
    Run similarity_search_with_score on the original + synonym variants,
    merge by keeping the best (lowest) distance per document.
    Returns a list sorted by distance.
    """
    variants = expand_query_with_synonyms(query)
    best: dict[str, tuple[Document,float]] = {}

    for q in variants:
        for doc, score in vdb.similarity_search_with_score(q, k=k_per):
            doc_id = f"{doc.metadata.get('source')}|{doc.metadata.get('chunk_index')}"
            if doc_id not in best or score < best[doc_id][1]:
                best[doc_id] = (doc, score)

    # sort by score (lower = closer) and return
    return sorted(best.values(), key=lambda pair: pair[1])

tavily_client = TavilyClient(api_key="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")

def fetch_tavily_summary(query: str) -> str:
    try:
        # 'search_depth' can be "basic", "medium", or "advanced"
        result = tavily_client.search(
            query=query,
            search_depth="basic",
            include_answer=True
        )
        print("\n=== Tavily Search Result ===")
        ai_answer = result.get("answer", "No AI-generated answer available.")
        print(f"AI-Generated Answer:\n{ai_answer}")

        links = result.get("links", [])
        print("\nLinks:")
        for i, link in enumerate(links):
            print(f"{i+1}. URL: {link['url']}")
            print(f"   Title: {link['title']}")
            print(f"   Snippet: {link['snippet']}\n")

        # Return the AI answer or something for the fallback
        return ai_answer

    except Exception as e:
        # Return a string describing the error
        err_msg = f"Error using Tavily: {e}"
        print(err_msg)
        return err_msg
    

# Embed a text string once

def embed(text: str):
    emb = st.session_state.embeddings                 # <-- get the shared object
    return st.session_state._embed_cache.setdefault(
        text,
        emb.embed_query(text)
    )

# lazy init of a per‑session cache
if "_embed_cache" not in st.session_state:
    st.session_state._embed_cache = {}

# Pick relevant turns from history
def build_relevant_chat_history(query: str,
                                messages: list[dict],
                                max_pairs: int = 6,
                                sim_thresh: float = 0.7) -> str:
    """
    Return a chat‑history string containing at most `max_pairs`
    user/assistant turns that are semantically similar to `query`.
    """
    query_vec = embed(query)
    scored = []

    # walk messages in reverse (newest first) and score
    for m in reversed(messages):
        if m["role"] not in ("user", "assistant"):
            continue
        vec   = embed(m["content"])
        # cosine similarity for unit‑norm vectors = dot product
        sim   = sum(q * v for q, v in zip(query_vec, vec))
        scored.append((sim, m))

    # keep top N turns above threshold
    picked = [m for sim, m in sorted(scored, reverse=True) if sim > sim_thresh][:max_pairs]

    # restore chronological order
    picked = list(reversed(picked))

    # build the text block
    chat_hist = ""
    for m in picked:
        role = "User" if m["role"] == "user" else "Assistant"
        chat_hist += f"{role}: {m['content']}\n"
    return chat_hist


# ---------------- The chatbot() function ----------------
def chatbot():
    # ---------- 0. Guard: user must be logged in ----------
    if not st.session_state.logged_in:
        st.error("Please log in first.")
        st.session_state.page = "Login"
        st.rerun()

    print("Logged in:", st.session_state.logged_in, "as", st.session_state.username)

    # ---------- header bar --------------------------------------------------
    col_title, col_profile, col_clear = st.columns([6, 1.2, 1.8])

    with col_title:
        st.markdown("## Chatbot")          # title keeps its native spacing

    # amount of padding that centres buttons (~ 0.85 em works for the default theme)
    PAD = "<div style='padding-top:0.85em'></div>"

    with col_profile:
        st.markdown(PAD, unsafe_allow_html=True)      # <‑‑ add vertical offset
        if st.button("Profile"):
            switch_page("Profile")

    with col_clear:
        st.markdown(PAD, unsafe_allow_html=True)      # <‑‑ same offset
        if st.button("Clear Chat", key="clr"):
            resp = st.session_state.session.delete(
                f"{BASE_URL}/clear_chat_records",
                params={"username": st.session_state.username},
            )
            if resp.status_code == 200:
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": "Your chat history has been cleared."
                }]
                st.success("Chat history cleared.")
                st.rerun()
            else:
                st.error("Failed to clear history: " + resp.text)

    # ---------- 2. Load conversation from DB (if empty) ----------
    if "messages" not in st.session_state:
        loaded_history = fetch_chat_history_from_db()
        if loaded_history:
            st.session_state.messages = loaded_history
        else:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! How can I help you with the RAG system?"}
            ]

    # ---------- 3. Render existing conversation ----------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("refs"):
                with st.expander("Referenced Documents"):
                    for ref in msg["refs"]:
                        src   = ref.get("source", "Unknown source")
                        snip  = ref.get("snippet", "")
                        page  = ref.get("page", None)
                        url   = ref.get("url", "")
                        if page is not None:
                            st.write(f"**Source**: {src} | **Page**: {page} | Snippet: {snip}...")
                        else:
                            st.write(f"**Source**: {src} | Snippet: {snip}...")
                        if url:
                            st.markdown(f"[Visit page]({url})")

    # ---------- 4. Handle new user input ----------
    if user_input := st.chat_input("Type your message..."):
        # 4‑A. Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 4‑B. Spell‑correct
        corrected_input = correct_query(user_input)

        sparse_retriever = st.session_state.sparse_retriever
        bm25_docs = sparse_retriever.get_relevant_documents(corrected_input)

        print("\n[Top 5 BM25 (Sparse) Retrieval Results]")
        for i, doc in enumerate(bm25_docs[:5]):
            short_snip = doc.page_content[:120].replace("\n", " ")
            source = doc.metadata.get("source", "?")
            print(f"[{i+1}] Source: {source} | Snippet: {short_snip}...")
        print("[End BM25]\n")

        # ---------- 4‑C. Hybrid similarity search (BM25 + FAISS) ----------
        retrieved_docs = hybrid_retrieve(corrected_input, k_dense=5, k_sparse=5, top_k_final=5)

        print("\n[Hybrid Retrieval Results]")
        for i, (doc, score, method) in enumerate(retrieved_docs):
            short_snip = doc.page_content[:120].replace("\n", " ")
            print(f"[{i+1}] {method.upper()} | Score: {score if score is not None else 'N/A'} | Source: {doc.metadata.get('source','?')} | Snippet: {short_snip}...")
        print("[End]\n")

        # ---------- 4‑D. Apply reranking ----------
        retrieved_docs = rerank_documents(corrected_input, retrieved_docs, top_k=5)

        print("\n[CrossEncoder Reranker Results]")
        for i, (doc, score, method) in enumerate(retrieved_docs):
            short_snip = doc.page_content[:120].replace("\n", " ")
            print(f"[{i+1}] Score: {score:.4f} | Source: {doc.metadata.get('source','?')} | Method: {method.upper()} | Snippet: {short_snip}...")
        print("[End Reranker Results]\n")

        # ---------- 5. Filter good documents using reranker score ----------
        RERANK_THRESH = 1.0  # Threshold: only keep passages with meaningful match

        good_pairs = []
        for doc, score, method in retrieved_docs:
            if score >= RERANK_THRESH:
                good_pairs.append((doc, score))

        # ---------- 6. Decide: fallback or build local prompt ----------
        if not good_pairs:
            # ---- 6‑A. No good reranked docs → fallback to Tavily ----
            tavily_answer = fetch_tavily_summary(corrected_input)
            final_prompt = f"""You are a helpful AI. Use the following info to answer the user query.

Tavily Web Search Summary:
{tavily_answer}

User Query: {user_input}

Answer:"""
            ref_list = [{
                "source": "Tavily",
                "snippet": tavily_answer[:200] if isinstance(tavily_answer, str) else "",
                "page": None,
                "url": ""
            }]
        else:
            
            MAX_CHUNKS = 3  # Adjust based on your model’s max token limit
            context_text = "\n\n".join([d.page_content for d, _ in good_pairs[:MAX_CHUNKS]])
            print("\n=== Final context sent to LLM ===")
            print(context_text)
            print("=== End of context ===\n")

            print(f"[Using Top {min(len(good_pairs), MAX_CHUNKS)} Chunks in Final Prompt]")

            # build chat history  –  keep only relevant prior turns
            chat_hist = build_relevant_chat_history(
                query      = user_input,
                messages   = st.session_state.messages[:-1],   # exclude current user turn
                max_pairs  = 1,    # <- tweak if you want more/less context
                sim_thresh = 0.65  # <- tweak similarity cut‑off (0‑1)
            )


            final_prompt = f"""You are a helpful assistant. Use the following context to answer the user query.

Context:
{context_text}

Conversation history:
{chat_hist}
User: {user_input}
Assistant:"""

            # Build reference list from good_pairs
            ref_list = []
            for d, s in good_pairs:
                ref_list.append({
                    "source" : d.metadata.get("source", "Unknown source"),
                    "snippet": d.page_content[:200],
                    "page"   : d.metadata.get("pdf_page", None),
                    "url"    : d.metadata.get("url", "")
                })

        # ---------- 7. Generate assistant response ----------
        with st.chat_message("assistant"):
            placeholder = st.empty()
            acc = ""
            try:
                for tok in st.session_state.llm.generate_stream(final_prompt):
                    acc += tok
                    placeholder.markdown(acc)
                final_answer = acc.strip()
            except Exception as e:
                final_answer = f"Error generating answer: {e}"
                placeholder.markdown(final_answer)

        # ---------- 8. Save assistant turn ----------
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "refs": ref_list           # store refs for future display
        })

        # ---------- 9. Show current references ----------
        with st.expander("Referenced Documents"):
            for ref in ref_list:
                src  = ref.get("source", "Unknown source")
                snip = ref.get("snippet", "")
                page = ref.get("page", None)
                url  = ref.get("url", "")
                if page is not None:
                    st.write(f"**Source**: {src} | **Page**: {page} | Snippet: {snip}...")
                else:
                    st.write(f"**Source**: {src} | Snippet: {snip}...")
                if url:
                    st.markdown(f"[Visit page]({url})")

        # ---------- 10. Persist to DB ----------
        store_chat_record_in_db(
            user_input,
            final_answer,
            json.dumps(ref_list)
        )
