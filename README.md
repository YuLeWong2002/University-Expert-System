# 🎓 University Expert Chatbot (Streamlit + Flask + RAG)

This project implements a **full-stack university chatbot system** using **Streamlit** (frontend) and **Flask** (backend), integrated with a **Retrieval-Augmented Generation (RAG)** model and **user authentication**.

---

## 📂 Project Structure

### **Streamlit Frontend (Client Interface)**
| File | Description |
|------|-------------|
| **`app.py`** | Main controller and router for the Streamlit app. Manages session state and routes to Login, Signup, Chatbot, Profile, and Change Password. Loads embeddings, vector store, and the Ollama-based LLM. |
| **`login.py`** | Login form. Sends credentials to `/login` API. Redirects to chatbot upon success. |
| **`signup.py`** | Registration form with email and password validation. Sends POST to `/signup`. Redirects to login on success. |
| **`chatbot.py`** | Main chat interface. Features: hybrid retrieval (BM25 + dense), CrossEncoder reranking, prompt construction, spell correction, synonym expansion, saving chat history, displaying referenced docs. |
| **`profile.py`** | Displays username and email. Options for logout, returning to chatbot, and changing password. |
| **`change_password.py`** | Allows password update with old password validation. Connects to `/update_password` API. |

---

### **Configuration & Utilities**
| File | Description |
|------|-------------|
| **`config.py`** | Initializes persistent session and default Streamlit state. Stores `BASE_URL` for backend API. Maintains session variables (`logged_in`, `username`, etc.). |
| **`tavily_config.py`** | Integrates **Tavily API** for fallback web search. Retrieves AI-assisted summaries from Google results when local retrieval fails. |
| **`utils.py`** | Navigation helpers for switching pages cleanly within the Streamlit app. |

---

### **Flask Backend (Server / API)**
| File | Description |
|------|-------------|
| **`backend.py`** | Flask server for authentication and chat storage APIs. Uses **MySQL + SQLAlchemy ORM**. Provides: |
|  | • `/signup` – Register user |
|  | • `/login` – Login with password hashing |
|  | • `/profile` – Retrieve user profile |
|  | • `/logout` – Logout user session |
|  | • `/update_password` – Change password |
|  | • `/store_chat_record` – Save conversation turns |
|  | • `/get_chat_records` – Retrieve chat history |
|  | • `/clear_chat_records` – Delete chat history |

---

## 🔒 Security Notes
- Session-based authentication with `flask_login`
- Password hashing via `werkzeug.security`
- Hybrid retrieval strategy with **Synonyms + Spell Correction**
- Ollama LLM runs locally with streaming responses

> ⚠ **Important:**  
> Ollama must be installed and running locally.  
> Download: [https://ollama.com/](https://ollama.com/)  
> Example model: `deepseek-coder`

---

## ⚙ Setup Instructions

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
