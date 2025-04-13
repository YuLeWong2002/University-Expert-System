import streamlit as st
import requests
import time
from pages.login import login
from pages.signup import signup
from pages.chatbot import chatbot
from pages.profile import profile
from pages.change_password import change_password
from pages.chatbot import StreamingOllamaCLI, build_or_load_vector_store

def set_page(page_name: str):
    """Helper to change pages inside this session."""
    st.session_state.page = page_name
    st.rerun()  # Force immediate page reload

# 1) Ensure necessary session variables exist
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Login"  # default page
if "session" not in st.session_state:
    st.session_state.session = requests.Session()

# 2) Immediately load the heavy resources if not in session yet
if "vector_store" not in st.session_state:
    folder_path = "f13_json"
    pdf_path = "pdf"
    st.session_state.vector_store = build_or_load_vector_store(folder_path, pdf_path)

if "llm" not in st.session_state:
    time.sleep(3)  # simulate heavy load
    st.session_state.llm = StreamingOllamaCLI(model="deepseek-r1:7b")

# 3) Hide Streamlit's default UI
st.markdown("""
    <style>
        [data-testid="stHeader"] {display: none;} /* Hide Streamlit's default header */
        [data-testid="stSidebar"] {display: none;} /* Hide Streamlit sidebar */
        [data-testid="stSidebarCollapsedControl"] {display: none;} /* Hide sidebar collapse button */
        footer {visibility: hidden;} /* Hide Streamlit footer */
    </style>
""", unsafe_allow_html=True)

# 4) Route to the correct page
page = st.session_state.page

# If user tries to access a page but they're not logged in, override
# and set them to the login page (except for signup).
if not st.session_state.logged_in and page not in ["Login", "Signup"]:
    st.session_state.page = "Login"
    st.rerun()

if page == "Login":
    login()
elif page == "Signup":
    signup()
elif page == "Chatbot":
    chatbot()
elif page == "Profile":
    profile()
elif page == "ChangePassword":
    change_password()
else:
    st.write("Error: Unknown page")
