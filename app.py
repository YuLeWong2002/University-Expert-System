import streamlit as st

# Import your pages
from pages.login import login
from pages.signup import signup
from pages.loading import loading
from pages.chatbot import chatbot
from pages.profile import profile
import requests

def set_page(page_name: str):
    """Helper to change pages inside this session."""
    st.session_state.page = page_name
    st.rerun()  # Force immediate page reload

# 1) Ensure necessary session variables exist
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    # Default to "Login" if not logged in, else "Chatbot".
    st.session_state.page = "Login"
if "session" not in st.session_state:
    st.session_state.session = requests.Session()

# 2) Hide Streamlit's default UI (optional)
st.markdown("""
    <style>
        [data-testid="stHeader"] {display: none;} /* Hide Streamlit's default header */
        [data-testid="stSidebar"] {display: none;} /* Hide Streamlit sidebar */
        [data-testid="stSidebarCollapsedControl"] {display: none;} /* Hide sidebar collapse button */
        footer {visibility: hidden;} /* Hide Streamlit footer */
    </style>
""", unsafe_allow_html=True)

# 3) Route to the correct page
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
elif page == "Loading":
    loading()
elif page == "Chatbot":
    chatbot()
elif page == "Profile":
    profile()
else:
    st.write("Error: Unknown page")
