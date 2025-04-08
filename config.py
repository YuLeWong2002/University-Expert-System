import streamlit as st
import requests

# Create a persistent requests session if it doesn't exist
if "session" not in st.session_state:
    st.session_state.session = requests.Session()
    
# Base URL for Flask backend
BASE_URL = "http://127.0.0.1:5000"

# Initialize session state variables
if "session" not in st.session_state:
    st.session_state.session = requests.Session()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "Login"  # Default page
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you?"}]
