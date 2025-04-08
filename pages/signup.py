import streamlit as st
from config import BASE_URL
from utils import switch_page

def signup():

    st.markdown(
        """
        <style>
        /* Target the main app container */
        [data-testid="stAppViewContainer"] {
            background-image: url("https://bdcdei-prod-media.s3.eu-west-1.amazonaws.com/images/UoNottingham.width-1510.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("Sign Up")
    username = st.text_input("Choose a Username", key="signup_username")
    password = st.text_input("Choose a Password", key="signup_password", type="password")
    email = st.text_input("Email", key="signup_email")

    if st.button("Sign Up"):
        payload = {"username": username, "password": password, "email": email}
        try:
            response = st.session_state.session.post(f"{BASE_URL}/signup", json=payload)
            if response.status_code == 201:
                st.success("Signup successful! Please log in.")
                switch_page("Login")
            else:
                st.error(response.json().get("message", "Signup failed."))
        except Exception as e:
            st.error(f"Error: {e}")

