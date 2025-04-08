import streamlit as st
from config import BASE_URL

def login():
    # Insert the CSS for the background image
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

    st.header("Login")

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", key="login_password", type="password")

    if st.button("Login"):
        payload = {"username": username, "password": password}
        try:
            response = st.session_state.session.post(f"{BASE_URL}/login", json=payload)
            if response.status_code == 200:
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.username = username
                # After successful login, go to Loading page to do heavy tasks
                st.session_state.page = "Loading"
                st.rerun()
            else:
                st.error(response.json().get("message", "Invalid credentials"))
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Sign Up"):
        st.session_state.page = "Signup"
        st.rerun()
