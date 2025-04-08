import streamlit as st
from config import BASE_URL

def profile():
    st.header("Profile")

    # Fetch user info from backend
    username_val = ""
    email_val = ""
    try:
        response = st.session_state.session.get(
            f"{BASE_URL}/profile",
            params={"username": st.session_state.username},
        )
        if response.status_code == 200:
            data = response.json()
            username_val = data.get("username", "")
            email_val = data.get("email", "Not provided")
        else:
            st.error(response.json().get("message", "Failed to load profile"))
    except Exception as e:
        st.error(f"Error: {e}")

    # Display user data in a form
    with st.form("profile_form"):
        st.text_input("Username", value=username_val, disabled=True)
        st.text_input("Email", value=email_val, disabled=True)

        col1, col2 = st.columns(2)
        with col1:
            logout_btn = st.form_submit_button("Logout")
        with col2:
            back_btn = st.form_submit_button("Back to Chatbot")

        if logout_btn:
            try:
                response = st.session_state.session.get(f"{BASE_URL}/logout")
                if response.status_code == 200:
                    st.success("Logged out successfully!")
                    st.session_state.logged_in = False
                    st.session_state.username = ""

                    # Switch page to Login
                    st.session_state.page = "Login"
                    st.rerun()
                else:
                    st.error("Logout failed.")
            except Exception as e:
                st.error(f"Error: {e}")

        if back_btn:
            st.session_state.page = "Chatbot"
            st.rerun()
