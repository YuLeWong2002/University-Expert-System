import streamlit as st
from config import BASE_URL

def change_password():
    # 1) If the user isn’t logged in, send them to Login
    if not st.session_state.get("logged_in"):
        st.error("Please log in first.")
        st.session_state.page = "Login"
        st.rerun()

    st.title("Change Password")

    # 2) Create a form for changing password
    with st.form("change_password_form"):
        old_pw = st.text_input("Old Password", type="password")
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm New Password", type="password")

        submit_btn = st.form_submit_button("Update Password")
        cancel_btn = st.form_submit_button("Cancel")

        if submit_btn:
            # Basic client-side checks
            if not old_pw or not new_pw or not confirm_pw:
                st.error("All fields are required.")
            elif new_pw != confirm_pw:
                st.error("New password fields do not match.")
            else:
                # 3) Submit a PUT request to /update_password
                payload = {
                    "username": st.session_state.get("username"),
                    "old_password": old_pw,
                    "new_password": new_pw
                }

                try:
                    response = st.session_state.session.put(
                        f"{BASE_URL}/update_password",
                        json=payload
                    )
                    if response.status_code == 200:
                        st.success("Password updated successfully!")
                        # Direct back to Profile or Chatbot
                        st.session_state.page = "Profile"
                        st.rerun()
                    else:
                        # Show any error message from server
                        res_data = response.json()
                        st.error(res_data.get("message", "Failed to update password."))
                except Exception as e:
                    st.error(f"Error updating password: {e}")

        if cancel_btn:
            # Return user to some page, e.g. "Profile"
            st.session_state.page = "Profile"
            st.rerun()
