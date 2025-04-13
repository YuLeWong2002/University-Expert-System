import streamlit as st
from config import BASE_URL

def login():
    # 1) Insert CSS for the full-page background image
    st.markdown(
        """
        <style>
        /* Make the entire page background an image */
        [data-testid="stAppViewContainer"] {
            background-image: url("https://bdcdei-prod-media.s3.eu-west-1.amazonaws.com/images/UoNottingham.width-1510.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }

        /* Style the form container specifically:
           target the st.form container by data-testid, 
           then the direct child 'div' which holds the form content */
        div[data-testid="stForm"] > div {
            background-color: white;
            color: black;
            padding: 2rem;
            margin: 2rem auto;
            border-radius: 0.5rem;
            max-width: 400px;  /* limit width if you want */
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
        }

        /* Text input fields: light gray background, rounded borders, black caret */
        input[type="text"], input[type="password"], textarea {
            background-color: #f0f2f6 !important;
            color: #000 !important;
            border: 1px solid #e4e7ea !important;
            border-radius: 0.375rem !important;
            padding: 0.5rem !important;
            caret-color: black !important; /* <-- Black typing cursor */
        }

        /* "Eye" icon (the show/hide password icon).
           This may depend on your Streamlit version; 
           [data-baseweb="input"] svg covers the typical case. */
        [data-baseweb="input"] svg {
            color: #6c757d !important;
            border: 0px;
            background-color: transparent !important;
            align-item: center;
        }

        /* Buttons: white background, black border/text, slight rounding */
        [data-testid="stFormSubmitButton"] > button {
            background-color: #fff !important;
            color: #000 !important;
            border: 1px solid #000 !important;
            border-radius: 0.375rem !important;
            padding: 0.5rem 1rem !important;
            margin-right: 0.5rem !important;
            cursor: pointer !important;
        }

        /* Hover state for buttons */
        [data-testid="stFormSubmitButton"] > button:hover {
            background-color: #e6e9ef !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

    # 2) Create the form (header included inside)
    with st.form("login_form"):
        st.header("Login")

        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", key="login_password", type="password")

        # Two form submit buttons
        login_submitted = st.form_submit_button("Login")
        signup_submitted = st.form_submit_button("Sign Up")

        # 3) Handle logic within the form
        if login_submitted:
            payload = {"username": username, "password": password}
            try:
                response = st.session_state.session.post(f"{BASE_URL}/login", json=payload)
                if response.status_code == 200:
                    st.success("Login successful!")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    # Clear or remove the old messages:
                    if "messages" in st.session_state:
                        del st.session_state["messages"]
                    # After successful login, go to Loading page
                    st.session_state.page = "Chatbot"
                    st.rerun()
                else:
                    st.error(response.json().get("message", "Invalid credentials"))
            except Exception as e:
                st.error(f"Error: {e}")

        if signup_submitted:
            st.session_state.page = "Signup"
            st.rerun()
