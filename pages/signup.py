import streamlit as st
import re
from config import BASE_URL
from utils import switch_page

# Regex for email validation (simple version)
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Regex for password:
# At least 8 chars, 1 upper, 1 lower, 1 digit, 1 special char (from @$!%*?&)
PASS_REGEX = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
)

def signup():
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

    # 2) Wrap entire page in a form
    with st.form("signup_form"):
        st.header("Sign Up")

        username = st.text_input("Choose a Username", key="signup_username")
        password = st.text_input("Choose a Password", key="signup_password", type="password")
        email = st.text_input("Email", key="signup_email")

        # Two form submit buttons
        signup_submitted = st.form_submit_button("Sign Up")
        back_submitted = st.form_submit_button("Back to Login")

        # 3) Handle logic within the form
        if signup_submitted:
            # Client-side validations
            if not EMAIL_REGEX.match(email):
                st.error("Please enter a valid email address (e.g. user@domain.com).")
            elif not PASS_REGEX.match(password):
                st.error(
                    "Password must be at least 8 characters long, "
                    "and include uppercase, lowercase, a digit, and a special symbol (@$!%*?&)."
                )
            else:
                # If validations pass, proceed with the POST request
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

        if back_submitted:
            switch_page("Login")
