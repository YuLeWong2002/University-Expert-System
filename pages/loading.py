import streamlit as st
import time
from pages.chatbot import StreamingOllamaCLI, build_or_load_vector_store

def loading():
    # If user not logged in, return to Login
    if not st.session_state.logged_in:
        st.error("You must log in first.")
        st.session_state.page = "Login"
        st.rerun()

    # 1) Add CSS to set the entire page's background image,
    #    plus a semi-transparent overlay so the text is readable.
    st.markdown(
        """
        <style>
        /* Set the full-page background image */
        [data-testid="stAppViewContainer"] {
            background-image: url("https://bdcdei-prod-media.s3.eu-west-1.amazonaws.com/images/UoNottingham.width-1510.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }

        /* The loading overlay covers the page, but we make it semi-transparent. */
        .loading-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;

            /* Instead of solid white, use a semi-transparent layer so the background is visible. */
            background-color: rgba(255, 255, 255, 0.7);
        }

        .main-content {
            visibility: hidden;
        }
        </style>
        <div class="loading-overlay">
            <h1 style="font-size: 2em; color: #000;">Loading, please wait...</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2) Build or load resources
    if "vector_store" not in st.session_state:
        folder_path = "f13_json"
        pdf_path = "pdf"
        st.session_state.vector_store = build_or_load_vector_store(folder_path, pdf_path)

    if "llm" not in st.session_state:
        # Simulate heavy load
        time.sleep(3)
        st.session_state.llm = StreamingOllamaCLI(model="deepseek-r1:7b")

    # 3) Remove overlay and reveal main content
    st.markdown(
        """
        <script>
        const overlay = window.parent.document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.parentNode.removeChild(overlay);
        }
        const mainBlock = window.parent.document.querySelector('.main-content');
        if (mainBlock) {
            mainBlock.style.visibility = 'visible';
        }
        </script>
        """,
        unsafe_allow_html=True
    )

    # 4) Done, move on to Chatbot
    st.success("All set! Redirecting to Chatbot...")
    st.session_state.page = "Chatbot"
    st.rerun()
