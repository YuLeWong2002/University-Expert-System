import streamlit as st

def switch_page(page_name):
    """Switch between pages"""
    st.session_state.page = page_name
    st.rerun()
