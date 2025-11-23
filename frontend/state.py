import streamlit as st


def init():
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "ratings_buffer" not in st.session_state:
        st.session_state.ratings_buffer = []