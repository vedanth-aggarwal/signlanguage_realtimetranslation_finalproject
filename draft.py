import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import time
import os
import openai
from dotenv import load_dotenv

def main():

    # All design and structural elements occupy wider area 
    st.set_page_config('wide')

    # Create sidebar - On button click call function corresponding to that page
    st.sidebar.title("Navigation Bar")
    pages = {
        "Home": homepage,
        "My Chatbot": chatbot_page
    }

    selected_page = st.sidebar.button("Home", key="home",use_container_width=True,type='primary')
    if selected_page:
        # URL in the browser's address bar will be updated to include the query parameter 'page=home'
        st.experimental_set_query_params(page="home")
    selected_page = st.sidebar.button("Chatbot", key="chatbot",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="chatbot")

    # Get the page name from the URL, default to "home"
    page_name = st.experimental_get_query_params().get('page', ['home'])[0]

    # Call the corresponding page based on the selected page_name
    if page_name == "home":
        homepage()
    elif page_name == "chatbot":
        chatbot_page()

def homepage():
    st.title('HE')

def chatbot_page():
    st.title('SHE')

if __name__ == "__main__":
    main()