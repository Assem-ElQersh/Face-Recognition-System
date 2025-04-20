"""
Streamlit web application for the Face Detection and Recognition System.
"""

import os
import sys
import streamlit as st
import cv2
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import time
import tempfile
from PIL import Image

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
import utils
from ui.components import (
    sidebar_content, 
    about_page, 
    detection_page, 
    recognition_page, 
    add_face_page, 
    settings_page
)

def initialize_session_state():
    """Initialize session state variables."""
    # Default settings
    if 'detector_method' not in st.session_state:
        st.session_state.detector_method = 'dnn'
    
    if 'recognition_model' not in st.session_state:
        st.session_state.recognition_model = 'hog'
    
    if 'recognition_tolerance' not in st.session_state:
        st.session_state.recognition_tolerance = 0.6
    
    # Initialize detector and recognizer
    if 'detector' not in st.session_state:
        st.session_state.detector = FaceDetector(method=st.session_state.detector_method)
    
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = FaceRecognizer(
            tolerance=st.session_state.recognition_tolerance,
            model=st.session_state.recognition_model
        )
    
    # Navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'About'
    
    # Image cache
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    
    if 'result_image' not in st.session_state:
        st.session_state.result_image = None
    
    # Results cache
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    
    if 'recognition_results' not in st.session_state:
        st.session_state.recognition_results = None

def run_app():
    """Run the Streamlit application."""
    # Set page config
    st.set_page_config(
        page_title="Face Detection and Recognition System",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    sidebar_content()
    
    # Main content
    if st.session_state.current_page == 'About':
        about_page()
    elif st.session_state.current_page == 'Face Detection':
        detection_page()
    elif st.session_state.current_page == 'Face Recognition':
        recognition_page()
    elif st.session_state.current_page == 'Add Face':
        add_face_page()
    elif st.session_state.current_page == 'Settings':
        settings_page()

if __name__ == "__main__":
    run_app()
