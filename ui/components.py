"""
UI components for the Face Detection and Recognition System.
"""

import glob
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer

# Import project modules
import utils


def sidebar_content():
    """Render sidebar content."""
    st.sidebar.title("Navigation")
    
    # Navigation options
    pages = [
        "About",
        "Face Detection",
        "Face Recognition",
        "Add Face",
        "Settings"
    ]
    
    # Navigation selection
    selected_page = st.sidebar.radio("Go to", pages)
    
    # Update current page in session state
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        # Reset result image when changing pages
        st.session_state.result_image = None
    
    # Display current settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Settings")
    st.sidebar.text(f"Detector: {st.session_state.detector_method}")
    st.sidebar.text(f"Recognition Model: {st.session_state.recognition_model}")
    st.sidebar.text(f"Recognition Tolerance: {st.session_state.recognition_tolerance:.2f}")
    
    # Check if face encodings exist
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    encodings_path = os.path.join(current_dir, 'models', 'face_encodings.pkl')
    if os.path.exists(encodings_path):
        # Add button to reload encodings
        if st.sidebar.button("Reload Face Encodings"):
            st.session_state.recognizer.encode_known_faces(force_reload=True)
            st.sidebar.success("Face encodings reloaded!")
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by Assem ELQersh")

def about_page():
    """Render the About page."""
    st.title("Face Detection and Recognition System")
    
    st.markdown("""
    ## About this Project
    
    This application demonstrates face detection and recognition capabilities using 
    **OpenCV** and the **face_recognition** library. The system allows you to:
    
    - **Detect faces** in images
    - **Recognize faces** against a database of known faces
    - **Add new faces** to the recognition database
    
    ## How to Use
    
    1. Use the sidebar to navigate between different features
    2. Upload images for processing
    3. View results and manage your face database
    
    ## Technologies Used
    
    - Python
    - OpenCV
    - face_recognition library (based on dlib)
    - Streamlit
    
    ## Getting Started
    
    To get the most out of this system, start by adding some faces to your database 
    using the **Add Face** feature.
    """)
    
    # Display sample images
    st.subheader("Sample Images")
    
    # Get sample images
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(current_dir, 'data', 'sample_images')
    
    if os.path.exists(sample_dir):
        # Get image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(sample_dir, ext)))
        
        if image_files:
            # Display images in a grid
            cols = st.columns(min(3, len(image_files)))
            for i, img_path in enumerate(image_files[:3]):  # Limit to 3 samples
                with cols[i % 3]:
                    img = Image.open(img_path)
                    st.image(img, caption=os.path.basename(img_path), use_column_width=True)
        else:
            st.info("No sample images found. You can add some to the 'data/sample_images' directory.")
    else:
        st.info("Sample images directory not found.")

def detection_page():
    """Render the Face Detection page."""
    st.title("Face Detection")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Process image if uploaded
    if uploaded_file is not None:
        # Read image
        image = process_uploaded_image(uploaded_file)
        
        if image is not None:
            # Show original image
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Detect faces
            with st.spinner("Detecting faces..."):
                faces = st.session_state.detector.detect(image)
            
            # Draw faces on image
            result_image = st.session_state.detector.draw_faces(image, faces)
            
            # Store in session state
            st.session_state.current_image = image
            st.session_state.result_image = result_image
            st.session_state.detection_results = faces
            
            # Show result
            st.subheader("Detection Result")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Show detection info
            st.info(f"Detected {len(faces)} faces in the image.")
            
            # Option to save result
            if st.button("Save Result"):
                # Create a temporary file to save the image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    cv2.imwrite(tmp_file.name, result_image)
                    
                    # Provide download link
                    with open(tmp_file.name, "rb") as file:
                        st.download_button(
                            label="Download Image",
                            data=file,
                            file_name="face_detection_result.jpg",
                            mime="image/jpeg"
                        )

def recognition_page():
    """Render the Face Recognition page."""
    st.title("Face Recognition")
    
    # Check if face encodings exist
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    encodings_path = os.path.join(current_dir, 'models', 'face_encodings.pkl')
    
    if not os.path.exists(encodings_path):
        st.warning("No face encodings found. Add some faces first.")
        
        # Add button to go to Add Face page
        if st.button("Go to Add Face"):
            st.session_state.current_page = "Add Face"
            st.experimental_rerun()
        
        return
    
    # Encode known faces if not already done
    if not st.session_state.recognizer.known_face_names:
        with st.spinner("Loading face encodings..."):
            st.session_state.recognizer.encode_known_faces()
            
        # Show known faces
        st.success(f"Loaded {len(st.session_state.recognizer.known_face_names)} known faces.")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Process image if uploaded
    if uploaded_file is not None:
        # Read image
        image = process_uploaded_image(uploaded_file)
        
        if image is not None:
            # Show original image
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Detect and recognize faces
            with st.spinner("Recognizing faces..."):
                # Detect faces
                faces = st.session_state.detector.detect(image)
                
                # Convert face locations format from (x, y, w, h) to (top, right, bottom, left)
                face_locations = []
                for (x, y, w, h) in faces:
                    top, right, bottom, left = y, x + w, y + h, x
                    face_locations.append((top, right, bottom, left))
                
                # Recognize faces
                face_locations, names, confidences = st.session_state.recognizer.recognize_faces(image, face_locations)
                
                # Draw results
                result_image = st.session_state.recognizer.draw_results(image, face_locations, names, confidences)
                
                # Store results in session state
                st.session_state.current_image = image
                st.session_state.result_image = result_image
                st.session_state.recognition_results = (face_locations, names, confidences)
            
            # Show result
            st.subheader("Recognition Result")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Show recognition info
            st.subheader("Recognition Details")
            
            if names:
                # Create a table of results
                results_data = []
                for i, (name, confidence) in enumerate(zip(names, confidences)):
                    results_data.append({
                        "Face #": i + 1,
                        "Name": name,
                        "Confidence": f"{confidence:.2f}"
                    })
                
                st.table(results_data)
            else:
                st.info("No faces recognized in the image.")
            
            # Option to save result
            if st.button("Save Result"):
                # Create a temporary file to save the image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    cv2.imwrite(tmp_file.name, result_image)
                    
                    # Provide download link
                    with open(tmp_file.name, "rb") as file:
                        st.download_button(
                            label="Download Image",
                            data=file,
                            file_name="face_recognition_result.jpg",
                            mime="image/jpeg"
                        )

def add_face_page():
    """Render the Add Face page."""
    st.title("Add Face to Database")
    
    # Input for person name
    person_name = st.text_input("Person Name", "")
    
    # Validate name
    if person_name:
        # Remove special characters and spaces
        valid_name = ''.join(c for c in person_name if c.isalnum() or c == '_')
        if valid_name != person_name:
            st.warning("Name should only contain alphanumeric characters and underscores. Spaces will be replaced with underscores.")
            person_name = valid_name
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image with a clear face", type=["jpg", "jpeg", "png"])
    
    # Process image if uploaded
    if uploaded_file is not None and person_name:
        # Read image
        image = process_uploaded_image(uploaded_file)
        
        if image is not None:
            # Show original image
            st.subheader("Uploaded Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Detect faces
            with st.spinner("Detecting faces..."):
                faces = st.session_state.detector.detect(image)
            
            if not faces:
                st.error("No faces detected in the image. Please upload a clearer image.")
            elif len(faces) > 1:
                st.warning(f"Multiple faces detected ({len(faces)}). Using the first face.")
                
                # Draw faces on image for preview
                preview_image = st.session_state.detector.draw_faces(image, faces)
                st.image(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Add button for each face
                st.subheader("Select a face to add")
                cols = st.columns(min(3, len(faces)))
                
                for i, face in enumerate(faces):
                    # Crop face
                    face_img = utils.crop_face(image, face)
                    
                    # Display in column
                    with cols[i % 3]:
                        st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption=f"Face #{i+1}")
                        if st.button(f"Add Face #{i+1}", key=f"add_face_{i}"):
                            # Add face to recognizer
                            success = st.session_state.recognizer.add_face(person_name, face_img)
                            if success:
                                st.success(f"Successfully added face for {person_name}!")
                            else:
                                st.error(f"Failed to add face for {person_name}.")
            else:
                # Single face detected
                face = faces[0]
                
                # Crop face with margin
                face_img = utils.crop_face(image, face)
                
                # Show cropped face
                st.subheader("Detected Face")
                st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), width=200)
                
                # Add face button
                if st.button("Add Face to Database"):
                    # Add face to recognizer
                    success = st.session_state.recognizer.add_face(person_name, face_img)
                    if success:
                        st.success(f"Successfully added face for {person_name}!")
                    else:
                        st.error(f"Failed to add face for {person_name}.")
    
    # Show existing faces
    st.markdown("---")
    st.subheader("Current Faces in Database")
    
    # Get known people
    known_people = utils.get_known_people()
    
    if known_people:
        # Display in a grid
        cols = st.columns(3)
        
        for i, person in enumerate(known_people):
            with cols[i % 3]:
                st.markdown(f"**{person}**")
                
                # Get images for this person
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                person_dir = os.path.join(current_dir, 'data', 'known_faces', person)
                
                if os.path.exists(person_dir):
                    # Get image files
                    image_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        image_files.extend(glob.glob(os.path.join(person_dir, ext)))
                    
                    if image_files:
                        # Display first image
                        img = Image.open(image_files[0])
                        st.image(img, width=150)
                        st.text(f"{len(image_files)} images")
    else:
        st.info("No faces in database yet.")

def settings_page():
    """Render the Settings page."""
    st.title("Settings")
    
    # Face Detection Settings
    st.subheader("Face Detection Settings")
    
    # Detection method
    detection_method = st.radio(
        "Detection Method",
        options=["dnn", "haar"],
        index=0 if st.session_state.detector_method == "dnn" else 1,
        help="DNN is more accurate but slower, Haar is faster but less accurate"
    )
    
    # Face Recognition Settings
    st.subheader("Face Recognition Settings")
    
    # Recognition model
    recognition_model = st.radio(
        "Recognition Model",
        options=["hog", "cnn"],
        index=0 if st.session_state.recognition_model == "hog" else 1,
        help="HOG is faster, CNN is more accurate but requires more resources"
    )
    
    # Recognition tolerance
    recognition_tolerance = st.slider(
        "Recognition Tolerance",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.recognition_tolerance,
        step=0.05,
        help="Lower values are stricter (fewer false positives, more false negatives)"
    )
    
    # Apply changes
    if st.button("Apply Settings"):
        # Update session state
        st.session_state.detector_method = detection_method
        st.session_state.recognition_model = recognition_model
        st.session_state.recognition_tolerance = recognition_tolerance
        
        # Re-initialize detector and recognizer
        st.session_state.detector = FaceDetector(method=detection_method)
        st.session_state.recognizer = FaceRecognizer(
            tolerance=recognition_tolerance,
            model=recognition_model
        )
        
        st.success("Settings updated successfully!")
        
    # Reset settings
    if st.button("Reset to Defaults"):
        # Reset to defaults
        st.session_state.detector_method = "dnn"
        st.session_state.recognition_model = "hog"
        st.session_state.recognition_tolerance = 0.6
        
        # Re-initialize detector and recognizer
        st.session_state.detector = FaceDetector(method="dnn")
        st.session_state.recognizer = FaceRecognizer(
            tolerance=0.6,
            model="hog"
        )
        
        st.success("Settings reset to defaults!")
    
    # Database Management
    st.markdown("---")
    st.subheader("Database Management")
    
    # Reload face encodings
    if st.button("Reload Face Encodings"):
        with st.spinner("Reloading face encodings..."):
            st.session_state.recognizer.encode_known_faces(force_reload=True)
        st.success("Face encodings reloaded successfully!")

def process_uploaded_image(uploaded_file) -> np.ndarray:
    """
    Process an uploaded image file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        np.ndarray: Image in BGR format (for OpenCV) or None if error
    """
    try:
        # Read the file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Resize large images
        image = utils.resize_image(image, max_size=1200)
        
        return image
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None
