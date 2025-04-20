"""
Utility functions for the face detection and recognition system.
Provides image processing, file handling, and other helper functions.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import base64
from io import BytesIO
from PIL import Image
import uuid
import glob

def load_image(path: str) -> np.ndarray:
    """
    Load an image from a file path.
    
    Args:
        path (str): Path to the image file
    
    Returns:
        np.ndarray: Loaded image in BGR format
    
    Raises:
        FileNotFoundError: If image file is not found
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    
    return image

def save_image(image: np.ndarray, path: str) -> bool:
    """
    Save an image to a file.
    
    Args:
        image (np.ndarray): Image to save (BGR format)
        path (str): Path to save the image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save image
        return cv2.imwrite(path, image)
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def resize_image(image: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image (np.ndarray): Input image
        max_size (int): Maximum dimension (width or height)
    
    Returns:
        np.ndarray: Resized image
    """
    height, width = image.shape[:2]
    
    # If image is already smaller than max_size
    if height <= max_size and width <= max_size:
        return image
    
    # Calculate new dimensions
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))
    
    # Resize image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def crop_face(image: np.ndarray, face_location: Tuple[int, int, int, int], 
             margin: float = 0.3) -> np.ndarray:
    """
    Crop a face from an image with margin.
    
    Args:
        image (np.ndarray): Input image
        face_location (Tuple[int, int, int, int]): Face location (top, right, bottom, left)
        margin (float): Margin around face as a fraction of face size
    
    Returns:
        np.ndarray: Cropped face image
    """
    top, right, bottom, left = face_location
    height, width = bottom - top, right - left
    
    # Calculate margins
    margin_x = int(width * margin)
    margin_y = int(height * margin)
    
    # Calculate new coordinates with margin
    new_top = max(0, top - margin_y)
    new_bottom = min(image.shape[0], bottom + margin_y)
    new_left = max(0, left - margin_x)
    new_right = min(image.shape[1], right + margin_x)
    
    # Crop image
    return image[new_top:new_bottom, new_left:new_right]

def detect_blur(image: np.ndarray, threshold: int = 100) -> bool:
    """
    Detect if an image is blurry using the Laplacian variance.
    
    Args:
        image (np.ndarray): Input image
        threshold (int): Blur threshold (lower values indicate blur)
    
    Returns:
        bool: True if image is blurry, False otherwise
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    # Return True if variance is below threshold (blurry)
    return variance < threshold

def image_to_base64(image: np.ndarray) -> str:
    """
    Convert an image to base64 string.
    
    Args:
        image (np.ndarray): Input image (BGR format)
    
    Returns:
        str: Base64 encoded image string
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Save to BytesIO
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    
    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"

def base64_to_image(base64_str: str) -> np.ndarray:
    """
    Convert a base64 string to an image.
    
    Args:
        base64_str (str): Base64 encoded image string
    
    Returns:
        np.ndarray: Image in BGR format
    
    Raises:
        ValueError: If base64 string cannot be decoded
    """
    try:
        # Remove header if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_str)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode base64 string to image")
        
        return img
    except Exception as e:
        raise ValueError(f"Error converting base64 to image: {e}")

def get_sample_images() -> List[str]:
    """
    Get list of sample images.
    
    Returns:
        List[str]: List of sample image paths
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(current_dir, 'data', 'sample_images')
    
    if not os.path.exists(sample_dir):
        return []
    
    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(sample_dir, ext)))
    
    return image_files

def get_known_people() -> List[str]:
    """
    Get list of known people directories.
    
    Returns:
        List[str]: List of known people names
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    known_faces_dir = os.path.join(current_dir, 'data', 'known_faces')
    
    if not os.path.exists(known_faces_dir):
        return []
    
    # Get all directories (one per person)
    return [d for d in os.listdir(known_faces_dir) 
            if os.path.isdir(os.path.join(known_faces_dir, d))]

def generate_unique_filename(extension: str = 'jpg') -> str:
    """
    Generate a unique filename using UUID.
    
    Args:
        extension (str): File extension (default: 'jpg')
    
    Returns:
        str: Unique filename
    """
    return f"{uuid.uuid4()}.{extension}"

def create_directory_structure():
    """
    Create the initial directory structure for the project.
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Directories to create
    directories = [
        os.path.join(current_dir, 'data', 'known_faces'),
        os.path.join(current_dir, 'data', 'sample_images'),
        os.path.join(current_dir, 'models'),
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    # Example usage
    create_directory_structure()
    
    # Get known people
    people = get_known_people()
    print(f"Known people: {people}")
    #Create a dir for each known person
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Ensure known_faces directory exists
    known_faces_dir = os.path.join(current_dir, 'data', 'known_faces')
    os.makedirs(known_faces_dir, exist_ok=True)
    
    # Create example people if none exist
    if not people:
        example_people = ["person1", "person2"]
        for person in example_people:
            person_dir = os.path.join(known_faces_dir, person)
            os.makedirs(person_dir, exist_ok=True)
            print(f"Created directory for: {person}")
    
    # Get sample images
    images = get_sample_images()
    print(f"Sample images: {images}")
    
    # Ensure sample_images directory exists
    sample_dir = os.path.join(current_dir, 'data', 'sample_images')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample image placeholder if none exist
    if not images:
        # Create a simple test image
        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Draw some shapes to make it recognizable
        cv2.circle(test_image, (100, 150), 50, (255, 0, 0), -1)  # Blue circle
        cv2.rectangle(test_image, (200, 100), (300, 200), (0, 255, 0), -1)  # Green rectangle
        cv2.line(test_image, (50, 50), (350, 250), (0, 0, 255), 5)  # Red line
        
        # Save the test image
        sample_path = os.path.join(sample_dir, "group1.jpg")
        cv2.imwrite(sample_path, test_image)
        print(f"Created sample image: {sample_path}")
