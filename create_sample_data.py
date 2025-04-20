"""
Script to create sample data for the Face Detection and Recognition System.
Creates directory structure and sample images for testing.
"""

import os
import sys
import cv2
import numpy as np
import urllib.request
import zipfile
import io
import shutil
from PIL import Image
from typing import List, Dict, Tuple

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project module
import utils

def create_directory_structure():
    """Create the initial directory structure for the project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
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

def download_sample_faces():
    """Download sample face images from public datasets."""
    # URLs for sample faces (using small public domain images)
    sample_faces = {
        "person1": [
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg"
        ],
        "person2": [
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
        ]
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Download faces for each person
    for person, urls in sample_faces.items():
        person_dir = os.path.join(current_dir, 'data', 'known_faces', person)
        os.makedirs(person_dir, exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                # Download image
                img_path = os.path.join(person_dir, f"image{i+1}.jpg")
                print(f"Downloading {url} to {img_path}")
                
                # Download using urllib
                urllib.request.urlretrieve(url, img_path)
                
                # Verify image was downloaded
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    print(f"Downloaded {img_path}")
                else:
                    print(f"Failed to download {url}")
            
            except Exception as e:
                print(f"Error downloading {url}: {e}")

def download_sample_group_images():
    """Download sample group images for testing face detection and recognition."""
    # URLs for sample group images
    sample_images = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg"
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(current_dir, 'data', 'sample_images')
    
    # Download group images
    for i, url in enumerate(sample_images):
        try:
            # Download image
            img_path = os.path.join(sample_dir, f"group{i+1}.jpg")
            print(f"Downloading {url} to {img_path}")
            
            # Download using urllib
            urllib.request.urlretrieve(url, img_path)
            
            # Verify image was downloaded
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                print(f"Downloaded {img_path}")
            else:
                print(f"Failed to download {url}")
        
        except Exception as e:
            print(f"Error downloading {url}: {e}")

def create_synthetic_faces():
    """Create synthetic face images when real images can't be downloaded."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create synthetic images for known faces
    for person in ["person1", "person2"]:
        person_dir = os.path.join(current_dir, 'data', 'known_faces', person)
        os.makedirs(person_dir, exist_ok=True)
        
        # Check if directory is empty
        if not os.listdir(person_dir):
            print(f"Creating synthetic images for {person}")
            
            # Create two synthetic images per person
            for i in range(2):
                # Create a blank image
                img = np.zeros((300, 300, 3), dtype=np.uint8)
                
                # Draw a simple face-like shape
                # Add a circle for the face
                cv2.circle(img, (150, 150), 100, (200, 200, 200), -1)
                
                # Add eyes
                cv2.circle(img, (110, 120), 20, (255, 255, 255), -1)
                cv2.circle(img, (190, 120), 20, (255, 255, 255), -1)
                cv2.circle(img, (110, 120), 10, (0, 0, 0), -1)
                cv2.circle(img, (190, 120), 10, (0, 0, 0), -1)
                
                # Add mouth
                cv2.ellipse(img, (150, 180), (40, 20), 0, 0, 180, (0, 0, 0), 5)
                
                # Make each image slightly different
                if i == 1:
                    # Add a hat
                    cv2.rectangle(img, (90, 30), (210, 70), (0, 0, 255), -1)
                else:
                    # Add glasses
                    cv2.line(img, (90, 120), (130, 120), (0, 0, 0), 3)
                    cv2.line(img, (170, 120), (210, 120), (0, 0, 0), 3)
                    cv2.line(img, (150, 115), (150, 125), (0, 0, 0), 2)
                
                # Save the image
                img_path = os.path.join(person_dir, f"synthetic{i+1}.jpg")
                cv2.imwrite(img_path, img)
                print(f"Created {img_path}")
    
    # Create synthetic images for sample images
    sample_dir = os.path.join(current_dir, 'data', 'sample_images')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Check if directory is empty
    if not os.listdir(sample_dir):
        print("Creating synthetic group images")
        
        # Create a group image with multiple faces
        img = np.zeros((500, 800, 3), dtype=np.uint8)
        
        # Add multiple face-like shapes
        positions = [(200, 200), (500, 250), (350, 150)]
        colors = [(200, 200, 200), (220, 190, 170), (180, 210, 220)]
        
        for i, (pos, color) in enumerate(zip(positions, colors)):
            # Add a circle for the face
            cv2.circle(img, pos, 80, color, -1)
            
            # Add eyes
            eye_offset_x = 30
            eye_offset_y = 20
            cv2.circle(img, (pos[0] - eye_offset_x, pos[1] - eye_offset_y), 15, (255, 255, 255), -1)
            cv2.circle(img, (pos[0] + eye_offset_x, pos[1] - eye_offset_y), 15, (255, 255, 255), -1)
            cv2.circle(img, (pos[0] - eye_offset_x, pos[1] - eye_offset_y), 7, (0, 0, 0), -1)
            cv2.circle(img, (pos[0] + eye_offset_x, pos[1] - eye_offset_y), 7, (0, 0, 0), -1)
            
            # Add mouth
            cv2.ellipse(img, (pos[0], pos[1] + 30), (30, 15), 0, 0, 180, (0, 0, 0), 4)
        
        # Save the image
        img_path = os.path.join(sample_dir, "group1.jpg")
        cv2.imwrite(img_path, img)
        print(f"Created {img_path}")
        
        # Create another group image
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Add a background
        img[:, :] = (100, 150, 100)
        
        # Add multiple face-like shapes in different positions
        positions = [(150, 300), (400, 200), (650, 350), (300, 450)]
        colors = [(200, 200, 200), (220, 190, 170), (180, 210, 220), (210, 180, 190)]
        
        for i, (pos, color) in enumerate(zip(positions, colors)):
            # Add a circle for the face
            cv2.circle(img, pos, 70, color, -1)
            
            # Add eyes
            eye_offset_x = 25
            eye_offset_y = 15
            cv2.circle(img, (pos[0] - eye_offset_x, pos[1] - eye_offset_y), 13, (255, 255, 255), -1)
            cv2.circle(img, (pos[0] + eye_offset_x, pos[1] - eye_offset_y), 13, (255, 255, 255), -1)
            cv2.circle(img, (pos[0] - eye_offset_x, pos[1] - eye_offset_y), 6, (0, 0, 0), -1)
            cv2.circle(img, (pos[0] + eye_offset_x, pos[1] - eye_offset_y), 6, (0, 0, 0), -1)
            
            # Add mouth - alternate between smile and neutral
            if i % 2 == 0:
                cv2.ellipse(img, (pos[0], pos[1] + 25), (25, 15), 0, 0, 180, (0, 0, 0), 3)
            else:
                cv2.line(img, (pos[0] - 25, pos[1] + 25), (pos[0] + 25, pos[1] + 25), (0, 0, 0), 3)
        
        # Save the image
        img_path = os.path.join(sample_dir, "group2.jpg")
        cv2.imwrite(img_path, img)
        print(f"Created {img_path}")

def main():
    """Main function to create sample data."""
    print("Creating sample data for Face Detection and Recognition System")
    
    # Create directory structure
    create_directory_structure()
    
    # Try to download real images
    print("\nAttempting to download sample faces...")
    download_sample_faces()
    
    print("\nAttempting to download sample group images...")
    download_sample_group_images()
    
    # Create synthetic images if needed
    print("\nCreating synthetic images if needed...")
    create_synthetic_faces()
    
    print("\nSample data creation completed!")

if __name__ == "__main__":
    main()
