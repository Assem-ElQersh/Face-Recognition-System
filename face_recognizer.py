"""
Face recognition module using face_recognition library.
Provides functionality to encode faces and recognize them against known faces.
"""

import face_recognition
import os
import pickle
import cv2
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import time
import glob

class FaceRecognizer:
    """
    Class for recognizing faces using face_recognition library.
    Handles encoding and matching of faces.
    """
    
    def __init__(self, known_faces_dir: str = None, encodings_path: str = None, 
                tolerance: float = 0.6, model: str = "hog") -> None:
        """
        Initialize the face recognizer.
        
        Args:
            known_faces_dir (str, optional): Directory containing folders of known faces
            encodings_path (str, optional): Path to save/load face encodings
            tolerance (float): Recognition tolerance (lower is stricter, default: 0.6)
            model (str): Face detection model, either "hog" (faster) or "cnn" (more accurate)
        """
        # Current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default paths if not provided
        if known_faces_dir is None:
            self.known_faces_dir = os.path.join(current_dir, 'data', 'known_faces')
        else:
            self.known_faces_dir = known_faces_dir
            
        if encodings_path is None:
            self.encodings_path = os.path.join(current_dir, 'models', 'face_encodings.pkl')
        else:
            self.encodings_path = encodings_path
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
        os.makedirs(self.known_faces_dir, exist_ok=True)
        
        self.tolerance = tolerance
        self.model = model
        
        # Dictionary to store encodings: {name: [encoding1, encoding2, ...]}
        self.known_face_encodings = {}
        self.known_face_names = []
        
        # Load existing encodings if available
        self.load_encodings()
    
    def load_encodings(self) -> bool:
        """
        Load face encodings from file if available.
        
        Returns:
            bool: True if encodings were loaded, False otherwise
        """
        if os.path.exists(self.encodings_path):
            try:
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = list(self.known_face_encodings.keys())
                print(f"Loaded {len(self.known_face_names)} known faces.")
                return True
            except Exception as e:
                print(f"Error loading encodings: {e}")
                return False
        return False
    
    def save_encodings(self) -> bool:
        """
        Save face encodings to file.
        
        Returns:
            bool: True if encodings were saved, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
            
            # Save encodings
            with open(self.encodings_path, 'wb') as f:
                pickle.dump({'encodings': self.known_face_encodings}, f)
            print(f"Saved {len(self.known_face_names)} face encodings.")
            return True
        except Exception as e:
            print(f"Error saving encodings: {e}")
            return False
    
    def encode_known_faces(self, force_reload: bool = False) -> int:
        """
        Encode all faces in the known_faces directory.
        Each subdirectory name is used as the person's name.
        
        Args:
            force_reload (bool): Force reloading of all face encodings (default: False)
        
        Returns:
            int: Number of faces encoded
        """
        # Skip if encodings already loaded and force_reload is False
        if self.known_face_names and not force_reload:
            return len(self.known_face_names)
        
        # Reset encodings if force_reload
        if force_reload:
            self.known_face_encodings = {}
            self.known_face_names = []
        
        # Check if directory exists
        if not os.path.exists(self.known_faces_dir):
            print(f"Known faces directory not found: {self.known_faces_dir}")
            return 0
        
        start_time = time.time()
        total_faces = 0
        
        # Get all subdirectories (one per person)
        person_dirs = [d for d in os.listdir(self.known_faces_dir) 
                     if os.path.isdir(os.path.join(self.known_faces_dir, d))]
        
        for person in person_dirs:
            name = person
            person_dir = os.path.join(self.known_faces_dir, person)
            
            # Get all image files
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(glob.glob(os.path.join(person_dir, ext)))
            
            if not image_paths:
                continue
            
            print(f"Encoding faces for: {name} ({len(image_paths)} images)")
            
            # Process each image
            person_encodings = []
            for img_path in image_paths:
                try:
                    # Load image
                    image = face_recognition.load_image_file(img_path)
                    
                    # Find face locations
                    face_locations = face_recognition.face_locations(image, model=self.model)
                    
                    # If no faces found, skip
                    if not face_locations:
                        print(f"No faces found in {img_path}")
                        continue
                    
                    # Get encodings for the first face (assume one face per image in known_faces)
                    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                    person_encodings.append(face_encoding)
                    total_faces += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Store encodings for this person
            if person_encodings:
                self.known_face_encodings[name] = person_encodings
        
        # Update names list
        self.known_face_names = list(self.known_face_encodings.keys())
        
        # Save encodings
        self.save_encodings()
        
        duration = time.time() - start_time
        print(f"Encoded {total_faces} faces for {len(self.known_face_names)} people in {duration:.2f} seconds")
        
        return len(self.known_face_names)
    
    def add_face(self, name: str, image: np.ndarray) -> bool:
        """
        Add a new face to the known faces.
        
        Args:
            name (str): Name of the person
            image (np.ndarray): Image containing the face
        
        Returns:
            bool: True if face was added successfully, False otherwise
        """
        try:
            # Find face locations
            face_locations = face_recognition.face_locations(image, model=self.model)
            
            # If no faces found, return False
            if not face_locations:
                print("No faces found in the image")
                return False
            
            # Get encodings for the first face
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            
            # Add to encodings
            if name in self.known_face_encodings:
                self.known_face_encodings[name].append(face_encoding)
            else:
                self.known_face_encodings[name] = [face_encoding]
                self.known_face_names.append(name)
            
            # Save encodings
            self.save_encodings()
            
            # Create directory for this person if it doesn't exist
            person_dir = os.path.join(self.known_faces_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save the image
            timestamp = int(time.time())
            image_path = os.path.join(person_dir, f"{timestamp}.jpg")
            cv2.imwrite(image_path, image)
            
            return True
        
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
    
    def recognize_faces(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]] = None) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[float]]:
        """
        Recognize faces in an image against known faces.
        
        Args:
            image (np.ndarray): Input image
            face_locations (List[Tuple[int, int, int, int]], optional): Pre-computed face locations
        
        Returns:
            Tuple:
                - List[Tuple[int, int, int, int]]: Face locations (top, right, bottom, left)
                - List[str]: Names of recognized faces
                - List[float]: Confidence scores
        """
        # Ensure we have known faces
        if not self.known_face_names:
            self.encode_known_faces()
            
            # If still no known faces, return empty results
            if not self.known_face_names:
                print("No known faces to recognize against")
                return [], [], []
        
        # Convert BGR to RGB if needed (face_recognition uses RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            if cv2.imwrite('temp.jpg', image[:, :, ::-1]):  # Simple check if BGR
                rgb_image = image[:, :, ::-1]  # BGR to RGB
                os.remove('temp.jpg')
            else:
                rgb_image = image  # Assume already RGB
        else:
            rgb_image = image
        
        # Find face locations if not provided
        if face_locations is None:
            face_locations = face_recognition.face_locations(rgb_image, model=self.model)
        
        # If no faces found, return empty results
        if not face_locations:
            return [], [], []
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Lists to store results
        names = []
        confidences = []
        
        # For each face encoding
        for face_encoding in face_encodings:
            name = "Unknown"
            confidence = 0.0
            
            # Check against all known face encodings
            for person_name, known_encodings in self.known_face_encodings.items():
                # Calculate distances to all encodings for this person
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                # Get best match
                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    best_distance = distances[best_match_index]
                    
                    # Convert distance to confidence (0-1)
                    current_confidence = 1 - best_distance
                    
                    # If better than current best and below tolerance
                    if current_confidence > confidence and best_distance < self.tolerance:
                        confidence = current_confidence
                        name = person_name
            
            names.append(name)
            confidences.append(confidence)
        
        return face_locations, names, confidences
    
    def draw_results(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]], 
                    names: List[str], confidences: List[float] = None) -> np.ndarray:
        """
        Draw recognition results on the image.
        
        Args:
            image (np.ndarray): Input image
            face_locations (List[Tuple[int, int, int, int]]): Face locations (top, right, bottom, left)
            names (List[str]): Names of recognized faces
            confidences (List[float], optional): Confidence scores
        
        Returns:
            np.ndarray: Image with recognition results drawn
        """
        img_copy = image.copy()
        
        # Process each face
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Draw rectangle around the face
            color = (0, 255, 0) if names[i] != "Unknown" else (0, 0, 255)
            cv2.rectangle(img_copy, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = names[i]
            if confidences and i < len(confidences):
                label += f" ({confidences[i]:.2f})"
            
            # Draw label background
            cv2.rectangle(img_copy, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(img_copy, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        return img_copy

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Initialize recognizer
    recognizer = FaceRecognizer()
    
    # Encode known faces
    recognizer.encode_known_faces()
    
    # Current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load a sample image
    sample_image_path = os.path.join(current_dir, 'data', 'sample_images', 'group1.jpg')
    
    if os.path.exists(sample_image_path):
        # Load image
        image = cv2.imread(sample_image_path)
        
        # Recognize faces
        face_locations, names, confidences = recognizer.recognize_faces(image)
        
        # Draw results
        result_image = recognizer.draw_results(image, face_locations, names, confidences)
        
        # Convert from BGR to RGB for displaying with matplotlib
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # Display result
        plt.figure(figsize=(10, 8))
        plt.imshow(result_image_rgb)
        plt.title(f"Recognized {len(names)} faces")
        plt.axis('off')
        plt.show()
    else:
        print(f"Sample image not found at {sample_image_path}")