"""
Face detection module using OpenCV.
Provides both Haar Cascade and DNN-based detectors.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Union

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

class FaceDetector:
    """
    Class for detecting faces in images using different methods.
    Supports both Haar Cascade and DNN-based face detection.
    """
    
    def __init__(self, method: str = "dnn") -> None:
        """
        Initialize the face detector with specified method.
        
        Args:
            method (str): Detection method, either "haar" or "dnn" (default: "dnn")
        """
        self.method = method
        
        if method == "haar":
            # Load Haar Cascade classifier
            haar_path = os.path.join(current_dir, 'models', 'haarcascade_frontalface_default.xml')
            
            # If model doesn't exist locally, use the OpenCV built-in one
            if not os.path.exists(haar_path):
                haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                
            self.face_cascade = cv2.CascadeClassifier(haar_path)
            
        elif method == "dnn":
            # Paths for DNN model files
            model_dir = os.path.join(current_dir, 'models')
            prototxt_path = os.path.join(model_dir, 'deploy.prototxt')
            model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
            
            # If models don't exist locally, we'll handle this in the detect method
            self.prototxt_path = prototxt_path
            self.model_path = model_path
            self.net = None
            
            # Check if model files exist, if not download or raise appropriate message
            if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
                # Ensure model directory exists
                os.makedirs(model_dir, exist_ok=True)
                
                # Try to download model files if they don't exist
                try:
                    print("Downloading DNN model files...")
                    # Download prototxt
                    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                    self._download_file(prototxt_url, prototxt_path)
                    
                    # Download model
                    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                    self._download_file(model_url, model_path)
                    
                    print("Model files downloaded successfully.")
                except Exception as e:
                    print(f"Error downloading model files: {e}")
                    print("Will use OpenCV's built-in face detection.")
                    self.method = "haar"
                    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            raise ValueError("Method must be either 'haar' or 'dnn'")
    
    def _download_file(self, url: str, destination: str) -> None:
        """
        Download a file from a URL to a destination path.
        
        Args:
            url (str): URL to download from
            destination (str): Local path to save the file
        """
        import urllib.request
        urllib.request.urlretrieve(url, destination)
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            confidence_threshold (float): Confidence threshold for DNN detection (default: 0.5)
        
        Returns:
            List[Tuple[int, int, int, int]]: List of face coordinates (x, y, w, h)
        """
        if self.method == "haar":
            return self._detect_haar(image)
        else:  # DNN method
            return self._detect_dnn(image, confidence_threshold)
    
    def _detect_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar Cascade classifier.
        
        Args:
            image (np.ndarray): Input image (BGR format)
        
        Returns:
            List[Tuple[int, int, int, int]]: List of face coordinates (x, y, w, h)
        """
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def _detect_dnn(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN-based detector.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            confidence_threshold (float): Confidence threshold (default: 0.5)
        
        Returns:
            List[Tuple[int, int, int, int]]: List of face coordinates (x, y, w, h)
        """
        # Initialize the DNN if not already done
        if self.net is None:
            if os.path.exists(self.prototxt_path) and os.path.exists(self.model_path):
                self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
            else:
                # Fall back to Haar cascade if DNN model files are not available
                print("DNN model files not found. Falling back to Haar Cascade.")
                self.method = "haar"
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                return self._detect_haar(image)
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Pass the blob through the network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Process detections
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter based on confidence
            if confidence > confidence_threshold:
                # Get coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within image boundaries
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Calculate width and height
                width = endX - startX
                height = endY - startY
                
                # Add to faces list in the format (x, y, w, h)
                faces.append((startX, startY, width, height))
        
        return faces
    
    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                  color: Tuple[int, int, int] = (0, 255, 0), 
                  thickness: int = 2) -> np.ndarray:
        """
        Draw rectangles around detected faces.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            faces (List[Tuple[int, int, int, int]]): List of face coordinates (x, y, w, h)
            color (Tuple[int, int, int]): Rectangle color in BGR format (default: green)
            thickness (int): Line thickness (default: 2)
        
        Returns:
            np.ndarray: Image with rectangles drawn around faces
        """
        img_copy = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        return img_copy

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Initialize detector
    detector = FaceDetector(method="dnn")
    
    # Load a sample image
    sample_image_path = os.path.join(current_dir, 'data', 'sample_images', 'group1.jpg')
    
    if os.path.exists(sample_image_path):
        image = cv2.imread(sample_image_path)
        
        # Detect faces
        faces = detector.detect(image)
        
        # Draw faces on image
        result_image = detector.draw_faces(image, faces)
        
        # Convert from BGR to RGB for displaying with matplotlib
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # Display result
        plt.figure(figsize=(10, 8))
        plt.imshow(result_image_rgb)
        plt.title(f"Detected {len(faces)} faces")
        plt.axis('off')
        plt.show()
    else:
        print(f"Sample image not found at {sample_image_path}")
