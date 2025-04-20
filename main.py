"""
Main entry point for the Face Detection and Recognition System.
Provides a command-line interface for basic operations.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from typing import List, Tuple, Dict, Union, Optional

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
import utils

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Face Detection and Recognition System')
    
    # Main operation mode
    parser.add_argument('--mode', type=str, default='detect',
                      choices=['detect', 'recognize', 'add_face', 'encode', 'ui'],
                      help='Operation mode')
    
    # Input image
    parser.add_argument('--image', type=str, help='Path to input image')
    
    # Output image
    parser.add_argument('--output', type=str, help='Path to output image')
    
    # Person name (for add_face mode)
    parser.add_argument('--name', type=str, help='Person name for add_face mode')
    
    # Detection method
    parser.add_argument('--detector', type=str, default='dnn',
                      choices=['haar', 'dnn'],
                      help='Face detection method')
    
    # Recognition model
    parser.add_argument('--model', type=str, default='hog',
                      choices=['hog', 'cnn'],
                      help='Face recognition model (hog is faster, cnn is more accurate)')
    
    # Recognition tolerance
    parser.add_argument('--tolerance', type=float, default=0.6,
                      help='Recognition tolerance (lower is stricter)')
    
    # Force reloading of encodings
    parser.add_argument('--force-reload', action='store_true',
                      help='Force reload of face encodings')
    
    return parser.parse_args()

def main():
    """
    Main function to run the Face Detection and Recognition System.
    """
    # Create initial directory structure if needed
    utils.create_directory_structure()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Start UI if selected
    if args.mode == 'ui':
        # Import here to avoid circular imports
        from ui.app import run_app
        run_app()
        return
    
    # Check if image is provided for other modes
    if args.mode != 'encode' and args.image is None:
        print("Error: Image path is required for this mode")
        return
    
    # Initialize detector and recognizer
    detector = FaceDetector(method=args.detector)
    recognizer = FaceRecognizer(tolerance=args.tolerance, model=args.model)
    
    # Handle different modes
    if args.mode == 'encode':
        # Encode known faces
        recognizer.encode_known_faces(force_reload=args.force_reload)
        print("Face encoding completed")
        return
    
    # Load input image
    try:
        image = utils.load_image(args.image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Resize large images
    image = utils.resize_image(image, max_size=1200)
    
    if args.mode == 'detect':
        # Detect faces
        faces = detector.detect(image)
        
        # Draw faces on image
        result_image = detector.draw_faces(image, faces)
        
        print(f"Detected {len(faces)} faces")
        
    elif args.mode == 'recognize':
        # Detect faces
        faces = detector.detect(image)
        
        # Convert face locations format from (x, y, w, h) to (top, right, bottom, left)
        face_locations = []
        for (x, y, w, h) in faces:
            top, right, bottom, left = y, x + w, y + h, x
            face_locations.append((top, right, bottom, left))
        
        # Recognize faces
        face_locations, names, confidences = recognizer.recognize_faces(image, face_locations)
        
        # Draw results
        result_image = recognizer.draw_results(image, face_locations, names, confidences)
        
        print(f"Recognized {len(names)} faces")
        for i, name in enumerate(names):
            confidence = confidences[i] if i < len(confidences) else 0.0
            print(f" - {name} ({confidence:.2f})")
        
    elif args.mode == 'add_face':
        # Check if name is provided
        if args.name is None:
            print("Error: Person name is required for add_face mode")
            return
        
        # Detect faces
        faces = detector.detect(image)
        
        if not faces:
            print("No faces detected in the image")
            return
        
        # Use the first face (or prompt to select if multiple)
        if len(faces) > 1:
            print(f"Multiple faces detected ({len(faces)}). Using the first one.")
        
        # Crop face with margin
        face_location = faces[0]
        face_image = utils.crop_face(image, face_location)
        
        # Add face to recognizer
        success = recognizer.add_face(args.name, face_image)
        
        if success:
            print(f"Successfully added face for {args.name}")
            # Draw rectangle around the added face
            result_image = detector.draw_faces(image, [face_location])
        else:
            print(f"Failed to add face for {args.name}")
            result_image = image
    
    # Save output image if path is provided
    if args.output:
        utils.save_image(result_image, args.output)
        print(f"Saved result to {args.output}")
    
    # Display result
    cv2.imshow('Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
