# Face Detection and Recognition System

A computer vision project demonstrating face detection and recognition capabilities using OpenCV and the face_recognition library. This system allows users to upload images, detect faces, and match them against a database of known faces.

## Features

- **Face Detection**: Identifies and locates faces in images using OpenCV
- **Face Recognition**: Matches detected faces against known faces using face_recognition library
- **User-friendly Interface**: Simple web interface built with Streamlit
- **Database Management**: Easily add new people to the recognition database

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Assem-ElQersh/face-recognition-system.git
cd face-recognition-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install system dependencies (required for face_recognition):

   For Ubuntu/Debian:
   ```bash
   sudo apt-get install cmake
   sudo apt-get install libboost-all-dev
   ```

   For Windows:
   See detailed instructions on the [dlib installation guide](https://github.com/davisking/dlib#installation)

## Usage

1. Start the web interface:
```bash
streamlit run ui/app.py
```

2. Navigate to the provided URL in your browser (typically http://localhost:8501)

3. Upload an image to detect faces

4. Register new faces by adding labeled images to the data/known_faces directory (one subdirectory per person)

## Sample Data

The repository includes sample images to help you get started:

- `data/known_faces/`: Contains subdirectories for different individuals, each with sample face images
- `data/sample_images/`: Contains sample group photos for testing the detection and recognition

## Project Structure

- `face_detector.py`: Contains the face detection implementation
- `face_recognizer.py`: Contains the face recognition implementation
- `ui/app.py`: The Streamlit web application
- `utils.py`: Utility functions for image processing and data handling

## Requirements

- Python 3.7+
- OpenCV
- face_recognition
- dlib
- numpy
- Streamlit

## Future Improvements

- Add real-time face recognition using webcam
- Implement user authentication system
- Add database integration for storing face encodings
- Improve recognition accuracy with additional algorithms

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Assem-ElQersh/Face-Recognition-System/blob/main/LICENSE) file for details.
