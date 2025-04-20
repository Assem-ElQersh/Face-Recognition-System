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
git clone https://github.com/Assem-ElQersh/Face-Recognition-Systemm.git
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

5. Create sample data (test images):
```bash
python create_sample_data.py
```

## Usage

1. Start the web interface:
```bash
streamlit run ui/app.py
```

2. Navigate to the provided URL in your browser (typically http://localhost:8501)

3. Upload an image to detect faces

4. Register new faces by adding labeled images to the data/known_faces directory (one subdirectory per person)

## Sample Data

The system includes a script to create sample data automatically. Run `python create_sample_data.py` to:

1. Create the necessary directory structure
2. Download sample face images from the OpenCV repository (if available)
3. Generate synthetic face images if downloads fail

The script will populate:
- `data/known_faces/person1/` and `data/known_faces/person2/` with individual face images
- `data/sample_images/` with group images for testing detection and recognition

If you want to use your own images:
- Place individual face images in the appropriate person folders (e.g., `data/known_faces/john/image1.jpg`)
- Place group/test images in the `data/sample_images/` directory

## Repository Structure

```
Face-Recognition-System/
├── README.md                     # Project documentation
├── QUICKSTART.md                 # Quick start guide
├── requirements.txt              # Dependencies
├── main.py                       # Main application entry point
├── create_sample_data.py         # Script to create sample data
├── face_detector.py              # Face detection module
├── face_recognizer.py            # Face recognition module
├── utils.py                      # Utility functions
├── ui/
│   ├── __init__.py
│   ├── app.py                    # UI application using Streamlit
│   └── components.py             # UI components
├── data/
│   ├── known_faces/              # Directory for known face images
│   │   ├── person1/              # Each person gets their own directory
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   └── person2/
│   │       ├── image1.jpg
│   │       └── image2.jpg
│   └── sample_images/            # Sample images for testing
│       ├── group1.jpg
│       └── group2.jpg
└── models/                       # Directory for model files (if needed)
```

## Command Line Interface

In addition to the web interface, you can use the command-line interface:

**Face Detection:**
```bash
python main.py --mode detect --image path/to/image.jpg --output results/output.jpg
```

**Face Recognition:**
```bash
python main.py --mode recognize --image path/to/image.jpg --output results/output.jpg
```

**Add a Face:**
```bash
python main.py --mode add_face --image path/to/face.jpg --name "Person Name"
```

**Encode Known Faces:**
```bash
python main.py --mode encode
```

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
