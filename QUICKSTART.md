# Face Detection and Recognition System - Quick Start Guide

This guide will help you quickly set up and run the Face Detection and Recognition System.

## Prerequisites

- Python 3.7 or higher
- Pip package manager
- Git (optional)

## Installation

1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/face-recognition-system.git
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

4. Create sample data:
```bash
python create_sample_data.py
```

## Running the System

There are two ways to use the system:

### 1. Web Interface (Recommended)

Start the Streamlit web application:
```bash
streamlit run ui/app.py
```

This will open a web browser with the user interface. Navigate through the sidebar menu to:
- Detect faces in images
- Recognize faces against your database
- Add new faces to your database
- Adjust system settings

### 2. Command Line Interface

The system also provides a command-line interface for various operations:

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

## Project Structure

- `face_detector.py`: Face detection module
- `face_recognizer.py`: Face recognition module
- `utils.py`: Utility functions
- `ui/app.py`: Streamlit web application
- `data/known_faces/`: Directory for known face images
- `data/sample_images/`: Sample images for testing
- `models/`: Directory for model files

## Tips for Best Results

1. **Adding Faces**:
   - Use clear, well-lit images of faces
   - Add multiple images per person for better recognition
   - Make sure the face is clearly visible and not too small in the image

2. **Recognition**:
   - Adjust the tolerance setting if you get too many false positives or negatives
   - The "dnn" detector is more accurate but slower than "haar"
   - The "cnn" recognition model is more accurate than "hog" but requires more resources

3. **Performance**:
   - For faster processing on less powerful machines, use the "haar" detector and "hog" recognition model
   - Resize large images before processing

## Troubleshooting

- **Installation Issues**: Make sure you have the required system dependencies installed (see README.md)
- **No Faces Detected**: Try using a clearer image with better lighting
- **Poor Recognition**: Add more reference images of the person from different angles
- **Error Loading DNN Models**: The application will fall back to Haar Cascade if DNN models cannot be loaded

## Next Steps

- Add more faces to your database
- Try the system with different settings
- Experiment with real-time webcam detection (coming in a future update)
