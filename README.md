# American Sign Language Detection

A real-time American Sign Language (ASL) detection system that uses computer vision and machine learning to recognize hand signs and convert them to text. The application provides a web interface for easy interaction and real-time feedback.

## Features

- **Real-time ASL Detection**: Uses MediaPipe and OpenCV for hand tracking and gesture recognition
- **Web Interface**: Flask-based web application with real-time video feed
- **Interactive Controls**: Space, backspace, clear, and save functionality
- **Text History**: Maintains a history of recognized signs
- **Confidence Scoring**: Provides confidence scores for each detection
- **Responsive Design**: Works on both desktop and mobile devices

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- Flask
- Flask-SocketIO
- TensorFlow Lite
- NumPy
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/20125A0511/ASL-Detection.git
cd ASL
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
ASL/
├── app.py                 # Main application file
├── cleanup.py            # Utility for cleaning up processes
├── requirements.txt      # Project dependencies
├── static/              # Static files (CSS, JS)
│   └── css/
│       └── style.css
├── templates/           # HTML templates
│   ├── index.html
│   └── exit.html
├── utils/              # Utility modules
│   └── cvfpscalc.py
└── model/             # Model files
    └── keypoint_classifier/
        ├── keypoint_classifier.tflite
        └── keypoint_classifier_label.csv
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the displayed URL (typically http://localhost:5000)

3. Use the web interface to:
   - View real-time video feed
   - See recognized ASL signs
   - Use control buttons (Space, Backspace, Clear, Save)
   - View text history

## Controls

- **Space**: Add a space to the text
- **Backspace**: Remove the last character
- **Clear**: Clear the entire text history
- **Save**: Save the current text and exit

## Development

The application uses:
- MediaPipe for hand tracking
- TensorFlow Lite for sign classification
- Flask for web server
- Socket.IO for real-time communication
- OpenCV for video processing
