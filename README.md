# AI-Based Home Security System with Email Alerts

An AI-powered home security system developed using Python, Streamlit, OpenCV, YOLOv8, and MediaPipe.  
The system detects motion, intruders, fire-like events, and fall incidents in real time and sends email alerts with image snapshots.

## Features

- Real-time webcam/video monitoring
- Person detection using YOLOv8 / HOG detector
- Motion detection using MOG2 background subtraction
- Fire detection using HSV color filtering
- Fall detection using MediaPipe pose estimation
- Email alerts with image snapshots
- Event logging with CSV export
- Streamlit interactive dashboard

## Technologies Used

- Python
- Streamlit
- OpenCV
- YOLOv8
- MediaPipe
- NumPy
- Pandas

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
AI-Based-Home-Security-System/
│
├── app.py
├── email_alert.py
├── requirements.txt
├── README.md
└── screenshots/
```

## Future Improvements

- Face recognition
- SMS alert integration
- Cloud database support
- Mobile application support
- Advanced fire detection using deep learning

## Disclaimer

This project is a prototype developed for learning and experimentation purposes.
