# YOLO Face Detector

A simple real-time face detector with a fun selfie-assistant overlay.

## Overview

This project uses a YOLO face model on webcam input and adds lightweight live effects:

- Face counter
- Target lock on the dominant face
- Selfie-ready indicator when your face is centered
- Focus mode (blur background, keep face sharp)
- Snapshot capture

## Requirements

- Python 3.11
- OpenCV
- Ultralytics YOLO

## Installation

```bash
pip install opencv-python ultralytics
```

## Usage

```bash
python main.py
```

## Features

- Real-time face detection
- Focus mode with a cinematic background blur
- Lock-on confidence overlay for the main face
- Live selfie guidance ring
- One-key snapshot saving

## Controls

- q: Quit
- m: Toggle focus mode
- s: Save snapshot image in the project folder

## License

MIT License
