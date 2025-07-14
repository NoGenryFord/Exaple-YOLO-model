# YOLO Object Detection & DeepSort Tracking

This project uses YOLOv8 for object detection and DeepSort for object tracking. It supports video file input or real-time camera processing, and is tested on both PC and Raspberry Pi (including Pi 5).

---

## Requirements

- Python 3.8+
- Virtual environment (`venv`)
- Required libraries (see below)
- For Raspberry Pi: Pi OS (Bookworm recommended), camera enabled, GStreamer installed

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NoGenryFord/Exaple-YOLO-model.git
   cd Exaple-YOLO-model
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # For Windows:
   .\venv\Scripts\activate
   # For Linux/Mac:
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Program

### Option 1: PyTorch Version (Original)
1. **Start the script**
   ```bash
   python main.py
   ```

### Option 2: TensorFlow Lite Version (Optimized for ARM/Raspberry Pi)
1. **Convert model to TFLite (if not already done)**
   ```bash
   python src/convert_to_tflite/onnx2tf_converter.py
   ```

2. **Run TFLite version**
   ```bash
   python main_tflite.py
   ```

3. **Test TFLite model**
   ```bash
   python test_tflite.py
   ```

### For Raspberry Pi 5 Deployment
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-pip python3-venv libgstreamer1.0-0 gstreamer1.0-plugins-base

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install TensorFlow Lite runtime (lighter than full TensorFlow)
pip install tflite-runtime

# Install other dependencies
pip install opencv-python deep-sort-realtime ultralytics numpy

# Run TFLite version
python3 main_tflite.py
```

### Controls (Both Versions)
   - `ESC`: Exit
   - `g`: Toggle grayscale mode
   - `c`: Switch to default camera
   - `1`: Switch to Raspberry Pi camera (GStreamer)
   - `v`: Restart video
   - `r`: Reset tracker
   - `r`: Reset selection (if implemented)

3. **Functionality**
   - **Object Detection**: YOLOv8 model detects objects in video/camera stream.
   - **Object Tracking**: DeepSort tracks detected objects across frames.
   - **Interactive**: Switch video sources, toggle modes, and view real-time FPS and confidence.

---

## Project Structure

```
Exaple-YOLO-model/
│
├── main.py                  # Main script (all logic here)
├── weights/
│   └── YOLO/
│       └── model_3_best.pt  # YOLOv8 model weights
├── data/
│   └── tank1.mp4            # Example input video
├── venv/                    # Virtual environment (not in Git)
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## Dependencies

Main libraries:

- ultralytics
- deep_sort_realtime
- opencv-python
- numpy

Install with:

```bash
pip install -r requirements.txt
```

**Note:** Ensure the model weights file (`model_3_best.pt`) is in the correct folder. For camera use, make sure the camera is connected and enabled on your device.

---

## Running on Raspberry Pi 5

- Make sure your Pi OS is up to date and camera is enabled (`libcamera-hello` should work).
- Install GStreamer and plugins:
  ```bash
  sudo apt update
  sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
  sudo apt install -y gstreamer1.0-libcamera
  ```
- For best performance, use the GStreamer pipeline for the Pi camera:
  - In the app, press `1` to switch to the Pi camera (`v4l2src device=/dev/video0 ! videoconvert ! appsink`).

---

## Performance Tips & Compilation

- For best speed on Raspberry Pi, use:
  - Lower video resolution (e.g., 320x240)
  - Lower FPS (e.g., 15)
- For maximum performance, use hardware acceleration (e.g., OpenVINO, Coral, or NPU if available on your Pi).

---

## License & Legal

**COMMERCIAL SOFTWARE**  
Copyright © 2025. All rights reserved.

This software is provided under a limited commercial license.  
See [LICENSE](LICENSE) for details.

For licensing, contact:

- Email: your.contact@example.com
- Phone: +XX XXX XXX XXXX
