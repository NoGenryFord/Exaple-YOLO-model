from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2 as cv
import numpy as np
import time
import torch

# ----------------------
# OpenCV performance settings
# ----------------------
cv.setUseOptimized(True)
cv.setNumThreads(4)

# ----------------------
# Constants
# ----------------------
STANDARD_WIDTH = 640  # Standard frame width
STANDARD_HEIGHT = 480  # Standard frame height
MAX_FPS = 30  # Maximum frames per second
YOLO_SKIP_FRAMES = 3  # Number of frames to skip for YOLO detection


# ----------------------
# Utility functions
# ----------------------
def resize_frame(frame, width=STANDARD_WIDTH, height=STANDARD_HEIGHT):
    """Resize frame to standard size."""
    return cv.resize(frame, (width, height))


def convert_to_gray(frame):
    """Convert frame to grayscale (3 channels)."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.merge([gray, gray, gray])


def draw_hints(frame, is_gray_mode, width, height):
    """Draw on-screen hints and controls with high-contrast background and smaller font."""
    color_bg = (0, 0, 0)  # Black background
    color_text = (255, 255, 255)  # White text
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45  # Smaller font size
    thickness = 1
    y = 30
    hints = [
        (
            "Gray mode ON" if is_gray_mode else "Gray mode OFF",
            width - 250 if is_gray_mode else width - 200,
            y,
        ),
        ("Press 'ESC' to exit", 10, y + 20),
        ("Press 'r' to reset selection", 10, y + 40),
        ("Press 'g' to toggle gray mode", 10, y + 60),
        ("Press 'c' to switch to camera", 10, y + 80),
        ("Press '1' to switch to Raspberry Pi camera", 10, y + 100),
        ("Press 'v' to restart video", 10, y + 120),
    ]
    for text, x, y_pos in hints:
        (text_width, text_height), baseline = cv.getTextSize(
            text, font, font_scale, thickness
        )
        # Draw background rectangle for text
        cv.rectangle(
            frame,
            (x - 2, y_pos - text_height - 2),
            (x + text_width + 2, y_pos + baseline + 2),
            color_bg,
            -1,
        )
        # Draw the text itself
        cv.putText(
            frame, text, (x, y_pos), font, font_scale, color_text, thickness, cv.LINE_AA
        )
    return frame


def limit_fps(frame_start_time, max_fps=30):
    """Sleep to limit the FPS to max_fps. Returns new frame_start_time."""
    frame_end_time = time.time()
    elapsed_time = frame_end_time - frame_start_time
    target_time_per_frame = 1.0 / max_fps
    if elapsed_time < target_time_per_frame:
        time.sleep(target_time_per_frame - elapsed_time)
    return time.time()


def draw_detection(frame, x1, y1, x2, y2, conf):
    """Draw detection bounding box, center, and label above the box with good readability."""
    color_box = (0, 255, 0)  # Green box
    color_center = (0, 0, 255)  # Red center dot
    color_text = (255, 255, 255)  # White text
    color_bg = (0, 0, 0)  # Black background for text
    box_width = x2 - x1
    box_height = y2 - y1
    shrink_factor = 0.7
    new_width = int(box_width * shrink_factor)
    new_height = int(box_height * shrink_factor)
    x_center = x1 + box_width // 2
    y_center = y1 + box_height // 2
    x1_new = x_center - new_width // 2
    y1_new = y_center + box_height // 2
    x2_new = x_center + new_width // 2
    y2_new = y_center - new_height // 2
    # Draw bounding box
    cv.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), color_box, 2)
    # Draw center point
    cv.circle(frame, (x_center, y_center), 2, color_center, -1)
    # Prepare label text (confidence value)
    label = f"{conf:.2f}"
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45  # Smaller font size for confidence
    thickness = 1
    (text_width, text_height), baseline = cv.getTextSize(
        label, font, font_scale, thickness
    )
    # Center the label above the box
    label_x = x_center - text_width // 2
    label_y = y1_new - 10  # 10 pixels above the box
    # Draw background rectangle for text
    cv.rectangle(
        frame,
        (label_x - 4, label_y - text_height - 4),
        (label_x + text_width + 4, label_y + baseline + 4),
        color_bg,
        -1,
    )
    # Draw the text itself
    cv.putText(
        frame,
        label,
        (label_x, label_y),
        font,
        font_scale,
        color_text,
        thickness,
        cv.LINE_AA,
    )
    return frame


# ----------------------
# Main application logic
# ----------------------
def main():
    # Load YOLOv8 model
    model = YOLO("weights/YOLO/model_3_best.pt")
    model.conf = 0.8
    video_path = "data/tank1.mp4"  # Path to the video file
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    cv.namedWindow("Frame")
    is_gray_mode = False
    frame_start_time = time.time()
    frame_count = 0
    fps = 0
    last_time = time.time()

    while True:
        # Read frame from video
        ret, frame = video.read()
        if not ret:
            # Show black screen with message if no signal
            frame = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH, 3), dtype=np.uint8)
            cv.putText(
                frame,
                "No Signal",
                (STANDARD_WIDTH // 2 - 100, STANDARD_HEIGHT // 2),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv.putText(
                frame,
                "Waiting for video...",
                (STANDARD_WIDTH // 2 - 150, STANDARD_HEIGHT // 2 + 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            frame_width = STANDARD_WIDTH
            frame_height = STANDARD_HEIGHT
        else:
            frame_count += 1
            # Resize frame
            frame = resize_frame(frame)
            # Convert to grayscale if needed
            if is_gray_mode:
                frame = convert_to_gray(frame)
            # Get frame dimensions
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            # Calculate and display FPS
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = current_time
            cv.putText(
                frame,
                f"FPS: {fps}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            # Run YOLO detection every YOLO_SKIP_FRAMES frames
            if frame_count % YOLO_SKIP_FRAMES == 0:
                results = model(frame)[0].boxes
                if results:
                    for b in results:
                        cords = b.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, cords)
                        conf = round(b.conf[0].item(), 2)
                        frame = draw_detection(frame, x1, y1, x2, y2, conf)
        # Draw on-screen hints
        frame = draw_hints(frame, is_gray_mode, frame_width, frame_height)
        # Show frame
        cv.imshow("Frame", frame)
        # Limit FPS
        frame_start_time = limit_fps(frame_start_time, MAX_FPS)
        # Keyboard controls (always active)
        key = cv.waitKey(10)
        if key == 27:  # Exit on ESC
            break
        elif key == ord("g"):  # Toggle grayscale mode
            is_gray_mode = not is_gray_mode
            print(
                "Switched to grayscale mode"
                if is_gray_mode
                else "Switched to color mode"
            )
        elif key == ord("c"):  # Switch to default camera
            video.release()
            video = cv.VideoCapture(0)
            print("Default camera opened")
            if not video.isOpened():
                print("Error: Could not open camera.")
                video.release()
            continue
        elif key == ord("1"):  # Switch to Raspberry Pi camera (GStreamer)
            video.release()
            video = cv.VideoCapture(
                "v4l2src device=/dev/video0 ! videoconvert ! appsink", cv.CAP_GSTREAMER
            )
            print("Switched to Raspberry Pi camera (GStreamer)")
            if not video.isOpened():
                print("Error: Could not open Raspberry Pi camera.")
                video.release()
            continue
        elif key == ord("v"):  # Restart video
            video.release()
            video = cv.VideoCapture(video_path)
            print("Video restarted")
            if not video.isOpened():
                print("Error: Could not open video.")
                video.release()
            continue
    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
# End of file
