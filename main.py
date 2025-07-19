from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2 as cv
import numpy as np
import time
import torch
import os

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

# Video paths
VIDEO_FILES = [
    "data/sample_battle_1.mp4",
    "data/sample_battle_2.mp4", 
    "data/sample_battle_3.MP4",
    "data/tank1.mp4",
    "data/tank2.mp4"
]


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
        ("Press 'v' to switch to video", 10, y + 120),
        ("Press 'n' for next video", 10, y + 140),
        ("Press 'p' for previous video", 10, y + 160),
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
    
    # ========================================
    # VIDEO/CAMERA SWITCHING FUNCTIONALITY
    # ========================================
    # Цей блок можна легко видалити, якщо не потрібен
    
    video = None
    current_video_index = 0
    is_camera_mode = True
    
    def open_camera():
        """Відкриває камеру"""
        video = cv.VideoCapture(0)
        if video.isOpened():
            video.set(cv.CAP_PROP_FRAME_WIDTH, STANDARD_WIDTH)
            video.set(cv.CAP_PROP_FRAME_HEIGHT, STANDARD_HEIGHT)
            video.set(cv.CAP_PROP_FPS, MAX_FPS)
            print("✅ Камера налаштована успішно")
            return video, True
        else:
            print("❌ Не вдалося відкрити камеру")
            return None, False
    
    def open_video(video_index):
        """Відкриває відео файл"""
        if video_index < len(VIDEO_FILES):
            video_path = VIDEO_FILES[video_index]
            if os.path.exists(video_path):
                video = cv.VideoCapture(video_path)
                if video.isOpened():
                    print(f"✅ Відео відкрито: {video_path}")
                    return video, False
                else:
                    print(f"❌ Не вдалося відкрити відео: {video_path}")
            else:
                print(f"❌ Відео файл не знайдено: {video_path}")
        return None, True
    
    def switch_to_next_video():
        """Переключає на наступне відео"""
        nonlocal current_video_index, video, is_camera_mode
        if not is_camera_mode:
            current_video_index = (current_video_index + 1) % len(VIDEO_FILES)
            video.release()
            video, is_camera_mode = open_video(current_video_index)
            return video is not None
        return False
    
    def switch_to_previous_video():
        """Переключає на попереднє відео"""
        nonlocal current_video_index, video, is_camera_mode
        if not is_camera_mode:
            current_video_index = (current_video_index - 1) % len(VIDEO_FILES)
            video.release()
            video, is_camera_mode = open_video(current_video_index)
            return video is not None
        return False
    
    def switch_to_camera():
        """Переключає на камеру"""
        nonlocal video, is_camera_mode
        if video:
            video.release()
        video, is_camera_mode = open_camera()
        return video is not None
    
    def switch_to_video():
        """Переключає на відео"""
        nonlocal video, is_camera_mode
        if video:
            video.release()
        video, is_camera_mode = open_video(current_video_index)
        return video is not None
    
    # ========================================
    # END OF VIDEO/CAMERA SWITCHING
    # ========================================
    
    # Налаштування початкового джерела
    video, is_camera_mode = open_camera()
    
    if video is None:
        # Якщо камера не працює, пробуємо відео
        video, is_camera_mode = open_video(current_video_index)
        if video is None:
            print("❌ Не вдалося відкрити ні камеру, ні відео")
            return
    
    print("📋 Керування:")
    print("   ESC - вихід")
    print("   g - перемикання в чорно-білий режим")
    print("   c - переключення на камеру")
    print("   1 - переключення на Raspberry Pi камеру")
    print("   v - переключення на відео")
    print("   n - наступне відео")
    print("   p - попереднє відео")
    
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
            if not is_camera_mode:
                # Якщо відео закінчилося, переходимо до наступного
                if not switch_to_next_video():
                    print("❌ Не вдалося відкрити наступне відео")
                    break
                continue
            else:
                # Show black screen with message if no signal from camera
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
                    "Waiting for camera...",
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
            print(f"🎨 Чорно-білий режим: {'ON' if is_gray_mode else 'OFF'}")
        elif key == ord("c"):  # Switch to default camera
            if not switch_to_camera():
                print("❌ Не вдалося переключитися на камеру")
        elif key == ord("1"):  # Switch to Raspberry Pi camera (GStreamer)
            if video:
                video.release()
            video = cv.VideoCapture(
                "v4l2src device=/dev/video0 ! videoconvert ! appsink", cv.CAP_GSTREAMER
            )
            print("🔄 Перемикання на Raspberry Pi камеру (GStreamer)")
            if not video.isOpened():
                print("❌ Не вдалося відкрити Raspberry Pi камеру")
                video.release()
        elif key == ord("v"):  # Switch to video
            if not switch_to_video():
                print("❌ Не вдалося переключитися на відео")
        elif key == ord("n"):  # Next video
            if not switch_to_next_video():
                print("❌ Не вдалося відкрити наступне відео")
        elif key == ord("p"):  # Previous video
            if not switch_to_previous_video():
                print("❌ Не вдалося відкрити попереднє відео")
    
    if video:
        video.release()
    cv.destroyAllWindows()
    print("👋 Програма завершена")


if __name__ == "__main__":
    main()
# End of file
