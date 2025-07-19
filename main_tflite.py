"""
TensorFlow Lite версія main.py для роботи з конвертованою YOLO моделлю
Оптимізована для Raspberry Pi та інших ARM пристроїв
"""

import tensorflow as tf
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2 as cv
import numpy as np
import time
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
MAX_FPS = 60  # Збільшуємо максимальний FPS
YOLO_SKIP_FRAMES = 2  # Зменшуємо кількість пропущених кадрів для кращої детекції

# Model paths
TFLITE_MODEL_PATH = "weights/YOLO/model_3_simple.tflite"
CONFIDENCE_THRESHOLD = 0.5  # Підвищуємо поріг для більш точної детекції
IOU_THRESHOLD = 0.4

# Video paths
VIDEO_FILES = [
    "data/sample_battle_1.mp4",
    "data/sample_battle_2.mp4", 
    "data/sample_battle_3.MP4",
    "data/tank1.mp4",
    "data/tank2.mp4"
]

# ----------------------
# TFLite Model Class
# ----------------------
class TFLiteYOLO:
    def __init__(self, model_path):
        """Ініціалізація TFLite YOLO моделі"""
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        
        self.load_model()
    
    def load_model(self):
        """Завантажує TFLite модель"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TFLite модель не знайдена: {self.model_path}")
        
        print(f"🔄 Завантажуємо TFLite модель: {self.model_path}")
        
        # Створюємо інтерпретатор
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        # Отримуємо деталі вхідних та вихідних тензорів
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Отримуємо розмір вхідних даних
        self.input_shape = self.input_details[0]['shape']
        print(f"📊 Вхідна форма: {self.input_shape}")
        print(f"📊 Вихідна форма: {self.output_details[0]['shape']}")
        
        print("✅ TFLite модель завантажена успішно!")
    
    def preprocess_frame(self, frame):
        """Підготовка кадру для інференсу"""
        # Отримуємо розмір моделі
        input_shape = self.input_shape
        
        # Розбираємо форму тензора
        if input_shape[1] == 3:  # NCHW format: [batch, channels, height, width]
            model_height = input_shape[2]
            model_width = input_shape[3]
        else:  # NHWC format: [batch, height, width, channels]
            model_height = input_shape[1]
            model_width = input_shape[2]
        
        # Змінюємо розмір кадру
        resized = cv.resize(frame, (model_width, model_height))
        
        # Конвертуємо BGR в RGB
        rgb_frame = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        
        # Нормалізуємо значення пікселів [0-255] -> [0-1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Готуємо тензор згідно з форматом моделі
        if input_shape[1] == 3:  # NCHW format
            # Транспонуємо в NCHW: [H, W, C] -> [C, H, W]
            normalized = np.transpose(normalized, (2, 0, 1))
            # Додаємо batch dimension: [C, H, W] -> [1, C, H, W]
            input_data = np.expand_dims(normalized, axis=0)
        else:  # NHWC format
            # Додаємо batch dimension: [H, W, C] -> [1, H, W, C]
            input_data = np.expand_dims(normalized, axis=0)
        
        # Перевіряємо форму (для debug - можна видалити)
        # print(f"🔧 Вхідна форма після препроцесингу: {input_data.shape}")
        # print(f"🔧 Очікувана форма: {input_shape}")
        
        return input_data.astype(np.float32)
    
    def postprocess_output(self, output_data, original_shape):
        """Обробка виходу TFLite YOLO моделі"""
        if output_data is None:
            return []
        
        # Отримуємо розміри оригінального кадру
        orig_height, orig_width = original_shape[:2]
        
        # Для цієї TFLite моделі вихід має форму [1, 5, 8400]
        # 5 = [x_center, y_center, width, height, confidence]
        predictions = output_data[0]  # Видаляємо batch dimension -> [5, 8400]
        
        # Транспонуємо для зручності: [5, 8400] -> [8400, 5]
        predictions = predictions.T
        
        detections = []
        
        for pred in predictions:
            # Отримуємо координати bbox та confidence
            x_center, y_center, width, height, confidence = pred
            
            # Перевіряємо confidence threshold
            if confidence > CONFIDENCE_THRESHOLD:
                # Нормалізовані координати потрібно помножити на розміри кадру
                # Припускаємо, що модель віддає нормалізовані координати [0, 1]
                x_center_abs = x_center * orig_width
                y_center_abs = y_center * orig_height
                width_abs = width * orig_width
                height_abs = height * orig_height
                
                # Конвертуємо з center format в corner format
                x1 = int(x_center_abs - width_abs / 2)
                y1 = int(y_center_abs - height_abs / 2)
                x2 = int(x_center_abs + width_abs / 2)
                y2 = int(y_center_abs + height_abs / 2)
                
                # Обмежуємо координати
                x1 = max(0, min(x1, orig_width - 1))
                y1 = max(0, min(y1, orig_height - 1))
                x2 = max(0, min(x2, orig_width - 1))
                y2 = max(0, min(y2, orig_height - 1))
                
                # Перевіряємо, чи бокс має розумний розмір
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2, confidence])
        
        return detections
    
    def apply_nms(self, detections, iou_threshold):
        """Застосовує Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Сортуємо за confidence
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        # Застосовуємо NMS
        keep = []
        while detections:
            # Беремо детекцію з найвищим confidence
            current = detections.pop(0)
            keep.append(current)
            
            # Видаляємо детекції з високим IoU
            remaining = []
            for detection in detections:
                if self.calculate_iou(current, detection) < iou_threshold:
                    remaining.append(detection)
            detections = remaining
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """Обчислює Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        # Обчислюємо площу перетину
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Обчислюємо площу об'єднання
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect(self, frame):
        """Виконання детекції об'єктів за допомогою TFLite моделі"""
        try:
            # Препроцесинг кадру
            input_data = self.preprocess_frame(frame)
            
            # Встановлюємо вхідний тензор
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Виконуємо інференс
            self.interpreter.invoke()
            
            # Отримуємо результат
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Постпроцесинг
            detections = self.postprocess_output(output_data, frame.shape)
            
            # Застосовуємо NMS
            detections = self.apply_nms(detections, IOU_THRESHOLD)
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Помилка TFLite детекції: {e}")
            return []

# ----------------------
# Utility functions (копіюємо з original main.py)
# ----------------------
def resize_frame(frame, width=STANDARD_WIDTH, height=STANDARD_HEIGHT):
    """Resize frame to standard size."""
    return cv.resize(frame, (width, height))

def convert_to_gray(frame):
    """Convert frame to grayscale (3 channels)."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.merge([gray, gray, gray])

def draw_hints(frame, is_gray_mode, width, height, fps=0, detections_count=0):
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
        ("TFLite Model Active", 10, y + 180),  # Додаємо індикатор TFLite
        (f"FPS: {fps:.1f}", 10, y + 200),  # FPS
        (f"Detections: {detections_count}", 10, y + 220),  # Кількість детекцій
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
    y1_new = y_center - new_height // 2
    x2_new = x1_new + new_width
    y2_new = y1_new + new_height
    cv.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), color_box, 2)
    cv.circle(frame, (x_center, y_center), 5, color_center, -1)
    
    # Label text
    label = f"Object {conf:.2f}"
    (text_width, text_height), baseline = cv.getTextSize(
        label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    label_y = y1_new - 10
    if label_y < text_height:
        label_y = y1_new + text_height + 10
    cv.rectangle(
        frame,
        (x1_new, label_y - text_height - 5),
        (x1_new + text_width, label_y + baseline),
        color_bg,
        -1,
    )
    cv.putText(
        frame, label, (x1_new, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1
    )

# ----------------------
# Main function
# ----------------------
def main():
    """Головна функція програми з TFLite моделлю"""
    print("🚀 Запуск TFLite YOLO + DeepSort системи...")
    
    # Ініціалізація TFLite моделі
    try:
        yolo_model = TFLiteYOLO(TFLITE_MODEL_PATH)
    except Exception as e:
        print(f"❌ Помилка завантаження TFLite моделі: {e}")
        return
    
    # Ініціалізація DeepSort
    deep_sort = DeepSort(max_age=30, n_init=3)
    
    # ========================================
    # VIDEO/CAMERA SWITCHING FUNCTIONALITY
    # ========================================
    # Цей блок можна легко видалити, якщо не потрібен
    
    cap = None
    current_video_index = 0
    is_camera_mode = True
    
    def open_camera():
        """Відкриває камеру"""
        cap = cv.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, STANDARD_WIDTH)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, STANDARD_HEIGHT)
            cap.set(cv.CAP_PROP_FPS, MAX_FPS)
            print("✅ Камера налаштована успішно")
            return cap, True
        else:
            print("❌ Не вдалося відкрити камеру")
            return None, False
    
    def open_video(video_index):
        """Відкриває відео файл"""
        if video_index < len(VIDEO_FILES):
            video_path = VIDEO_FILES[video_index]
            if os.path.exists(video_path):
                cap = cv.VideoCapture(video_path)
                if cap.isOpened():
                    print(f"✅ Відео відкрито: {video_path}")
                    return cap, False
                else:
                    print(f"❌ Не вдалося відкрити відео: {video_path}")
            else:
                print(f"❌ Відео файл не знайдено: {video_path}")
        return None, True
    
    def switch_to_next_video():
        """Переключає на наступне відео"""
        nonlocal current_video_index, cap, is_camera_mode
        if not is_camera_mode:
            current_video_index = (current_video_index + 1) % len(VIDEO_FILES)
            cap.release()
            cap, is_camera_mode = open_video(current_video_index)
            return cap is not None
        return False
    
    def switch_to_previous_video():
        """Переключає на попереднє відео"""
        nonlocal current_video_index, cap, is_camera_mode
        if not is_camera_mode:
            current_video_index = (current_video_index - 1) % len(VIDEO_FILES)
            cap.release()
            cap, is_camera_mode = open_video(current_video_index)
            return cap is not None
        return False
    
    def switch_to_camera():
        """Переключає на камеру"""
        nonlocal cap, is_camera_mode
        cap.release()
        cap, is_camera_mode = open_camera()
        return cap is not None
    
    def switch_to_video():
        """Переключає на відео"""
        nonlocal cap, is_camera_mode
        cap.release()
        cap, is_camera_mode = open_video(current_video_index)
        return cap is not None
    
    # ========================================
    # END OF VIDEO/CAMERA SWITCHING
    # ========================================
    
    # Налаштування початкового джерела
    cap, is_camera_mode = open_camera()
    
    if cap is None:
        # Якщо камера не працює, пробуємо відео
        cap, is_camera_mode = open_video(current_video_index)
        if cap is None:
            print("❌ Не вдалося відкрити ні камеру, ні відео")
            return
    
    print("📋 Керування:")
    print("   ESC - вихід")
    print("   g - перемикання в чорно-білий режим")
    print("   r - скидання трекера")
    print("   c - переключення на камеру")
    print("   1 - переключення на Raspberry Pi камеру")
    print("   v - переключення на відео")
    print("   n - наступне відео")
    print("   p - попереднє відео")
    
    # Основний цикл
    frame_count = 0
    is_gray_mode = False
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0
    
    while True:
        frame_start_time = time.time()
        
        # Зчитуємо кадр
        ret, frame = cap.read()
        if not ret:
            if not is_camera_mode:
                # Якщо відео закінчилося, переходимо до наступного
                if not switch_to_next_video():
                    print("❌ Не вдалося відкрити наступне відео")
                    break
                continue
            else:
                print("❌ Не вдалося зчитати кадр з камери")
                break
        
        # Змінюємо розмір кадру
        frame = resize_frame(frame)
        
        # Чорно-білий режим
        if is_gray_mode:
            frame = convert_to_gray(frame)
        
        # Детекція об'єктів (кожен YOLO_SKIP_FRAMES кадр для продуктивності)
        detections = []
        if frame_count % YOLO_SKIP_FRAMES == 0:
            try:
                detections = yolo_model.detect(frame)
            except Exception as e:
                print(f"⚠️ Помилка детекції: {e}")
        
        # Малюємо детекції
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            if conf > CONFIDENCE_THRESHOLD:
                draw_detection(frame, int(x1), int(y1), int(x2), int(y2), conf)
        
        # Обчислюємо FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Підказки
        frame = draw_hints(frame, is_gray_mode, STANDARD_WIDTH, STANDARD_HEIGHT, 
                          fps_display, len(detections))
        
        # Показуємо кадр
        cv.imshow("TFLite YOLO + DeepSort", frame)
        
        # Обробка клавіш
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('g'):
            is_gray_mode = not is_gray_mode
            print(f"🎨 Чорно-білий режим: {'ON' if is_gray_mode else 'OFF'}")
        elif key == ord('r'):
            deep_sort = DeepSort(max_age=30, n_init=3)
            print("🔄 Трекер скинуто")
        elif key == ord('c'):
            # Переключення на камеру
            if not switch_to_camera():
                print("❌ Не вдалося переключитися на камеру")
                break
        elif key == ord('1'):  # Switch to Raspberry Pi camera (GStreamer)
            if cap:
                cap.release()
            cap = cv.VideoCapture(
                "v4l2src device=/dev/video0 ! videoconvert ! appsink", cv.CAP_GSTREAMER
            )
            is_camera_mode = True
            print("🔄 Перемикання на Raspberry Pi камеру (GStreamer)")
            if not cap.isOpened():
                print("❌ Не вдалося відкрити Raspberry Pi камеру")
                # Fallback to regular camera
                cap, is_camera_mode = open_camera()
                if cap is None:
                    print("❌ Не вдалося відкрити звичайну камеру")
                    break
        elif key == ord('v'):
            # Переключення на відео
            if not switch_to_video():
                print("❌ Не вдалося переключитися на відео")
                break
        elif key == ord('n'):
            # Наступне відео
            if not switch_to_next_video():
                print("❌ Не вдалося відкрити наступне відео")
                break
        elif key == ord('p'):
            # Попереднє відео
            if not switch_to_previous_video():
                print("❌ Не вдалося відкрити попереднє відео")
                break
        
        # Обмеження FPS (збільшуємо до 60 для кращої продуктивності)
        frame_start_time = limit_fps(frame_start_time, 60)
        frame_count += 1
    
    # Cleanup
    if cap:
        cap.release()
    cv.destroyAllWindows()
    print("👋 Програма завершена")

if __name__ == "__main__":
    main()
