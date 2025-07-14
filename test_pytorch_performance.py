"""
Тестування оригінальної PyTorch YOLO моделі для порівняння з TFLite версією
"""

from ultralytics import YOLO
import cv2 as cv
import numpy as np
import time
import os

# Налаштування
CONFIDENCE_THRESHOLD = 0.3
VIDEO_FILES = [
    "data/sample_battle_1.mp4",
    "data/sample_battle_2.mp4", 
    "data/sample_battle_3.MP4",
    "data/tank1.mp4",
    "data/tank2.mp4"
]

def test_pytorch_vs_tflite():
    """Порівняння продуктивності PyTorch та TFLite"""
    
    print("=== Тест продуктивності PyTorch YOLO ===")
    
    # Завантажуємо PyTorch модель
    try:
        model = YOLO("weights/YOLO/model_3_best.pt")
        print("✅ PyTorch модель завантажена")
    except Exception as e:
        print(f"❌ Помилка завантаження PyTorch моделі: {e}")
        return
    
    # Відкриваємо відео
    video_path = VIDEO_FILES[0] if os.path.exists(VIDEO_FILES[0]) else None
    if video_path:
        cap = cv.VideoCapture(video_path)
        print(f"✅ Відео відкрито: {video_path}")
    else:
        cap = cv.VideoCapture(0)
        print("✅ Камера відкрита")
    
    if not cap.isOpened():
        print("❌ Не вдалося відкрити джерело відео")
        return
    
    # Тестуємо продуктивність
    frame_count = 0
    detection_times = []
    total_detections = 0
    
    print("🔄 Тестування продуктивності...")
    
    while frame_count < 100:  # Тестуємо 100 кадрів
        ret, frame = cap.read()
        if not ret:
            break
        
        # Змінюємо розмір кадру
        frame = cv.resize(frame, (640, 480))
        
        # Тестуємо детекцію кожен 3-й кадр
        if frame_count % 3 == 0:
            start_time = time.time()
            
            # PyTorch детекція
            results = model(frame, conf=CONFIDENCE_THRESHOLD)
            
            detection_time = time.time() - start_time
            detection_times.append(detection_time)
            
            # Підраховуємо детекції
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    total_detections += len(boxes)
        
        frame_count += 1
        
        # Показуємо прогрес
        if frame_count % 10 == 0:
            print(f"   Оброблено {frame_count}/100 кадрів...")
    
    cap.release()
    
    # Показуємо результати
    if detection_times:
        avg_detection_time = np.mean(detection_times)
        max_detection_time = max(detection_times)
        min_detection_time = min(detection_times)
        
        print(f"\n📊 Результати PyTorch YOLO:")
        print(f"   Середній час детекції: {avg_detection_time:.3f}s")
        print(f"   Мінімальний час: {min_detection_time:.3f}s")
        print(f"   Максимальний час: {max_detection_time:.3f}s")
        print(f"   Теоретичний FPS: {1.0/avg_detection_time:.1f}")
        print(f"   Всього детекцій: {total_detections}")
        print(f"   Середньо детекцій на кадр: {total_detections/len(detection_times):.1f}")
        
        # Порівняння з TFLite
        print(f"\n🔄 Порівняння:")
        print(f"   PyTorch: {1.0/avg_detection_time:.1f} FPS")
        print(f"   TFLite (очікується): ~30-60 FPS")
        
        if avg_detection_time > 0.033:  # 30 FPS
            print("   ⚠️  PyTorch модель повільна для реального часу")
        else:
            print("   ✅ PyTorch модель достатньо швидка")

def run_pytorch_demo():
    """Демонстрація PyTorch моделі"""
    
    print("\n=== Демонстрація PyTorch YOLO ===")
    print("Натисніть ESC для виходу, 'v' для переключення відео")
    
    try:
        model = YOLO("weights/YOLO/model_3_best.pt")
        print("✅ PyTorch модель завантажена")
    except Exception as e:
        print(f"❌ Помилка завантаження PyTorch моделі: {e}")
        return
    
    # Відкриваємо відео
    current_video = 0
    cap = cv.VideoCapture(VIDEO_FILES[current_video] if os.path.exists(VIDEO_FILES[current_video]) else 0)
    
    fps_counter = 0
    fps_start_time = time.time()
    fps_display = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Переключаємося на наступне відео
            current_video = (current_video + 1) % len(VIDEO_FILES)
            cap.release()
            cap = cv.VideoCapture(VIDEO_FILES[current_video])
            continue
        
        # Змінюємо розмір кадру
        frame = cv.resize(frame, (640, 480))
        
        # Детекція
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Малюємо результати
        if results and len(results) > 0:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
        
        # Рахуємо FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Додаємо FPS до кадру
        cv.putText(annotated_frame, f"PyTorch FPS: {fps_display}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv.imshow("PyTorch YOLO Demo", annotated_frame)
        
        # Обробка клавіш
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('v'):
            current_video = (current_video + 1) % len(VIDEO_FILES)
            cap.release()
            cap = cv.VideoCapture(VIDEO_FILES[current_video])
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    test_pytorch_vs_tflite()
    
    # Опціонально запускаємо демонстрацію
    response = input("\n🎬 Запустити демонстрацію PyTorch? (y/n): ")
    if response.lower() == 'y':
        run_pytorch_demo()
