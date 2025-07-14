"""
Тестування TFLite моделі на зображенні
Для перевірки роботи без камери
"""

import tensorflow as tf
import cv2 as cv
import numpy as np
import sys
import os

# Додаємо шлях до main_tflite.py
sys.path.append('.')
from main_tflite import TFLiteYOLO, draw_detection

def test_tflite_model():
    """Тестує TFLite модель на тестовому зображенні"""
    
    model_path = "weights/YOLO/model_3_simple.tflite"
    
    print("=== Тестування TFLite моделі ===")
    
    # Завантажуємо модель
    try:
        yolo_model = TFLiteYOLO(model_path)
    except Exception as e:
        print(f"❌ Помилка завантаження моделі: {e}")
        return
    
    # Створюємо тестове зображення
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Додаємо кольорові прямокутники для імітації об'єктів
    cv.rectangle(test_image, (100, 100), (300, 250), (0, 255, 0), -1)  # Зелений
    cv.rectangle(test_image, (400, 200), (550, 350), (255, 0, 0), -1)  # Синій
    cv.rectangle(test_image, (200, 300), (350, 400), (0, 0, 255), -1)  # Червоний
    
    print("🖼️ Створено тестове зображення з об'єктами")
    
    # Тестуємо детекцію
    try:
        print("🔄 Виконуємо детекцію...")
        detections = yolo_model.detect(test_image)
        print(f"✅ Детекція завершена! Знайдено {len(detections)} об'єктів")
        
        # Малюємо детекції
        result_image = test_image.copy()
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf = detection
            print(f"   Об'єкт {i+1}: [{x1}, {y1}, {x2}, {y2}] confidence={conf:.3f}")
            draw_detection(result_image, int(x1), int(y1), int(x2), int(y2), conf)
        
        # Показуємо результат
        cv.imshow("Test Image", test_image)
        cv.imshow("Detection Results", result_image)
        
        print("🎉 Тест завершено! Натисніть будь-яку клавішу для закриття...")
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Помилка під час детекції: {e}")
        import traceback
        traceback.print_exc()

def test_model_performance():
    """Тестує продуктивність моделі"""
    
    model_path = "weights/YOLO/model_3_simple.tflite"
    
    print("\n=== Тест продуктивності ===")
    
    try:
        yolo_model = TFLiteYOLO(model_path)
        
        # Створюємо тестове зображення
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Прогрів моделі
        print("🔥 Прогрів моделі...")
        for _ in range(5):
            yolo_model.detect(test_image)
        
        # Тестуємо швидкість
        num_tests = 20
        import time
        
        start_time = time.time()
        for i in range(num_tests):
            detections = yolo_model.detect(test_image)
            if i % 5 == 0:
                print(f"   Тест {i+1}/{num_tests}...")
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_tests
        fps = 1.0 / avg_time
        
        print(f"📊 Результати тесту продуктивності:")
        print(f"   Всього тестів: {num_tests}")
        print(f"   Загальний час: {total_time:.2f}s")
        print(f"   Середній час інференсу: {avg_time:.3f}s")
        print(f"   Теоретичний FPS: {fps:.1f}")
        
        if fps > 10:
            print("✅ Продуктивність хороша для реального часу!")
        elif fps > 5:
            print("⚠️ Продуктивність прийнятна")
        else:
            print("❌ Продуктивність низька")
            
    except Exception as e:
        print(f"❌ Помилка тесту продуктивності: {e}")

if __name__ == "__main__":
    test_tflite_model()
    test_model_performance()
