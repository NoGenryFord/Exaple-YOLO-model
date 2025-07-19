#!/usr/bin/env python3
"""
Простий тест TFLite моделі для перевірки детекції
"""

import tensorflow as tf
import cv2 as cv
import numpy as np
import os

# Параметри
TFLITE_MODEL_PATH = "weights/YOLO/model_3_simple.tflite"
TEST_IMAGE_PATH = "data/sample_battle_1.mp4"  # Візьмемо перший кадр з відео

def test_tflite_model():
    """Тестуємо TFLite модель на одному кадрі"""
    print("🧪 Тестування TFLite моделі...")
    
    # Завантажуємо модель
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"❌ TFLite модель не знайдена: {TFLITE_MODEL_PATH}")
        return
    
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Отримуємо деталі тензорів
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"📊 Вхідна форма: {input_details[0]['shape']}")
    print(f"📊 Вихідна форма: {output_details[0]['shape']}")
    
    # Завантажуємо тестовий кадр
    cap = cv.VideoCapture(TEST_IMAGE_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Не вдалося зчитати кадр")
        return
    
    print(f"📸 Розмір кадру: {frame.shape}")
    
    # Препроцесинг
    input_shape = input_details[0]['shape']
    model_height, model_width = input_shape[2], input_shape[3]
    
    # Змінюємо розмір кадру
    resized = cv.resize(frame, (model_width, model_height))
    
    # Конвертуємо BGR в RGB
    rgb_frame = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    
    # Нормалізуємо значення пікселів [0-255] -> [0-1]
    normalized = rgb_frame.astype(np.float32) / 255.0
    
    # Транспонуємо в NCHW: [H, W, C] -> [C, H, W] і додаємо batch dimension
    input_data = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)
    
    print(f"🔧 Форма вхідних даних: {input_data.shape}")
    
    # Виконуємо інференс
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Отримуємо результат
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"🔧 Форма вихідних даних: {output_data.shape}")
    print(f"🔧 Мін/макс значення виходу: {np.min(output_data):.6f} / {np.max(output_data):.6f}")
    
    # Аналізуємо вихід
    predictions = output_data[0]  # [5, 8400]
    print(f"🔧 Форма predictions: {predictions.shape}")
    
    # Транспонуємо для зручності: [5, 8400] -> [8400, 5]
    predictions = predictions.T
    
    # Рахуємо кількість детекцій з різними порогами
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        count = np.sum(predictions[:, 4] > threshold)  # confidence в 5-му стовпці
        print(f"📊 Детекції з confidence > {threshold}: {count}")
    
    # Показуємо топ-10 детекцій
    top_indices = np.argsort(predictions[:, 4])[-10:][::-1]
    print("\n🔝 Топ-10 детекцій:")
    for i, idx in enumerate(top_indices):
        x_center, y_center, width, height, confidence = predictions[idx]
        print(f"   {i+1}. Confidence: {confidence:.6f}, Center: ({x_center:.3f}, {y_center:.3f}), Size: {width:.3f}x{height:.3f}")
    
    print("\n✅ Тест завершено!")

if __name__ == "__main__":
    test_tflite_model()
