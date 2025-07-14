"""
Фінальний тест проекту - перевірка всіх компонентів
"""

import os
import sys
import subprocess

def check_file_exists(path, description):
    """Перевіряє існування файлу"""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} не знайдено: {path}")
        return False

def check_model_files():
    """Перевіряє наявність файлів моделей"""
    print("=== Перевірка файлів моделей ===")
    
    files_to_check = [
        ("weights/YOLO/model_3_best.pt", "PyTorch модель"),
        ("weights/YOLO/model_3_best.onnx", "ONNX модель"),
        ("weights/YOLO/model_3_simple.tflite", "TensorFlow Lite модель"),
    ]
    
    all_good = True
    for path, description in files_to_check:
        if not check_file_exists(path, description):
            all_good = False
        else:
            # Показуємо розмір файлу
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   📊 Розмір: {size_mb:.2f} MB")
    
    return all_good

def check_python_files():
    """Перевіряє наявність Python файлів"""
    print("\n=== Перевірка Python файлів ===")
    
    files_to_check = [
        ("main.py", "Основний файл (PyTorch)"),
        ("main_tflite.py", "TensorFlow Lite версія"),
        ("test_tflite.py", "Тест TFLite моделі"),
        ("src/convert_to_tflite/onnx2tf_converter.py", "Конвертер ONNX → TFLite"),
        ("requirements.txt", "Файл залежностей"),
        ("README.md", "Документація"),
    ]
    
    all_good = True
    for path, description in files_to_check:
        if not check_file_exists(path, description):
            all_good = False
    
    return all_good

def test_import_tflite():
    """Тестує імпорт TensorFlow Lite"""
    print("\n=== Тест імпорту TensorFlow Lite ===")
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow імпортовано успішно")
        print(f"   📊 Версія TensorFlow: {tf.__version__}")
        
        # Перевіряємо TFLite
        interpreter = tf.lite.Interpreter("weights/YOLO/model_3_simple.tflite")
        interpreter.allocate_tensors()
        print("✅ TFLite інтерпретатор працює")
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   📊 Вхідна форма: {input_details[0]['shape']}")
        print(f"   📊 Вихідна форма: {output_details[0]['shape']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка імпорту TensorFlow Lite: {e}")
        return False

def test_opencv():
    """Тестує OpenCV"""
    print("\n=== Тест OpenCV ===")
    
    try:
        import cv2 as cv
        print("✅ OpenCV імпортовано успішно")
        print(f"   📊 Версія OpenCV: {cv.__version__}")
        
        # Тестуємо створення зображення
        import numpy as np
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = cv.resize(test_img, (320, 240))
        print("✅ OpenCV операції працюють")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка OpenCV: {e}")
        return False

def test_deep_sort():
    """Тестує Deep Sort"""
    print("\n=== Тест Deep Sort ===")
    
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        deep_sort = DeepSort(max_age=30, n_init=3)
        print("✅ Deep Sort працює")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка Deep Sort: {e}")
        return False

def run_quick_tflite_test():
    """Швидкий тест TFLite моделі"""
    print("\n=== Швидкий тест TFLite моделі ===")
    
    try:
        # Імпортуємо наш клас
        sys.path.append('.')
        from main_tflite import TFLiteYOLO
        
        # Створюємо модель
        yolo_model = TFLiteYOLO("weights/YOLO/model_3_simple.tflite")
        
        # Тестове зображення
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Детекція
        detections = yolo_model.detect(test_image)
        
        print(f"✅ TFLite модель працює! Знайдено {len(detections)} об'єктів")
        
        # Показуємо кілька перших детекцій
        for i, detection in enumerate(detections[:3]):
            x1, y1, x2, y2, conf = detection
            print(f"   Об'єкт {i+1}: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] conf={conf:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка TFLite тесту: {e}")
        return False

def main():
    """Основна функція тестування"""
    print("🧪 Фінальний тест проекту Drone AI v.0.9.1t")
    print("=" * 50)
    
    # Список тестів
    tests = [
        ("Файли моделей", check_model_files),
        ("Python файли", check_python_files),
        ("TensorFlow Lite", test_import_tflite),
        ("OpenCV", test_opencv),
        ("Deep Sort", test_deep_sort),
        ("TFLite модель", run_quick_tflite_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Критична помилка в тесті '{test_name}': {e}")
            results.append((test_name, False))
    
    # Підсумок
    print("\n" + "=" * 50)
    print("🎯 ПІДСУМОК ТЕСТУВАННЯ")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ ПРОЙДЕНО" if result else "❌ НЕ ПРОЙДЕНО"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Результат: {passed}/{total} тестів пройдено")
    
    if passed == total:
        print("🎉 ВСІ ТЕСТИ ПРОЙДЕНО! Проект готовий до використання.")
        print("\n🚀 Для запуску:")
        print("   PyTorch версія: python main.py")
        print("   TFLite версія: python main_tflite.py")
    else:
        print("⚠️  Деякі тести не пройдено. Перевірте помилки вище.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
