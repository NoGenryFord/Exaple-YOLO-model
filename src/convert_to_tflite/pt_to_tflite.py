"""
Прямий експорт YOLOv8 .pt моделі в TensorFlow Lite формат
Використовує ultralytics YOLO для стабільної конвертації
"""

from ultralytics import YOLO
import os

def convert_pt_to_tflite():
    """Конвертує PyTorch модель YOLO в TensorFlow Lite"""
    
    # Шляхи до файлів
    pt_model_path = "weights/YOLO/model_3_best.pt"
    output_dir = "weights/YOLO/"
    
    print(f"Завантажуємо PyTorch модель: {pt_model_path}")
    
    # Перевіряємо, чи існує файл
    if not os.path.exists(pt_model_path):
        print(f"❌ Файл {pt_model_path} не знайдено!")
        return False
        
    try:
        # Завантажуємо модель
        model = YOLO(pt_model_path)
        
        print("🔄 Експортуємо модель в TensorFlow Lite...")
        
        # Експортуємо в TFLite (без INT8 квантизації, щоб уникнути залежностей)
        tflite_path = model.export(
            format='tflite',
            int8=False,  # Вимкнена INT8 квантизація
            optimize=True,
            dynamic=False,
            simplify=True
        )
        
        print(f"✅ TensorFlow Lite модель збережено: {tflite_path}")
        
        # Копіюємо до нашої папки з більш зрозумілою назвою
        target_path = os.path.join(output_dir, "model_3_best.tflite")
        if os.path.exists(tflite_path) and tflite_path != target_path:
            import shutil
            shutil.copy2(tflite_path, target_path)
            print(f"✅ Модель скопійовано до: {target_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Помилка під час експорту: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Конвертація PyTorch → TensorFlow Lite ===")
    success = convert_pt_to_tflite()
    
    if success:
        print("\n🎉 Конвертація завершена успішно!")
        print("Тепер можна оновити main.py для роботи з TFLite моделлю.")
    else:
        print("\n❌ Конвертація не вдалася. Перевірте помилки вище.")
