"""
Експорт YOLO моделі у TensorFlow SavedModel, а потім в TFLite
Обходимо проблеми з ai-edge-litert
"""

from ultralytics import YOLO
import tensorflow as tf
import os

def convert_pt_to_savedmodel():
    """Конвертує PyTorch модель YOLO в TensorFlow SavedModel"""
    
    pt_model_path = "weights/YOLO/model_3_best.pt"
    
    print(f"Завантажуємо PyTorch модель: {pt_model_path}")
    
    if not os.path.exists(pt_model_path):
        print(f"❌ Файл {pt_model_path} не знайдено!")
        return None
        
    try:
        # Завантажуємо модель
        model = YOLO(pt_model_path)
        
        print("🔄 Експортуємо модель в TensorFlow SavedModel...")
        
        # Експорт у SavedModel
        saved_model_path = model.export(format='saved_model')
        
        print(f"✅ SavedModel збережено: {saved_model_path}")
        return saved_model_path
        
    except Exception as e:
        print(f"❌ Помилка під час експорту SavedModel: {str(e)}")
        return None

def convert_savedmodel_to_tflite(saved_model_path):
    """Конвертує SavedModel в TensorFlow Lite"""
    
    if not saved_model_path or not os.path.exists(saved_model_path):
        print(f"❌ SavedModel не знайдено: {saved_model_path}")
        return False
        
    try:
        print("🔄 Конвертуємо SavedModel в TensorFlow Lite...")
        
        # Створюємо TFLite конвертер
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # Налаштування оптимізації
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Конвертуємо
        tflite_model = converter.convert()
        
        # Зберігаємо TFLite модель
        tflite_path = "weights/YOLO/model_3_best.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"✅ TensorFlow Lite модель збережено: {tflite_path}")
        
        # Отримуємо інфо про модель
        model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        print(f"📊 Розмір TFLite моделі: {model_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка під час конвертації в TFLite: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Конвертація PyTorch → SavedModel → TensorFlow Lite ===")
    
    # Крок 1: PyTorch → SavedModel
    saved_model_path = convert_pt_to_savedmodel()
    
    if saved_model_path:
        # Крок 2: SavedModel → TFLite
        success = convert_savedmodel_to_tflite(saved_model_path)
        
        if success:
            print("\n🎉 Конвертація завершена успішно!")
            print("Тепер можна оновити main.py для роботи з TFLite моделлю.")
        else:
            print("\n❌ Конвертація в TFLite не вдалася.")
    else:
        print("\n❌ Конвертація в SavedModel не вдалася.")
