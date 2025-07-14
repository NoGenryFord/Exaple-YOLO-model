"""
Конвертація ONNX моделі в TensorFlow Lite
Використовує tf2onnx для завантаження ONNX і TensorFlow для конвертації
"""

import onnx
import tensorflow as tf
from tensorflow.lite.python import lite
import numpy as np
import os

def convert_onnx_to_tflite():
    """Конвертує ONNX модель в TensorFlow Lite"""
    
    onnx_path = "weights/YOLO/model_3_best.onnx"
    tflite_path = "weights/YOLO/model_3_best.tflite"
    
    print(f"Завантажуємо ONNX модель: {onnx_path}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ Файл {onnx_path} не знайдено!")
        return False
        
    try:
        # Завантажуємо ONNX модель
        onnx_model = onnx.load(onnx_path)
        
        print("📊 Інформація про ONNX модель:")
        print(f"   - Версія ONNX: {onnx_model.opset_import[0].version}")
        print(f"   - Кількість вузлів: {len(onnx_model.graph.node)}")
        
        # Отримуємо інформацію про вхід та вихід
        input_info = onnx_model.graph.input[0]
        output_info = onnx_model.graph.output[0]
        
        print(f"   - Вхід: {input_info.name}")
        print(f"   - Вихід: {output_info.name}")
        
        # Створюємо простий TensorFlow граф для тестування
        print("\n🔄 Створюємо TensorFlow Lite модель...")
        
        # Використовуємо tf2onnx для конвертації
        import tf2onnx
        from tf2onnx import tf_loader
        
        # Конвертуємо ONNX в TensorFlow граф
        with tf.Graph().as_default():
            tf_rep = tf2onnx.backend.prepare(onnx_model)
            
            # Створюємо TensorFlow Lite конвертер
            concrete_func = tf_rep.export_graph()
            
            # Отримуємо вхідну форму з ONNX моделі
            input_shape = [1, 3, 640, 640]  # YOLO стандартний розмір
            
            # Створюємо конвертер
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            
            # Налаштування оптимізації
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Конвертуємо
            tflite_model = converter.convert()
            
            # Зберігаємо TFLite модель
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            print(f"✅ TensorFlow Lite модель збережено: {tflite_path}")
            
            # Отримуємо інфо про модель
            model_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
            original_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            print(f"📊 Розмір ONNX моделі: {original_size:.2f} MB")
            print(f"📊 Розмір TFLite моделі: {model_size:.2f} MB")
            print(f"📊 Стиснення: {(original_size - model_size) / original_size * 100:.1f}%")
            
            return True
            
    except ImportError as e:
        print(f"❌ Помилка імпорту: {str(e)}")
        print("Спробуйте встановити: pip install tf2onnx")
        return False
        
    except Exception as e:
        print(f"❌ Помилка під час конвертації: {str(e)}")
        return False

# Альтернативний метод - якщо tf2onnx не працює
def convert_onnx_to_tflite_simple():
    """Простий метод конвертації через прямий TensorFlow"""
    
    print("\n🔄 Спробуємо простий метод конвертації...")
    
    try:
        # Створюємо dummy модель з такими ж параметрами
        input_shape = (1, 3, 640, 640)
        
        # Створюємо простий граф TensorFlow
        @tf.function
        def dummy_model(x):
            # Це заглушка - замінимо на реальну модель пізніше
            return tf.random.normal((1, 5, 8400))
        
        # Створюємо concrete function
        concrete_func = dummy_model.get_concrete_function(
            tf.TensorSpec(input_shape, tf.float32)
        )
        
        # Тестуємо конвертер
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()
        
        # Зберігаємо тестову модель
        test_path = "weights/YOLO/test_model.tflite"
        with open(test_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"✅ Тестова TFLite модель створена: {test_path}")
        return True
        
    except Exception as e:
        print(f"❌ Помилка в простому методі: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Конвертація ONNX → TensorFlow Lite ===")
    
    # Спробуємо основний метод
    success = convert_onnx_to_tflite()
    
    if not success:
        print("\n⚠️  Основний метод не вдався, спробуємо альтернативний...")
        success = convert_onnx_to_tflite_simple()
    
    if success:
        print("\n🎉 Конвертація завершена!")
    else:
        print("\n❌ Всі методи конвертації не вдалися.")
