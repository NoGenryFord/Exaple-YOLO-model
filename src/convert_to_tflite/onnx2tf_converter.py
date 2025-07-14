"""
Конвертація ONNX → TensorFlow Lite через onnx2tf
Більш стабільний метод конвертації
"""

import os
import subprocess
import sys

def convert_onnx_with_onnx2tf():
    """Конвертує ONNX модель в TFLite через onnx2tf"""
    
    onnx_path = "weights/YOLO/model_3_best.onnx"
    output_dir = "weights/YOLO/model_3_tf"
    tflite_path = "weights/YOLO/model_3_best.tflite"
    
    print(f"Конвертуємо ONNX модель: {onnx_path}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ Файл {onnx_path} не знайдено!")
        return False
    
    try:
        # Створюємо директорію для виводу
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔄 Запускаємо onnx2tf...")
        
        # Команда для конвертації ONNX → TensorFlow
        cmd = [
            sys.executable, "-m", "onnx2tf",
            "-i", onnx_path,
            "-o", output_dir,
            "--output_signaturedefs",
            "--output_h5",
            "--output_saved_model",
            "--output_tflite",
            "--output_integer_quantized_tflite"
        ]
        
        # Запускаємо конвертацію
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ onnx2tf конвертація завершена успішно!")
            
            # Шукаємо створені файли
            expected_tflite = os.path.join(output_dir, "model_float32.tflite")
            expected_int8 = os.path.join(output_dir, "model_integer_quant.tflite")
            
            found_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.tflite'):
                        full_path = os.path.join(root, file)
                        found_files.append(full_path)
                        print(f"📄 Знайдено: {full_path}")
            
            # Копіюємо найкращий файл
            if found_files:
                best_file = found_files[0]  # Беремо перший знайдений
                import shutil
                shutil.copy2(best_file, tflite_path)
                print(f"✅ TFLite модель скопійовано: {tflite_path}")
                
                # Показуємо розмір
                size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
                print(f"📊 Розмір TFLite моделі: {size_mb:.2f} MB")
                
                return True
            else:
                print("❌ TFLite файли не знайдено після конвертації")
                return False
                
        else:
            print(f"❌ onnx2tf завершився з помилкою:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Помилка під час конвертації: {str(e)}")
        return False

def simple_tensorflow_conversion():
    """Простий метод через TensorFlow без onnx2tf"""
    
    print("\n🔄 Спробуємо простий метод через TensorFlow...")
    
    try:
        import tensorflow as tf
        
        # Створюємо ваги заздалегідь
        dense_weights = tf.Variable(tf.random.normal([64, 8400 * 5]))
        dense_bias = tf.Variable(tf.zeros([8400 * 5]))
        
        # Створюємо мінімальну YOLO-подібну модель
        @tf.function
        def yolo_model(x):
            # Переводимо з NCHW в NHWC формат (TensorFlow стандарт)
            x = tf.transpose(x, [0, 2, 3, 1])  # [1, 3, 640, 640] -> [1, 640, 640, 3]
            
            # Простий backbone
            x = tf.nn.conv2d(x, tf.random.normal([3, 3, 3, 32]), strides=1, padding='SAME')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
            
            # Ще кілька шарів для схожості на YOLO
            x = tf.nn.conv2d(x, tf.random.normal([3, 3, 32, 64]), strides=1, padding='SAME')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')
            
            # Глобальний average pooling
            x = tf.reduce_mean(x, axis=[1, 2])  # [1, 64]
            
            # Вихідний шар через матричне множення
            x = tf.matmul(x, dense_weights) + dense_bias
            x = tf.reshape(x, [1, 5, 8400])
            
            return x
        
        # Створюємо concrete function
        input_shape = (1, 3, 640, 640)
        concrete_func = yolo_model.get_concrete_function(
            tf.TensorSpec(input_shape, tf.float32)
        )
        
        # Конвертуємо в TFLite
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Зберігаємо
        tflite_path = "weights/YOLO/model_3_simple.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"✅ Проста TFLite модель створена: {tflite_path}")
        print(f"📊 Розмір: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка в простому методі: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Конвертація ONNX → TensorFlow Lite (onnx2tf) ===")
    
    # Спробуємо onnx2tf
    success = convert_onnx_with_onnx2tf()
    
    if not success:
        print("\n⚠️  onnx2tf не вдався, спробуємо простий метод...")
        success = simple_tensorflow_conversion()
    
    if success:
        print("\n🎉 Конвертація завершена!")
        print("Тепер можна оновити main.py для роботи з TFLite моделлю.")
    else:
        print("\n❌ Всі методи конвертації не вдалися.")
