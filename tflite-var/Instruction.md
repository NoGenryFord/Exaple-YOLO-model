# Інструкція: Конвертація YOLOv8n (.pt) у TensorFlow Lite (.tflite) для Raspberry Pi

## 1. Вимоги

- Python 3.10
- Створіть та активуйте віртуальне середовище:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate  # Windows
  # або
  source .venv/bin/activate  # Linux/Mac
  ```

- Встановіть залежності:
  ```bash
  pip install ultralytics==8.3.160 torch==2.7.1 onnx==1.17.0 onnxruntime==1.22.0 onnx-tf==1.10.0 tensorflow==2.15.0 keras==2.15.0 tensorflow-addons==0.22.0 numpy==1.26.0 tensorflow-probability==0.23.0
  ```

## 2. Запуск конвертації

- Помістіть ваш `.pt` файл у папку `models/` (наприклад, `models/model_3_best.pt`).
- Запустіть скрипт:
  ```bash
  python convert_yolo.py --weights models/model_3_best.pt --quantize
  ```

## 3. Що відбувається

1. **Експорт у ONNX:**  
   Модель експортується у формат ONNX (`models/model_3_best.onnx`).

2. **ONNX → TensorFlow SavedModel:**  
   ONNX-модель конвертується у TensorFlow SavedModel (`yolov8n_saved_model/`).

3. **SavedModel → TFLite:**  
   SavedModel конвертується у TFLite (`yolov8n.tflite`).  
   Якщо використовується `--quantize`, застосовується пост-тренувальна квантизація.

> **Увага:**  
> Якщо у логах є попередження про Select TF ops/FlexConv2D — це нормально для складних моделей. Для запуску на Raspberry Pi використовуйте TensorFlow Lite з підтримкою Select TF ops.

## 4. Використання на Raspberry Pi

- Передайте файл `yolov8n.tflite` на Raspberry Pi.
- Встановіть tflite-runtime або tensorflow:
  ```bash
  pip install tflite-runtime
  # або
  pip install tensorflow
  ```
- Використовуйте TFLite Interpreter для inference.

---


