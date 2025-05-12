# Drone-AI

## ВСТАНОВЛЕННЯ

1. **Розпакуйте архів**
   - Розпакуйте архів проекту у зручне місце на вашому комп'ютері.
2. **Створіть та активуйте віртуальне середовище**
    - Відкрийте термінал (або командний рядок) і перейдіть у папку проекту:
   ```bash
     cd <шлях_до_папки_проекту>
     ```
    
    - Створіть віртуальне середовище:
    ```bash
     python -m venv venv
     ```
    - Активуйте віртуальне середовище:
    - Для Windows:
       ```bash
       venv\Scripts\activate
       ```
    - Для Linux/Mac:
       ```bash
       source venv/bin/activate
       ```

3. **Встановлення залежностей**
    pip install -r requirements.txt

4. **Перевірка наявності моделі та відео**
    Переконайтеся, що файл моделі знаходиться за шляхом:
    # military_test/weights/best_new_a0.1.pt
    Вхідне відео має бути за шляхом
    # military_test/test/tank1.mp4


## ЗАПУСК
1. Запустіть основний скрипт:
    # python military_test/test001.py

2. Для виходу натисніть клавішу ESC.

## СТРУКТУРА ПРОЕКТУ
    military_test/weights/ — зберігає модель YOLOv8.

    military_test/test/ — зберігає вхідні відеофайли.
    
    military_test/test001.py — основний скрипт для запуску проекту.

## ЗАЛЕЖНОСТІ
    Усі залежності вказані у файлі requirements.txt. Для встановлення використовуйте:
    pip install -r requirements.txt
    або
    pip install -r [requirements.txt](http://_vscodecontentref_/1)