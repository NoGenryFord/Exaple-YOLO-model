from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2 as cv
import numpy as np
import time
import torch
# Перевірка наявності GPU

    


# OpenCV performance settings
cv.setUseOptimized(True)  # Використання оптимізованих функцій OpenCV
cv.setNumThreads(4)  # Кількість потоків для OpenCV

# Модель і трекер
model = YOLO('military_test/weights/model_3_best.pt')  # Завантаження моделі YOLOv8
model.conf = 0.8  # Впевненість детектора

# Вхідне відео
video_path = 'military_test/test/apc1.mp4'  # Шлях до відео
VIDEO_LIFE = 0 #Use for life_translation tracking (camera)
video = cv.VideoCapture(video_path)
# Перевірка відкриття відео
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Змінні для трекінгу
tracks = []  # Глобальна змінна для треків


# Явно створюємо вікно і додаємо callback
cv.namedWindow('Frame')

# Постійні змінні
# Стандартизований розмір кадру
STANDARD_WIDTH = 640
STANDARD_HEIGHT = 480


MAX_FPS = 30 # Максимальна частота кадрів
YOLO_SKIP_FRAMES = 3 # Кількість пропущених кадрів для YOLO

# Функція для зміни розміру кадру
def resize_frame(frame, width=STANDARD_WIDTH, height=STANDARD_HEIGHT):
    return cv.resize(frame, (width, height))
# Функція для зміни кольорового простору на сірий
def convert_to_gray(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Зміна кольорового простору на сірий
    gray_3ch = cv.merge([gray, gray, gray]) # Зміна кольорового простору на 3 канали
    return gray_3ch

is_gray_mode = False # Змінна для перевірки кольорового простору

# Змінні для частоти кадрів
frame_start_time = time.time() # Час початку кадру
frame_count = 0  # Ініціалізація лічильника кадрів
fps = 0  # Ініціалізація змінної для FPS
last_time = time.time()  # Ініціалізація часу для підрахунку FPS

# Основний цикл
while True:

    start_time = time.time() # Час початку кадру
    # Читання кадру з відео
    ret, frame = video.read()

    # Перевірка на наявність сигналу
    if not ret:
        # Якщо немає сигналу,тоді чорний екран із повідомленням
        frame = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH, 3), dtype=np.uint8) # Чорний екран
        # Відображення повідомлення
        cv.putText(frame, "No Signal", (STANDARD_WIDTH // 2 - 100, STANDARD_HEIGHT // 2), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.putText(frame, "Waiting for video...", (STANDARD_WIDTH // 2 - 150, STANDARD_HEIGHT // 2 + 40), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.imshow('Frame', frame)

    frame_count += 1 # Лічильник кадрів

    # Зміна розміру кадру
    frame = resize_frame(frame)

    # Зміна кольорового простору на сірий, якщо потрібно
    if is_gray_mode:
        frame = convert_to_gray(frame)

    # Відображення розміру кадру
    # Отримання розміру кадру
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    # Відображення розміру кадру
    current_time = time.time() # Час поточного кадру
    if current_time - last_time >= 1.0:
        fps = frame_count
        frame_count = 0 # Скидання лічильника кадрів
        last_time = current_time # Оновлення часу останнього кадру
    cv.putText(frame, f"FPS: {fps}", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Відображення частоти кадрів

    # Отримання об'єктів від моделі
    if frame_count % YOLO_SKIP_FRAMES == 0:  # Пропуск кадрів для YOLO
        results = model(frame)[0].boxes
        if results:  # Перевірка, чи є результати
            for b in results:
                cords = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, cords)
                conf = round(b.conf[0].item(), 2)

        # Відображення треків
            color = (0, 255, 0)  # Колір треку
            box_width = x2 - x1
            box_height = y2 - y1
            shrink_factor = 0.7  # Фактор зменшення
            new_width = int(box_width * shrink_factor)
            new_height = int(box_height * shrink_factor)
            # Обчислення центру прямокутника
            x_center = x1 + box_width // 2
            y_center = y1 + box_height // 2

            x1_new = x_center - new_width // 2
            y1_new = y_center + box_height // 2
            x2_new = x_center + new_width // 2
            y2_new = y_center - new_height // 2
        
            # Відображення треків
            cv.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), color, 2)
            cv.circle(frame, (x_center, y_center), 2, (0, 0, 255), -1)  # Відображення центру треку
            cv.putText(frame, "Target", (x1_new, y1_new - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            

        
    # Відображення тексту для підказок
    # Режимоння кольорового простору
    if is_gray_mode:
        cv.putText(frame, "Gray mode ON", (600, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Чорна обведення
        cv.putText(frame, "Gray mode ON", (600, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Червоний текст
    else:
        cv.putText(frame, "Gray mode OFF", (600, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Чорна обведення
        cv.putText(frame, "Gray mode OFF", (600, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Червоний текст           
    # Вказівка на вихід
    cv.putText(frame, "Press 'ESC' to exit", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Чорна обведення
    cv.putText(frame, "Press 'ESC' to exit", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Червоний текст
    # Вказівка на вибір об'єкта
    cv.putText(frame, "Click on object to select", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Чорна обведення
    cv.putText(frame, "Click on object to select", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Червоний текст
     # Вказівка на скидання вибору
    cv.putText(frame, "Press 'r' to reset selection", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Чорна обведення
    cv.putText(frame, "Press 'r' to reset selection", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Червоний текст
    # Вказівка на перемикання кольорового простору
    cv.putText(frame, "Press 'g' to toggle gray mode", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Чорна обведення
    cv.putText(frame, "Press 'g' to toggle gray mode", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Червоний текст
    # Вказівка на перемикання камери
    cv.putText(frame, "Press 'c' to switch to camera", (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Чорна обведення
    cv.putText(frame, "Press 'c' to switch to camera", (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Червоний текст

    # Відображення кадру
    cv.imshow('Frame', frame)

    # Відображення часу затримки
    frame_end_time = time.time() # Час закінчення кадру
    elapsed_time = frame_end_time - frame_start_time # Час затримки
    frame_start_time = frame_end_time # Оновлення часу початку кадру
    # Затримка для досягнення бажаної частоти кадрів
    target_time_per_frame = 1.0 / MAX_FPS
    if elapsed_time < target_time_per_frame:
        time.sleep(target_time_per_frame - elapsed_time)

    # Налаштування клавіш
    key = cv.waitKey(10) # Затримка для відображення кадру
    if key == 27:  # Вихід при ESC
        break
    elif key == ord('g'): # Перемикання кольорового простору на сірий
        is_gray_mode = not is_gray_mode
        if is_gray_mode:
            print("Перемикання на сірий")
        else:
            print("Перемикання на кольоровий")
    elif key == ord('c'): # Перемикання на камеру
        video.release()
        video = cv.VideoCapture(0) # Відкриття камери
        print("Відкрито камеру")
        if not video.isOpened():
            print("Error: Could not open camera.")
            video.release()
        continue

video.release()
cv.destroyAllWindows()

if __name__ == "__main__":
    pass
# End of file