from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2 as cv
import numpy as np


# Модель і трекер
model = YOLO('military_test/weights/model_3_best.pt')  # Завантаження моделі YOLOv8
model.conf = 0.4  # Впевненість детектора
# Задаємо параметри трекера
tracker = DeepSort(max_age=30, n_init=3)  # Збільшено max_age для кращого супроводу

# Вхідне відео
video_path = 'military_test/test/tank5.mp4'
VIDEO_LIFE = 0 #Use for life_translation tracking (camera)
video = cv.VideoCapture(video_path)
# Перевірка відкриття відео
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Змінні для трекінгу
selected_track_id = None
tracks = []  # Глобальна змінна для треків

# Функція вибору об'єкта
def select_object(event, x, y, flags, param):
    global selected_track_id
    if event == cv.EVENT_LBUTTONDOWN:
        for track in tracks:
            x1, y1, x2, y2 = track.to_tlbr()
            if x1 < x < x2 and y1 < y < y2:
                selected_track_id = "Target"  # Вибрано об'єкт
                print("Вибрано об'єкт: Target")
                break

# Явно створюємо вікно і додаємо callback
cv.namedWindow('Frame')
cv.setMouseCallback('Frame', select_object)

# Стандартизований розмір кадру
STANDARD_WIDTH = 800
STANDARD_HEIGHT = 600

# Функція для зміни розміру кадру
def resize_frame(frame, width=STANDARD_WIDTH, height=STANDARD_HEIGHT):
    return cv.resize(frame, (width, height))
# Функція для зміни кольорового простору на сірий
def convert_to_gray(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Зміна кольорового простору на сірий
    gray_3ch = cv.merge([gray, gray, gray]) # Зміна кольорового простору на 3 канали
    return gray_3ch

is_gray_mode = False # Змінна для перевірки кольорового простору


# Основний цикл
while True:
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

    # Зміна розміру кадру
    frame = resize_frame(frame)

    # Зміна кольорового простору на сірий, якщо потрібно
    if is_gray_mode:
        frame = convert_to_gray(frame)

    # Відображення розміру кадру
    # Отримання розміру кадру
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    # Отримання об'єктів від моделі
    results = []
    boxes = model(frame)[0].boxes
    for b in boxes:
        cords = b.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, cords)
        conf = round(b.conf[0].item(), 2)  # Впевненість детектора
        obj_id = hash((x1, y1, x2, y2))  # Унікальний ID на основі координат
        results.append([[x1, y1, x2, y2], conf, obj_id])

    # Оновлення трекера
    tracks = tracker.update_tracks(results, frame=frame)

    # Відображення треків
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        color = (0, 255, 0) if track_id != selected_track_id else (242, 78, 43)
        # Зменшення розміру прямокутника для відображення
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
        color = (0, 255, 0) if track_id != selected_track_id else (242, 78, 43)
        cv.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), color, 2)
        
        # Відображення центру треку
        cv.circle(frame, (x_center, y_center), 5, (0,0,255), -1)  # Відображення центру треку

        # Відображення ID треку
        cv.putText(frame, "Target", (x1_new, y1_new - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Додано відображення впевненості
        #conf = track.get_det_conf()  # Отримуємо впевненість трекера
        #if conf is not None:
           # cv.putText(frame, f"ID: {track_id} ({conf:.2f})", (x1, y1 - 10),
                       #cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Вибраний об'єкт
    if selected_track_id is not None:
        cv.putText(frame, f"Selected: {selected_track_id}",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
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
    key = cv.waitKey(10) # Затримка для відображення кадру
    if key == 27:  # Вихід при ESC
        break
    elif key == ord('g'): # Перемикання кольорового простору на сірий
        is_gray_mode = not is_gray_mode
        if is_gray_mode:
            print("Перемикання на сірий")
        else:
            print("Перемикання на кольоровий")
    elif key == ord('r'): # Скидання вибору
        selected_track_id = None
        print("Скинуто вибір об'єкта")
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