import cv2

def check_gstreamer_support():
    """
    Перевіряє, чи OpenCV підтримує GStreamer.
    """
    build_info = cv2.getBuildInformation()
    if "GStreamer" in build_info and "YES" in build_info:
        print("OpenCV підтримує GStreamer!")
    else:
        print("OpenCV не підтримує GStreamer. Перевірте налаштування OpenCV.")

def test_gstreamer_pipeline():
    """
    Тестує GStreamer-пайплайн для роботи з камерою.
    """
    pipeline = "libcamerasrc ! videoconvert ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Помилка: Не вдалося відкрити GStreamer-пайплайн.")
        return

    print("GStreamer-пайплайн успішно відкрито. Натисніть 'ESC' для виходу.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не вдалося отримати кадр.")
            break

        cv2.imshow("GStreamer Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Натисніть ESC для виходу
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Перевірка підтримки GStreamer в OpenCV...")
    check_gstreamer_support()

    print("\nТестування GStreamer-пайплайну...")
    test_gstreamer_pipeline()