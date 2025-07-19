#!/usr/bin/env python3
"""
Скрипт для тестування різних методів підключення до Raspberry Pi камери
Допомагає визначити, який метод працює на вашій системі
"""

import cv2 as cv
import subprocess
import os
import time

def test_rpicam_hello():
    """Тестує базову роботу rpicam-hello"""
    print("\n🧪 Тест 1: rpicam-hello")
    try:
        result = subprocess.run(
            ["rpicam-hello", "--timeout", "2000"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✅ rpicam-hello працює")
            return True
        else:
            print(f"❌ rpicam-hello помилка: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ rpicam-hello не знайдено. Встановіть: sudo apt install camera-utils")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️ rpicam-hello зависло")
        return False

def test_dev_video():
    """Тестує /dev/video0"""
    print("\n🧪 Тест 2: /dev/video0")
    if os.path.exists("/dev/video0"):
        print("✅ /dev/video0 існує")
        cap = cv.VideoCapture("/dev/video0")
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ /dev/video0 працює")
                return True
            else:
                print("❌ /dev/video0 не може зчитати кадр")
                return False
        else:
            print("❌ /dev/video0 не може відкритися")
            return False
    else:
        print("❌ /dev/video0 не існує")
        return False

def test_gstreamer():
    """Тестує GStreamer"""
    print("\n🧪 Тест 3: GStreamer")
    try:
        cap = cv.VideoCapture(
            "v4l2src device=/dev/video0 ! videoconvert ! appsink",
            cv.CAP_GSTREAMER
        )
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ GStreamer працює")
                return True
            else:
                print("❌ GStreamer не може зчитати кадр")
                return False
        else:
            print("❌ GStreamer не може відкритися")
            return False
    except Exception as e:
        print(f"❌ GStreamer помилка: {e}")
        return False

def test_rpicam_vid_pipe():
    """Тестує rpicam-vid через named pipe"""
    print("\n🧪 Тест 4: rpicam-vid через named pipe")
    pipe_path = "/tmp/test_rpicam_pipe"
    
    try:
        # Видаляємо старий pipe
        if os.path.exists(pipe_path):
            os.unlink(pipe_path)
        
        # Створюємо named pipe
        os.mkfifo(pipe_path)
        print(f"📡 Створено pipe: {pipe_path}")
        
        # Запускаємо rpicam-vid
        cmd = [
            "rpicam-vid",
            "--timeout", "5000",  # 5 секунд
            "--width", "640",
            "--height", "480",
            "--framerate", "15",
            "--output", pipe_path,
            "--codec", "mjpeg",
            "--inline"
        ]
        
        print(f"🚀 Запускаємо: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Чекаємо трохи
        time.sleep(2)
        
        # Спробуємо відкрити pipe
        cap = cv.VideoCapture(pipe_path)
        success = False
        
        if cap.isOpened():
            print("✅ Pipe відкритий, читаємо кадри...")
            for i in range(10):  # Спробуємо прочитати 10 кадрів
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Кадр {i+1} прочитано")
                    success = True
                    break
                time.sleep(0.1)
        
        cap.release()
        process.terminate()
        process.wait()
        
        # Очищуємо
        if os.path.exists(pipe_path):
            os.unlink(pipe_path)
        
        if success:
            print("✅ rpicam-vid через named pipe працює")
            return True
        else:
            print("❌ rpicam-vid через named pipe не працює")
            return False
            
    except FileNotFoundError:
        print("❌ rpicam-vid не знайдено")
        return False
    except Exception as e:
        print(f"❌ Помилка rpicam-vid: {e}")
        return False

def main():
    """Головна функція тестування"""
    print("🔍 Тестування методів підключення до Raspberry Pi камери")
    print("=" * 60)
    
    results = {}
    
    # Тестуємо всі методи
    results["rpicam-hello"] = test_rpicam_hello()
    results["/dev/video0"] = test_dev_video()
    results["gstreamer"] = test_gstreamer()
    results["rpicam-vid pipe"] = test_rpicam_vid_pipe()
    
    # Виводимо результати
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТИ ТЕСТУВАННЯ:")
    print("=" * 60)
    
    working_methods = []
    for method, success in results.items():
        status = "✅ ПРАЦЮЄ" if success else "❌ НЕ ПРАЦЮЄ"
        print(f"{method:20} : {status}")
        if success:
            working_methods.append(method)
    
    print("\n💡 РЕКОМЕНДАЦІЇ:")
    if working_methods:
        print(f"✅ Працюючі методи: {', '.join(working_methods)}")
        if "rpicam-vid pipe" in working_methods:
            print("🎯 Рекомендуємо використовувати rpicam-vid через named pipe")
        elif "/dev/video0" in working_methods:
            print("🎯 Рекомендуємо використовувати /dev/video0")
        elif "gstreamer" in working_methods:
            print("🎯 Рекомендуємо використовувати GStreamer")
    else:
        print("❌ Жоден метод не працює!")
        print("💡 Перевірте:")
        print("   1. Камера підключена та увімкнена")
        print("   2. sudo raspi-config -> Interface Options -> Camera -> Enable")
        print("   3. Перезавантажте Raspberry Pi")
        print("   4. Встановіть: sudo apt install camera-utils")

if __name__ == "__main__":
    main()
