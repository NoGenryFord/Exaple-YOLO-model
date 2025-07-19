#!/usr/bin/env python3
"""
Альтернативний метод підключення до Raspberry Pi камери через HTTP streaming
Цей скрипт запускає rpicam-vid в режимі HTTP сервера та підключається до нього
"""

import cv2 as cv
import subprocess
import time
import threading
import os
import signal

class RaspberryPiCamera:
    def __init__(self, width=640, height=480, fps=30, port=8080):
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.process = None
        self.cap = None
        
    def start_streaming(self):
        """Запускає HTTP streaming з rpicam-vid"""
        try:
            # Команда для запуску rpicam-vid з HTTP streaming
            cmd = [
                "rpicam-vid",
                "--timeout", "0",  # Безкінечна зйомка
                "--width", str(self.width),
                "--height", str(self.height),
                "--framerate", str(self.fps),
                "--listen", "-o", f"tcp://0.0.0.0:{self.port}",
                "--codec", "mjpeg",
                "--inline"
            ]
            
            print(f"🚀 Запускаємо HTTP streaming: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Чекаємо поки сервер запуститься
            time.sleep(3)
            
            # Підключаємося до HTTP потоку
            stream_url = f"tcp://127.0.0.1:{self.port}"
            self.cap = cv.VideoCapture(stream_url)
            
            if self.cap.isOpened():
                print(f"✅ Підключено до HTTP stream: {stream_url}")
                return True
            else:
                print("❌ Не вдалося підключитися до HTTP stream")
                self.cleanup()
                return False
                
        except Exception as e:
            print(f"❌ Помилка запуску HTTP streaming: {e}")
            self.cleanup()
            return False
    
    def read(self):
        """Читає кадр з камери"""
        if self.cap:
            return self.cap.read()
        return False, None
    
    def isOpened(self):
        """Перевіряє, чи камера відкрита"""
        return self.cap and self.cap.isOpened()
    
    def release(self):
        """Звільняє ресурси"""
        self.cleanup()
    
    def cleanup(self):
        """Очищає всі ресурси"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
            self.process = None

def test_pi_camera():
    """Тестова функція для перевірки роботи камери"""
    print("🧪 Тестування Raspberry Pi камери...")
    
    pi_cam = RaspberryPiCamera(width=640, height=480, fps=30)
    
    if not pi_cam.start_streaming():
        print("❌ Не вдалося запустити камеру")
        return
    
    print("📹 Натисніть ESC для виходу")
    
    while True:
        ret, frame = pi_cam.read()
        if ret:
            cv.imshow("Raspberry Pi Camera", frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        else:
            print("❌ Не вдалося зчитати кадр")
            break
    
    pi_cam.cleanup()
    cv.destroyAllWindows()
    print("✅ Тест завершено")

if __name__ == "__main__":
    test_pi_camera()
