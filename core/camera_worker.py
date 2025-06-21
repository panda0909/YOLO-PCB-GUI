import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class CameraWorker(QThread):
    frame_captured = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_id=0, fps=1, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.fps = fps
        self.running = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.error_occurred.emit(f"無法開啟攝像頭: {self.camera_id}")
            return
        self.running = True
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_captured.emit(frame)
            self.msleep(int(1000 / self.fps))
        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()
