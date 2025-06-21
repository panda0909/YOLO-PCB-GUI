from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class AnalyzeWorker(QThread):
    analysis_done = pyqtSignal(object, object)  # (frame, result)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_loader, parent=None):
        super().__init__(parent)
        self.model_loader = model_loader  # 應為可呼叫的推論物件
        self.frame = None
        self.running = False
        self._new_frame = False

    def analyze(self, frame):
        self.frame = frame
        self._new_frame = True
        if not self.isRunning():
            self.start()

    def run(self):
        self.running = True
        while self.running:
            if self._new_frame and self.frame is not None:
                try:
                    result = self.model_loader(self.frame)
                    self.analysis_done.emit(self.frame, result)
                except Exception as e:
                    self.error_occurred.emit(f"分析失敗: {str(e)}")
                self._new_frame = False
            self.msleep(10)

    def stop(self):
        self.running = False
        self.wait()
