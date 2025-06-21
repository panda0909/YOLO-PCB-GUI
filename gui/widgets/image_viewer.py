"""
圖像查看器小部件
用於顯示檢測結果圖像
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen


class ImageViewer(QLabel):
    """圖像顯示小部件"""
    
    # 信號定義
    image_clicked = pyqtSignal(int, int)  # 點擊位置信號
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 設置基本屬性
        self.setMinimumSize(320, 240)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #f8f8f8;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 初始化變數
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.show_placeholder()
        
    def show_placeholder(self):
        """顯示佔位文字"""
        self.setText("等待圖像輸入...")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 5px;
                background-color: #f8f8f8;
                color: #666666;
                font-size: 14px;
            }
        """)
    
    def display_opencv_image(self, cv_image):
        """顯示OpenCV格式的圖像"""
        try:
            if cv_image is None:
                self.show_placeholder()
                return
                
            # 保存原始圖像
            self.original_image = cv_image.copy()
            
            # 轉換顏色空間
            if len(cv_image.shape) == 3:
                # BGR轉RGB
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                # 灰度圖像
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            
            # 創建QImage
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 縮放圖像以適應標籤大小
            pixmap = QPixmap.fromImage(q_image)
            self.display_scaled_image(pixmap)
            
        except Exception as e:
            print(f"顯示圖像失敗: {str(e)}")
            self.show_placeholder()
    
    def display_pixmap(self, pixmap):
        """顯示QPixmap"""
        try:
            if pixmap.isNull():
                self.show_placeholder()
                return
                
            self.display_scaled_image(pixmap)
            
        except Exception as e:
            print(f"顯示Pixmap失敗: {str(e)}")
            self.show_placeholder()
    
    def display_scaled_image(self, pixmap):
        """顯示縮放後的圖像"""
        try:
            # 計算縮放比例
            label_size = self.size()
            image_size = pixmap.size()
            
            # 計算保持寬高比的縮放
            scale_w = label_size.width() / image_size.width()
            scale_h = label_size.height() / image_size.height()
            self.scale_factor = min(scale_w, scale_h, 1.0)  # 不放大
            
            # 縮放圖像
            scaled_size = image_size * self.scale_factor
            scaled_pixmap = pixmap.scaled(
                scaled_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            # 顯示圖像
            self.setPixmap(scaled_pixmap)
            self.display_image = scaled_pixmap
            
            # 更新樣式
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 5px;
                    background-color: white;
                }
            """)
            
        except Exception as e:
            print(f"縮放圖像失敗: {str(e)}")
            self.show_placeholder()
    
    def clear_image(self):
        """清除圖像"""
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.show_placeholder()
    
    def get_original_image(self):
        """獲取原始圖像"""
        return self.original_image
    
    def save_image(self, file_path):
        """保存當前顯示的圖像"""
        try:
            if self.display_image and not self.display_image.isNull():
                return self.display_image.save(file_path)
            return False
        except Exception as e:
            print(f"保存圖像失敗: {str(e)}")
            return False
    
    def mousePressEvent(self, event):
        """處理滑鼠點擊事件"""
        if event.button() == Qt.LeftButton and self.display_image:
            # 獲取點擊位置
            x = event.x()
            y = event.y()
            
            # 轉換為圖像坐標
            if self.original_image is not None:
                # 計算圖像在標籤中的位置
                label_size = self.size()
                image_size = self.display_image.size()
                
                # 圖像在標籤中居中顯示
                x_offset = (label_size.width() - image_size.width()) // 2
                y_offset = (label_size.height() - image_size.height()) // 2
                
                # 調整點擊坐標
                image_x = x - x_offset
                image_y = y - y_offset
                
                # 檢查是否在圖像範圍內
                if 0 <= image_x < image_size.width() and 0 <= image_y < image_size.height():
                    # 轉換為原始圖像坐標
                    if self.scale_factor > 0:
                        original_x = int(image_x / self.scale_factor)
                        original_y = int(image_y / self.scale_factor)
                        self.image_clicked.emit(original_x, original_y)
        
        super().mousePressEvent(event)
    
    def resizeEvent(self, event):
        """處理重新調整大小事件"""
        super().resizeEvent(event)
        
        # 如果有圖像，重新縮放
        if self.original_image is not None:
            self.display_opencv_image(self.original_image)


class AnnotatedImageViewer(ImageViewer):
    """帶標註的圖像查看器"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotations = []  # 標註列表
    
    def set_image(self, cv_image):
        """設置圖像並清除標註"""
        self.clear_annotations()
        self.display_opencv_image(cv_image)
        
    def add_annotation(self, x, y, width, height, label="", color=(255, 0, 0)):
        """添加標註框"""
        annotation = {
            'x': x,
            'y': y, 
            'width': width,
            'height': height,
            'label': label,
            'color': color
        }
        if hasattr(self, 'log_message'):
            self.log_message.emit(f"add_annotation: x={x}, y={y}, w={width}, h={height}, label={label}, color={color}")
        self.annotations.append(annotation)
        self.update_display()
        
    def clear_annotations(self):
        """清除所有標註"""
        self.annotations = []
        self.update_display()
        
    def update_display(self):
        """更新顯示（重繪標註）"""
        if self.original_image is None:
            return
        annotated_image = self.original_image.copy()
        for ann in self.annotations:
            x, y, w, h = ann['x'], ann['y'], ann['width'], ann['height']
            color = ann['color']
            label = ann['label']
            if hasattr(self, 'log_message'):
                self.log_message.emit(f"draw_box: x={x}, y={y}, w={w}, h={h}, label={label}, color={color}")
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(annotated_image, 
                            (x, y - text_height - 10), 
                            (x + text_width, y), 
                            color, -1)
                cv2.putText(annotated_image, label, 
                          (x, y - 5), font, font_scale, (255, 255, 255), thickness)
        self.display_opencv_image(annotated_image)
    
    def set_detections(self, detections):
        """設置檢測結果"""
        self.clear_annotations()
        
        # 類別顏色映射
        colors = {
            'missing_hole': (0, 0, 255),      # 紅色
            'mouse_bite': (0, 255, 0),        # 綠色
            'open_circuit': (255, 0, 0),      # 藍色
            'short': (0, 255, 255),           # 黃色
            'spur': (255, 0, 255),            # 紫色
            'spurious_copper': (255, 255, 0)  # 青色
        }
        
        for detection in detections:
            if 'bbox' in detection:
                bbox = detection['bbox']
                class_name = detection.get('class', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                # 獲取顏色
                color = colors.get(class_name, (128, 128, 128))
                
                # 創建標籤
                label = f"{class_name} {confidence:.2f}"
                
                # 添加標註
                self.add_annotation(
                    int(bbox[0]), int(bbox[1]), 
                    int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]),
                    label, color
                )
