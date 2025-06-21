"""
檢測頁籤模組
提供YOLO-PCB檢測功能的GUI介面
"""

import os
import datetime
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox,
                             QFileDialog, QProgressBar, QComboBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QSlider, QMessageBox, QShortcut)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QFont, QKeySequence
import numpy as np


class DetectTab(QWidget):
    """檢測功能頁籤"""
    
    # 信號定義
    detection_started = pyqtSignal()
    detection_finished = pyqtSignal(str)  # 結果路徑
    log_message = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.current_source = ""
        self.current_weights = ""
        self.output_dir = ""
        
        self.init_ui()
        self.connect_signals()
        # ESC鍵中斷攝像頭檢測快捷鍵
        esc_seq = QKeySequence(Qt.Key_Escape)
        self.esc_shortcut = QShortcut(esc_seq, self)
        self.esc_shortcut.setContext(Qt.ApplicationShortcut)  # 全局快捷鍵
        self.esc_shortcut.activated.connect(self.on_escape_pressed)
        self.add_log("ESC shortcut initialized")
        
    def init_ui(self):
        """初始化使用者介面"""
        # 創建主要水平布局
        main_layout = QHBoxLayout(self)
          # 創建左側控制面板（40%寬度）
        left_panel = QWidget()
        left_panel.setMaximumWidth(650)  # 增加左側最大寬度
        left_panel.setMinimumWidth(500)  # 設置最小寬度
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)  # 控制群組間距
        
        # 在左側面板添加控制群組
        self.create_input_group(left_layout)
        self.create_model_group(left_layout)
        self.create_detection_group(left_layout)
        self.create_output_group(left_layout)
        self.create_control_group(left_layout)
        
        left_layout.addStretch()
          # 創建右側圖像顯示面板（60%寬度）
        right_panel = QWidget()
        right_panel.setMinimumWidth(800)  # 設置最小寬度確保圖像顯示空間
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(5)
        
        self.create_image_display_group(right_layout)
        
        # 將左右面板添加到主布局
        main_layout.addWidget(left_panel, 2)  # 左側占2份
        main_layout.addWidget(right_panel, 3)  # 右側占3份（60%）
        
        # 設置布局間距
        main_layout.setSpacing(10)
        
    def create_input_group(self, parent_layout):
        """創建輸入來源群組"""
        group = QGroupBox("輸入來源")
        layout = QGridLayout(group)

        # 輸入來源選擇
        self.source_label = QLabel("來源:")
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("攝像頭ID (通常為 0)")
        self.source_input.setText("0")  # 預設為 0
        self.browse_source_btn = QPushButton("瀏覽...")

        layout.addWidget(self.source_label, 0, 0)
        layout.addWidget(self.source_input, 0, 1)
        layout.addWidget(self.browse_source_btn, 0, 2)

        # 輸入類型選擇
        self.input_type_label = QLabel("輸入類型:")
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItems(["自動檢測", "圖片", "影片", "資料夾", "攝像頭"])
        self.input_type_combo.setCurrentText("攝像頭")  # 預設為攝像頭

        layout.addWidget(self.input_type_label, 1, 0)
        layout.addWidget(self.input_type_combo, 1, 1)
        
        parent_layout.addWidget(group)
        
    def create_model_group(self, parent_layout):
        """創建模型設定群組"""
        group = QGroupBox("模型設定")
        layout = QGridLayout(group)
        
        # 權重檔案
        self.weights_label = QLabel("權重檔:")
        self.weights_input = QLineEdit()
        self.weights_input.setPlaceholderText("選擇.pt權重檔案")
        self.weights_input.setText("./weights/baseline.pt")  # 預設.pt檔案路徑
        self.browse_weights_btn = QPushButton("瀏覽...")
        
        layout.addWidget(self.weights_label, 0, 0)
        layout.addWidget(self.weights_input, 0, 1)
        layout.addWidget(self.browse_weights_btn, 0, 2)
        
        # 模型大小
        self.model_size_label = QLabel("模型大小:")
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["640", "416", "320", "自定義"])
        self.model_size_combo.setCurrentText("640")
        
        layout.addWidget(self.model_size_label, 1, 0)
        layout.addWidget(self.model_size_combo, 1, 1)
        
        # 設備選擇
        self.device_label = QLabel("計算設備:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda:0"])
        
        layout.addWidget(self.device_label, 2, 0)
        layout.addWidget(self.device_combo, 2, 1)
        
        parent_layout.addWidget(group)
        
    def create_detection_group(self, parent_layout):
        """創建檢測參數群組"""
        group = QGroupBox("檢測參數")
        layout = QGridLayout(group)
        
        # 信心閾值
        self.conf_label = QLabel("信心閾值:")
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.0, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(0.25)
        self.conf_spinbox.setDecimals(2)
        
        layout.addWidget(self.conf_label, 0, 0)
        layout.addWidget(self.conf_spinbox, 0, 1)
        
        # IOU 閾值
        self.iou_label = QLabel("IOU閾值:")
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.0, 1.0)
        self.iou_spinbox.setSingleStep(0.05)
        self.iou_spinbox.setValue(0.45)
        self.iou_spinbox.setDecimals(2)
        
        layout.addWidget(self.iou_label, 1, 0)
        layout.addWidget(self.iou_spinbox, 1, 1)
        
        # 最大檢測數
        self.max_det_label = QLabel("最大檢測數:")
        self.max_det_spinbox = QSpinBox()
        self.max_det_spinbox.setRange(1, 1000)
        self.max_det_spinbox.setValue(1000)
        
        layout.addWidget(self.max_det_label, 2, 0)
        layout.addWidget(self.max_det_spinbox, 2, 1)
        
        # 類別過濾
        self.classes_label = QLabel("檢測類別:")
        self.classes_input = QLineEdit()
        self.classes_input.setPlaceholderText("留空檢測所有類別，或輸入類別ID (如: 0,1,2)")
        
        layout.addWidget(self.classes_label, 3, 0)
        layout.addWidget(self.classes_input, 3, 1, 1, 2)
        
        parent_layout.addWidget(group)
        
    def create_output_group(self, parent_layout):
        """創建輸出設定群組"""
        group = QGroupBox("輸出設定")
        layout = QGridLayout(group)
        
        # 輸出目錄
        self.output_label = QLabel("輸出目錄:")
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("選擇結果輸出目錄")
        self.browse_output_btn = QPushButton("瀏覽...")
        
        layout.addWidget(self.output_label, 0, 0)
        layout.addWidget(self.output_input, 0, 1)
        layout.addWidget(self.browse_output_btn, 0, 2)
        
        # 輸出選項
        self.save_txt_checkbox = QCheckBox("儲存標註文件(.txt)")
        self.save_conf_checkbox = QCheckBox("儲存信心度")
        self.save_crop_checkbox = QCheckBox("儲存裁剪圖片")
        
        layout.addWidget(self.save_txt_checkbox, 1, 0)
        layout.addWidget(self.save_conf_checkbox, 1, 1)
        layout.addWidget(self.save_crop_checkbox, 1, 2)
        
        # 可視化選項
        self.show_labels_checkbox = QCheckBox("顯示標籤")
        self.show_conf_checkbox = QCheckBox("顯示信心度")
        self.line_thickness_label = QLabel("線條粗細:")
        self.line_thickness_spinbox = QSpinBox()
        self.line_thickness_spinbox.setRange(1, 10)
        self.line_thickness_spinbox.setValue(3)
        
        self.show_labels_checkbox.setChecked(True)
        self.show_conf_checkbox.setChecked(True)
        
        layout.addWidget(self.show_labels_checkbox, 2, 0)
        layout.addWidget(self.show_conf_checkbox, 2, 1)
        layout.addWidget(self.line_thickness_label, 3, 0)
        layout.addWidget(self.line_thickness_spinbox, 3, 1)
        
        parent_layout.addWidget(group)
        
    def create_control_group(self, parent_layout):
        """創建控制群組"""
        group = QGroupBox("檢測控制")
        layout = QVBoxLayout(group)
        
        # 按鈕區域
        button_layout = QHBoxLayout()
        
        self.detect_btn = QPushButton("開始檢測")
        self.detect_btn.setMinimumHeight(40)
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.stop_btn = QPushButton("停止檢測")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c3150a;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.clear_btn = QPushButton("清除設定")
        self.clear_btn.setMinimumHeight(40)
        
        button_layout.addWidget(self.detect_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.clear_btn)
        
        # 進度條
        self.progress_label = QLabel("就緒")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
          # 日誌輸出 - 調整高度讓左側更緊湊
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)  # 減少日誌區域高度
        self.log_text.setMinimumHeight(80)   # 設置最小高度
        self.log_text.setPlaceholderText("檢測日誌將顯示在此...")
        font = QFont("Consolas", 9)
        self.log_text.setFont(font)
        layout.addLayout(button_layout)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_text)
        
        parent_layout.addWidget(group)
    def create_image_display_group(self, parent_layout):
        """創建圖像顯示群組"""
        group = QGroupBox("檢測結果顯示")
        layout = QVBoxLayout(group)
        
        # 創建統計信息面板
        stats_layout = QHBoxLayout()
        
        # 檢測統計
        self.detection_stats_label = QLabel("等待檢測...")
        self.detection_stats_label.setStyleSheet("font-weight: bold; color: #0066cc; font-size: 14px;")
        stats_layout.addWidget(QLabel("檢測結果:"))
        stats_layout.addWidget(self.detection_stats_label)
        
        stats_layout.addStretch()
        
        # 處理狀態
        self.processing_status_label = QLabel("就緒")
        self.processing_status_label.setStyleSheet("color: green; font-size: 14px;")
        stats_layout.addWidget(QLabel("狀態:"))
        stats_layout.addWidget(self.processing_status_label)
        
        layout.addLayout(stats_layout)
        
        # 創建圖像查看器 - 佔用更多空間
        try:
            from gui.widgets.image_viewer import AnnotatedImageViewer
            self.image_viewer = AnnotatedImageViewer()
            # 設置更大的最小尺寸
            self.image_viewer.setMinimumSize(800, 600)
            self.image_viewer.setSizePolicy(self.image_viewer.sizePolicy().Expanding, 
                                          self.image_viewer.sizePolicy().Expanding)
            
            # 連接圖像查看器的信號
            if hasattr(self.image_viewer, 'image_clicked'):
                self.image_viewer.image_clicked.connect(self.on_image_clicked)
                
            layout.addWidget(self.image_viewer)
            
        except ImportError:
            # 如果無法導入自定義查看器，使用基本標籤
            self.image_viewer = QLabel()
            self.image_viewer.setMinimumSize(800, 600)
            self.image_viewer.setSizePolicy(self.image_viewer.sizePolicy().Expanding, 
                                          self.image_viewer.sizePolicy().Expanding)
            self.image_viewer.setStyleSheet("border: 2px solid #ccc; background-color: #f8f8f8; border-radius: 5px;")
            self.image_viewer.setAlignment(Qt.AlignCenter)
            self.image_viewer.setText("檢測結果將顯示在此\n\n請選擇輸入來源並開始檢測")
            layout.addWidget(self.image_viewer)
        
        # 添加圖像控制面板 - 更緊湊的佈局
        control_layout = QHBoxLayout()
        
        # 顯示控制
        self.show_annotations_checkbox = QCheckBox("顯示標註")
        self.show_annotations_checkbox.setChecked(True)
        self.show_annotations_checkbox.toggled.connect(self.toggle_annotations)
        
        self.show_confidence_checkbox = QCheckBox("顯示信心度")
        self.show_confidence_checkbox.setChecked(True)
        self.show_confidence_checkbox.toggled.connect(self.toggle_confidence)
        
        control_layout.addWidget(self.show_annotations_checkbox)
        control_layout.addWidget(self.show_confidence_checkbox)
        control_layout.addStretch()
        
        # 保存按鈕
        self.save_image_btn = QPushButton("保存當前圖像")
        self.save_image_btn.setEnabled(False)
        self.save_image_btn.clicked.connect(self.save_current_image)
        self.save_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }        """)
        
        control_layout.addWidget(self.save_image_btn)
        
        layout.addLayout(control_layout)
        parent_layout.addWidget(group)
        
    def connect_signals(self):
        """連接信號與槽"""
        # 按鈕信號
        self.browse_source_btn.clicked.connect(self.browse_source)
        self.browse_weights_btn.clicked.connect(self.browse_weights)
        self.browse_output_btn.clicked.connect(self.browse_output)
        self.detect_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.clear_btn.clicked.connect(self.clear_settings)
        
        # 組合框信號
        self.input_type_combo.currentTextChanged.connect(self.on_input_type_changed)
        
        # 初始化檢測工作器為None，在需要時才創建
        self.detection_worker = None
        self.add_log("信號連接完成，檢測器將在開始檢測時初始化")
        
        # 連接圖像顯示器信號
        if hasattr(self, 'image_viewer'):
            self.image_viewer.image_clicked.connect(self.on_image_clicked)
            
    def cleanup_worker(self):
        """清理檢測工作執行緒"""
        if self.detection_worker:
            if self.detection_worker.isRunning():
                self.detection_worker.running = False
                self.detection_worker.quit()
                self.detection_worker.wait(3000)
            self.detection_worker = None
            
    def connect_worker_signals(self):
        """連接工作執行緒信號，全部強制用 Qt.QueuedConnection"""
        from PyQt5.QtCore import Qt
        if self.detection_worker:
            self.detection_worker.log_message.connect(self.add_log, Qt.QueuedConnection)
            self.detection_worker.detection_finished.connect(self.on_detection_finished, Qt.QueuedConnection)
            self.detection_worker.error_occurred.connect(self.add_log, Qt.QueuedConnection)
            self.detection_worker.frame_processed.connect(self.on_frame_processed, Qt.QueuedConnection)
            self.detection_worker.detection_result.connect(self.on_detection_result, Qt.QueuedConnection)
            self.detection_worker.progress_updated.connect(self.progress_bar.setValue, Qt.QueuedConnection)
            
    @pyqtSlot(QPixmap)
    def on_frame_processed(self, pixmap):
        """處理檢測完成的幀（已優化為QPixmap）"""
        try:
            # 確保pixmap是正確的QPixmap格式
            if not isinstance(pixmap, QPixmap):
                self.add_log(f"警告: 接收到非QPixmap格式的數據: {type(pixmap)}")
                return
            if pixmap.isNull():
                self.add_log("警告: 接收到空的QPixmap")
                return
            # 強制縮圖，避免大圖阻塞UI
            max_w, max_h = 800, 600
            if pixmap.width() > max_w or pixmap.height() > max_h:
                pixmap = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if hasattr(self, 'image_viewer'):
                self.image_viewer.setPixmap(pixmap)
                self.add_log(f"圖像已更新: {pixmap.size().width()}x{pixmap.size().height()}")
            else:
                self.add_log("警告: 圖像顯示器未初始化")
            self.update_detection_stats()
        except Exception as e:
            error_msg = f"幀處理錯誤: {str(e)}"
            self.add_log(error_msg)
            # 發生錯誤時不要重置UI，只記錄錯誤
            
    @pyqtSlot(dict)
    def on_detection_result(self, result):
        """處理檢測結果（輕量級，標註已在worker完成）"""
        try:
            # 更新結果統計
            detections = result.get('detections', [])
            count = result.get('count', 0)
            # 更新統計顯示
            if hasattr(self, 'detection_stats_label'):
                self.detection_stats_label.setText(f"檢測到 {count} 個缺陷")
            # 顯示標註（即使worker已畫框，UI端也需重繪以支援互動/縮放）
            if hasattr(self, 'image_viewer') and hasattr(self.image_viewer, 'set_detections'):
                self.image_viewer.set_detections(detections)
        except Exception as e:
            self.add_log(f"結果處理錯誤: {str(e)}")
            
    def update_detection_stats(self):
        """更新檢測統計信息"""
        try:
            # 這裡可以添加更多統計信息的更新
            pass
        except Exception as e:
            self.add_log(f"統計更新錯誤: {str(e)}")
            
    @pyqtSlot(int, int)
    def on_image_clicked(self, x, y):
        """處理圖像點擊事件"""
        try:
            self.add_log(f"點擊位置: ({x}, {y})")
            # 這裡可以添加更多點擊處理邏輯
        except Exception as e:
            self.add_log(f"圖像點擊處理錯誤: {str(e)}")
            
    def on_input_type_changed(self, input_type):
        """處理輸入類型變更"""
        if input_type == "攝像頭":
            self.source_input.setPlaceholderText("攝像頭ID (通常為 0)")
            self.source_input.setText("0")
        else:
            self.source_input.setPlaceholderText("選擇圖片、影片或資料夾路徑")
            self.source_input.clear()
        
    def browse_source(self):
        """瀏覽輸入來源"""
        input_type = self.input_type_combo.currentText()
        
        if input_type == "資料夾":
            path = QFileDialog.getExistingDirectory(
                self, "選擇輸入資料夾", ""
            )
        elif input_type == "影片":
            path, _ = QFileDialog.getOpenFileName(
                self, "選擇影片檔案", "",
                "影片檔案 (*.mp4 *.avi *.mov *.mkv);;所有檔案 (*)"
            )
        else:  # 圖片或自動檢測
            path, _ = QFileDialog.getOpenFileName(
                self, "選擇圖片檔案", "",
                "圖片檔案 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有檔案 (*)"
            )
        
        if path:
            self.source_input.setText(path)
            self.current_source = path
            
    def browse_weights(self):
        """瀏覽權重檔案"""
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇權重檔案", "",
            "PyTorch權重 (*.pt *.pth);;所有檔案 (*)"
        )
        
        if path:
            self.weights_input.setText(path)
            self.current_weights = path
            
    def browse_output(self):
        """瀏覽輸出目錄"""
        path = QFileDialog.getExistingDirectory(
            self, "選擇輸出目錄", ""
        )
        
        if path:
            self.output_input.setText(path)
            self.output_dir = path
            
    def validate_inputs(self):
        """驗證輸入參數"""
        source_valid = bool(self.source_input.text().strip())
        weights_valid = bool(self.weights_input.text().strip())
        
        self.detect_btn.setEnabled(source_valid and weights_valid)
        
    def get_detection_params(self):
        """獲取檢測參數"""
        params = {
            'source': self.source_input.text().strip(),
            'weights': self.weights_input.text().strip(),
            'output': self.output_input.text().strip() or 'runs/detect',
            'imgsz': int(self.model_size_combo.currentText()) if self.model_size_combo.currentText().isdigit() else 640,
            'conf_thres': self.conf_spinbox.value(),
            'iou_thres': self.iou_spinbox.value(),
            'max_det': self.max_det_spinbox.value(),
            'device': self.device_combo.currentText(),
            'save_txt': self.save_txt_checkbox.isChecked(),
            'save_conf': self.save_conf_checkbox.isChecked(),
            'save_crop': self.save_crop_checkbox.isChecked(),
            'hide_labels': not self.show_labels_checkbox.isChecked(),            'hide_conf': not self.show_conf_checkbox.isChecked(),
            'line_thickness': self.line_thickness_spinbox.value(),
        }
        
        # 處理類別過濾
        classes_text = self.classes_input.text().strip()
        if classes_text:
            try:
                classes = [int(x.strip()) for x in classes_text.split(',') if x.strip()]
                params['classes'] = classes
            except ValueError:
                self.log_message.emit("警告: 類別ID格式不正確，將檢測所有類別")
        
        return params
        
    def start_detection(self):
        """開始檢測"""
        try:
            # 檢查是否已有檢測在運行
            if self.detection_worker and self.detection_worker.isRunning():
                self.add_log("警告: 檢測已在運行中")
                return
            
            # 立即設置按鈕狀態，防止重複點擊
            self.detect_btn.setEnabled(False)
            self.add_log("正在驗證檢測參數...")
            
            # 強制處理UI事件，確保按鈕狀態立即更新
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            
            params = self.get_detection_params()
            
            # 驗證必要參數
            source_valid = self.validate_source(params['source'])
            if not source_valid:
                self.add_log("錯誤: 輸入來源驗證失敗")
                self.reset_ui_state()
                return
                
            if not os.path.exists(params['weights']):
                self.add_log("錯誤: 權重檔案不存在")
                QMessageBox.warning(self, "錯誤", "權重檔案不存在！")
                self.reset_ui_state()
                return
                
            # 驗證通過，正式啟動檢測
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            
            # 強制處理UI事件
            QApplication.processEvents()
            
            # 如果是攝像頭檢測，使用無限進度條；否則使用確定進度條
            if params['source'].isdigit():  # 攝像頭
                self.progress_bar.setRange(0, 0)  # 無限進度條
                self.progress_label.setText("攝像頭檢測中...")
                self.add_log("攝像頭檢測模式 - 按ESC鍵或點擊停止按鈕可隨時中斷")
            else:
                self.progress_bar.setRange(0, 100)  # 確定進度條
                self.progress_label.setText("檢測進行中...")
            
            # 清除之前的日誌和圖像
            self.log_text.clear()
            if hasattr(self.image_viewer, 'clear_image'):
                self.image_viewer.clear_image()
            
            self.add_log("開始檢測...")
            self.add_log(f"來源: {params['source']}")
            self.add_log(f"權重: {params['weights']}")
            self.add_log(f"輸出: {params['output']}")
            
            # 確保UI更新
            QApplication.processEvents()
            
            # 嘗試使用核心檢測功能
            try:
                from core.detector import DetectionWorker
                
                # 清理舊的檢測工作執行緒
                if self.detection_worker:
                    self.cleanup_worker()
                
                # 創建新的檢測工作執行緒
                self.detection_worker = DetectionWorker()
                self.detection_worker.set_parameters(params)
                
                # 設置顯示區域大小（優化UI性能）
                if hasattr(self, 'image_viewer'):
                    viewer_size = self.image_viewer.size()
                    self.detection_worker.set_display_size(viewer_size.width(), viewer_size.height())
                
                # 連接信號
                self.connect_worker_signals()
                
                # 載入模型
                if self.detection_worker.load_model(params['weights'], params.get('device', 'auto')):
                    # 開始檢測
                    self.detection_worker.start()
                    self.detection_started.emit()
                    self.add_log("檢測已啟動")
                else:
                    self.add_log("檢測啟動失敗: 模型載入失敗")
                    self.reset_ui_state()
                    
            except ImportError as e:
                self.add_log(f"核心檢測模組載入失敗: {str(e)}")
                self.add_log("使用模擬檢測模式")
                # 模擬檢測也需要正確的按鈕狀態
                self._simulate_detection(params)
                
        except Exception as e:
            self.add_log(f"檢測啟動錯誤: {str(e)}")
            self.reset_ui_state()
            
    def _simulate_detection(self, params):
        """模擬檢測過程（當核心模組不可用時）"""
        import time
        
        self.add_log("注意: 這是模擬檢測，實際部署時會使用真實的檢測功能")
          # 模擬檢測進度
        self.simulation_progress = 0
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self._update_simulation)
        self.simulation_timer.start(100)  # 每100ms更新一次
        
    def _update_simulation(self):
        """更新模擬檢測進度"""
        self.simulation_progress += 5
        
        if self.simulation_progress >= 100:
            self.simulation_timer.stop()
            self.add_log("模擬檢測完成")
            self.reset_ui_state()
            # 模擬檢測結果
            result_path = self.get_detection_params().get('output', 'runs/detect')
            self.detection_finished.emit(result_path)
        else:
            # 更新進度條（如果是確定進度條）
            if self.progress_bar.maximum() != 0:
                self.progress_bar.setValue(self.simulation_progress)
                
    def stop_detection(self):
        """停止檢測（非阻塞，僅設flag）"""
        self.add_log("用戶請求停止檢測...")
        self.stop_btn.setEnabled(False)
        if hasattr(self, 'detection_worker') and self.detection_worker:
            try:
                self.detection_worker.running = False
                if hasattr(self.detection_worker, 'stop_detection'):
                    self.detection_worker.stop_detection()
                self.add_log("檢測執行緒停止信號已發送")
                # 不呼叫wait/quit/terminate，讓worker自己emit finished
            except Exception as e:
                self.add_log(f"停止檢測執行緒時發生錯誤: {str(e)}")
        if hasattr(self, 'simulation_timer') and self.simulation_timer.isActive():
            self.simulation_timer.stop()
            self.add_log("模擬檢測已停止")
        self.add_log("檢測已請求停止，等待執行緒結束...")
        # UI狀態不立即reset，等worker emit finished後再reset
    def reset_ui_state(self):
        """重置UI狀態"""
        self.detect_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("就緒")
        
    def clear_settings(self):
        """清除所有設定"""
        reply = QMessageBox.question(
            self, "確認", "確定要清除所有設定嗎？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 清除輸入
            self.source_input.clear()
            self.weights_input.clear()
            self.output_input.clear()
            self.classes_input.clear()
            
            # 重置為預設值
            self.input_type_combo.setCurrentIndex(0)
            self.model_size_combo.setCurrentText("640")
            self.device_combo.setCurrentIndex(0)
            self.conf_spinbox.setValue(0.25)
            self.iou_spinbox.setValue(0.45)
            self.max_det_spinbox.setValue(1000)
            self.line_thickness_spinbox.setValue(3)
            
            # 重置核取方塊
            self.save_txt_checkbox.setChecked(False)
            self.save_conf_checkbox.setChecked(False)
            self.save_crop_checkbox.setChecked(False)
            self.show_labels_checkbox.setChecked(True)
            self.show_conf_checkbox.setChecked(True)
              # 清除日誌
            self.log_text.clear()
            
            self.add_log("設定已清除")
            
    def add_log(self, message):
        """添加日誌信息"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        self.log_message.emit(formatted_message)
        
    @pyqtSlot(int)
    def update_progress(self, value):
        """更新進度條"""
        if self.progress_bar.maximum() != 0:
            self.progress_bar.setValue(value)
            
    @pyqtSlot(str)
    def on_detection_finished(self, result_path):
        """檢測完成處理"""
        self.reset_ui_state()
        self.add_log(f"檢測完成！結果保存至: {result_path}")
        self.detection_finished.emit(result_path)
        
    @pyqtSlot(str)
    def on_detection_error(self, error_message):
        """檢測錯誤處理"""
        self.add_log(f"檢測錯誤: {error_message}")
        
        # 立即重置 UI 狀態
        self.reset_ui_state()
        
        # 如果是嚴重錯誤，顯示對話框
        if "Tensor" in error_message or "模型" in error_message:
            QMessageBox.critical(self, "檢測錯誤", 
                               f"檢測過程發生錯誤:\n{error_message}\n\n請檢查模型檔案或輸入格式。")
        else:
            self.add_log(f"錯誤已處理，UI狀態已重置")
    
    def toggle_annotations(self, show):
        """切換標註顯示"""
        if hasattr(self.image_viewer, 'set_show_annotations'):
            self.image_viewer.set_show_annotations(show)
            self.add_log(f"標註顯示: {'開啟' if show else '關閉'}")
    
    def toggle_confidence(self, show):
        """切換信心度顯示"""
        if hasattr(self.image_viewer, 'set_show_confidence'):
            self.image_viewer.set_show_confidence(show)
            self.add_log(f"信心度顯示: {'開啟' if show else '關閉'}")
    
    def save_current_image(self):
        """保存當前顯示的圖像"""
        try:
            if hasattr(self.image_viewer, 'save_current_image'):
                filename, _ = QFileDialog.getSaveFileName(
                    self, "保存圖像", "detection_result.jpg",
                    "圖像檔案 (*.jpg *.jpeg *.png *.bmp);;所有檔案 (*)"
                )
                if filename:
                    if self.image_viewer.save_current_image(filename):
                        self.add_log(f"圖像已保存至: {filename}")
                        QMessageBox.information(self, "成功", f"圖像已保存至:\n{filename}")
                    else:
                        QMessageBox.warning(self, "錯誤", "圖像保存失敗")
            else:
                QMessageBox.information(self, "提示", "當前圖像查看器不支援保存功能")
        except Exception as e:
            self.add_log(f"保存圖像時發生錯誤: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"保存圖像時發生錯誤:\n{str(e)}")
    
    def update_detection_display(self, image, detections=None):
        """更新檢測結果顯示"""
        try:
            # 更新圖像
            if hasattr(self.image_viewer, 'set_image'):
                self.image_viewer.set_image(image)
                self.save_image_btn.setEnabled(True)
            
            # 更新統計信息
            if detections:
                count = len(detections)
                self.detection_stats_label.setText(f"檢測到 {count} 個缺陷")
                self.detection_stats_label.setStyleSheet("font-weight: bold; color: #cc0000;" if count > 0 else "font-weight: bold; color: #0066cc;")
                
                # 如果有標註功能，設置標註
                if hasattr(self.image_viewer, 'set_annotations'):
                    annotations = []
                    for det in detections:
                        if isinstance(det, dict):
                            annotations.append({
                                'bbox': det.get('bbox', [0, 0, 0, 0]),
                                'label': f"{det.get('name', 'unknown')}: {det.get('confidence', 0.0):.2f}",
                                'color': (0, 255, 0)
                            })
                    self.image_viewer.set_annotations(annotations)
            else:
                self.detection_stats_label.setText("檢測完成，無缺陷")
                self.detection_stats_label.setStyleSheet("font-weight: bold; color: #00cc00;")
                
        except Exception as e:
            self.add_log(f"更新顯示時發生錯誤: {str(e)}")
    
    def set_processing_status(self, status, color="green"):
        """設置處理狀態"""
        self.processing_status_label.setText(status)
        self.processing_status_label.setStyleSheet(f"color: {color};")
    
    def validate_source(self, source):
        """驗證輸入來源"""
        if not source:
            QMessageBox.warning(self, "錯誤", "請指定輸入來源！")
            return False
        
        input_type = self.input_type_combo.currentText()
        
        if input_type == "攝像頭":
            # 驗證攝像頭ID
            try:
                camera_id = int(source)
                if camera_id < 0:
                    QMessageBox.warning(self, "錯誤", "攝像頭ID必須是非負整數！")
                    return False
                
                # 嘗試打開攝像頭測試
                import cv2
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    cap.release()
                    QMessageBox.warning(self, "錯誤", f"無法打開攝像頭 {camera_id}！\n請檢查攝像頭是否連接正常。")
                    return False
                cap.release()
                
                self.add_log(f"攝像頭 {camera_id} 驗證成功")
                return True
                
            except ValueError:
                QMessageBox.warning(self, "錯誤", "攝像頭ID必須是數字！")
                return False
        else:
            # 驗證文件或目錄路徑
            if not os.path.exists(source):
                QMessageBox.warning(self, "錯誤", f"輸入來源不存在：\n{source}")
                return False
            
            # 額外驗證文件類型
            if input_type == "圖片":
                valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                if not any(source.lower().endswith(ext) for ext in valid_extensions):
                    QMessageBox.warning(self, "錯誤", "不支援的圖片格式！\n支援格式：jpg, jpeg, png, bmp, tiff")
                    return False
            elif input_type == "影片":
                valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
                if not any(source.lower().endswith(ext) for ext in valid_extensions):
                    QMessageBox.warning(self, "錯誤", "不支援的影片格式！\n支援格式：mp4, avi, mov, mkv, wmv, flv")
                    return False
            elif input_type == "資料夾":
                if not os.path.isdir(source):
                    QMessageBox.warning(self, "錯誤", "指定的路徑不是資料夾！")
                    return False
        
        return True
    def on_escape_pressed(self):
        """處理ESC鍵：當為攝像頭檢測且可停止時，終止檢測"""
        # 日誌：ESC按鍵事件觸發
        self.add_log("ESC pressed - on_escape_pressed called")
        try:
            mode = self.input_type_combo.currentText()
            stop_enabled = self.stop_btn.isEnabled()
            self.add_log(f"檢測模式: {mode}, 停止按鈕狀態: {stop_enabled}")
            if mode == "攝像頭" and stop_enabled:
                self.add_log("ESC觸發停止檢測")
                self.stop_detection()
            else:
                self.add_log("ESC未觸發停止：非攝像頭模式或停止按鈕未啟用")
        except Exception as e:
            self.add_log(f"ESC按鍵處理錯誤: {e}")
    def keyPressEvent(self, event):
        """攔截鍵盤事件，捕捉ESC鍵中斷檢測"""
        if event.key() == Qt.Key_Escape:
            self.add_log("keyPressEvent: ESC key detected")
            self.on_escape_pressed()
            event.accept()
        else:
            super().keyPressEvent(event)
