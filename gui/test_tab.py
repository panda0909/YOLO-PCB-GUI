"""
測試頁籤模組
提供YOLO-PCB測試功能的GUI介面
"""

import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox,
                             QFileDialog, QProgressBar, QComboBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont


class TestTab(QWidget):
    """測試功能頁籤"""
      # 信號定義
    test_started = pyqtSignal()
    test_finished = pyqtSignal(dict)  # 測試結果
    test_progress = pyqtSignal(int)   # 進度百分比
    log_message = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.current_weights = ""
        self.current_data_config = ""
        self.test_results = {}
        
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        """初始化使用者介面"""
        layout = QVBoxLayout(self)
        
        # 創建標籤頁
        self.tab_widget = QTabWidget()
        
        # 測試配置標籤頁
        self.config_tab = QWidget()
        self.create_config_tab()
        self.tab_widget.addTab(self.config_tab, "測試配置")
        
        # 測試控制標籤頁
        self.control_tab = QWidget()
        self.create_control_tab()
        self.tab_widget.addTab(self.control_tab, "測試控制")
        
        # 結果顯示標籤頁
        self.results_tab = QWidget()
        self.create_results_tab()
        self.tab_widget.addTab(self.results_tab, "測試結果")
        
        layout.addWidget(self.tab_widget)
        
    def create_config_tab(self):
        """創建測試配置標籤頁"""
        layout = QVBoxLayout(self.config_tab)
        
        # 模型配置群組
        model_group = QGroupBox("模型配置")
        model_layout = QGridLayout(model_group)
        
        # 權重檔案
        self.weights_label = QLabel("權重檔:")
        self.weights_input = QLineEdit()
        self.weights_input.setPlaceholderText("選擇訓練好的.pt權重檔案")
        self.browse_weights_btn = QPushButton("瀏覽...")
        
        model_layout.addWidget(self.weights_label, 0, 0)
        model_layout.addWidget(self.weights_input, 0, 1)
        model_layout.addWidget(self.browse_weights_btn, 0, 2)
        
        # 設備選擇
        self.device_label = QLabel("計算設備:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda:0"])
        
        model_layout.addWidget(self.device_label, 1, 0)
        model_layout.addWidget(self.device_combo, 1, 1)
        
        layout.addWidget(model_group)
        
        # 資料配置群組
        data_group = QGroupBox("資料配置")
        data_layout = QGridLayout(data_group)
        
        # 資料配置檔案
        self.data_config_label = QLabel("資料配置檔:")
        self.data_config_input = QLineEdit()
        self.data_config_input.setPlaceholderText("選擇.yaml資料配置檔案")
        self.browse_data_config_btn = QPushButton("瀏覽...")
        
        data_layout.addWidget(self.data_config_label, 0, 0)
        data_layout.addWidget(self.data_config_input, 0, 1)
        data_layout.addWidget(self.browse_data_config_btn, 0, 2)
        
        # 模型大小
        self.imgsz_label = QLabel("圖片大小:")
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(320, 1280)
        self.imgsz_spinbox.setSingleStep(32)
        self.imgsz_spinbox.setValue(640)
        
        data_layout.addWidget(self.imgsz_label, 1, 0)
        data_layout.addWidget(self.imgsz_spinbox, 1, 1)
        
        # 批次大小
        self.batch_size_label = QLabel("批次大小:")
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 128)
        self.batch_size_spinbox.setValue(32)
        
        data_layout.addWidget(self.batch_size_label, 2, 0)
        data_layout.addWidget(self.batch_size_spinbox, 2, 1)
        
        # 工作線程數
        self.workers_label = QLabel("工作線程數:")
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setRange(0, 16)
        self.workers_spinbox.setValue(8)
        
        data_layout.addWidget(self.workers_label, 3, 0)
        data_layout.addWidget(self.workers_spinbox, 3, 1)
        
        layout.addWidget(data_group)
        
        # 測試參數群組
        test_group = QGroupBox("測試參數")
        test_layout = QGridLayout(test_group)
        
        # 信心閾值
        self.conf_thres_label = QLabel("信心閾值:")
        self.conf_thres_spinbox = QDoubleSpinBox()
        self.conf_thres_spinbox.setRange(0.001, 1.0)
        self.conf_thres_spinbox.setSingleStep(0.05)
        self.conf_thres_spinbox.setValue(0.001)
        self.conf_thres_spinbox.setDecimals(3)
        
        test_layout.addWidget(self.conf_thres_label, 0, 0)
        test_layout.addWidget(self.conf_thres_spinbox, 0, 1)
        
        # IOU閾值
        self.iou_thres_label = QLabel("IOU閾值:")
        self.iou_thres_spinbox = QDoubleSpinBox()
        self.iou_thres_spinbox.setRange(0.1, 0.9)
        self.iou_thres_spinbox.setSingleStep(0.05)
        self.iou_thres_spinbox.setValue(0.6)
        self.iou_thres_spinbox.setDecimals(2)
        
        test_layout.addWidget(self.iou_thres_label, 1, 0)
        test_layout.addWidget(self.iou_thres_spinbox, 1, 1)
        
        # 最大檢測數
        self.max_det_label = QLabel("最大檢測數:")
        self.max_det_spinbox = QSpinBox()
        self.max_det_spinbox.setRange(1, 1000)
        self.max_det_spinbox.setValue(300)
        
        test_layout.addWidget(self.max_det_label, 2, 0)
        test_layout.addWidget(self.max_det_spinbox, 2, 1)
        
        layout.addWidget(test_group)
        
        # 測試選項群組
        options_group = QGroupBox("測試選項")
        options_layout = QGridLayout(options_group)
        
        # 測試集選擇
        self.task_label = QLabel("測試任務:")
        self.task_combo = QComboBox()
        self.task_combo.addItems(["val", "test", "速度測試"])
        self.task_combo.setCurrentText("val")
        
        options_layout.addWidget(self.task_label, 0, 0)
        options_layout.addWidget(self.task_combo, 0, 1)
        
        # 保存選項
        self.save_txt_checkbox = QCheckBox("保存預測文件")
        self.save_conf_checkbox = QCheckBox("保存信心度")
        self.save_json_checkbox = QCheckBox("保存JSON結果")
        
        options_layout.addWidget(self.save_txt_checkbox, 1, 0)
        options_layout.addWidget(self.save_conf_checkbox, 1, 1)
        options_layout.addWidget(self.save_json_checkbox, 2, 0)
        
        # 詳細輸出選項
        self.verbose_checkbox = QCheckBox("詳細輸出")
        self.verbose_checkbox.setChecked(True)
        
        options_layout.addWidget(self.verbose_checkbox, 2, 1)
        
        layout.addWidget(options_group)
        layout.addStretch()
        
    def create_control_tab(self):
        """創建測試控制標籤頁"""
        layout = QVBoxLayout(self.control_tab)
        
        # 輸出設定群組
        output_group = QGroupBox("輸出設定")
        output_layout = QGridLayout(output_group)
        
        # 專案名稱
        self.project_label = QLabel("專案名稱:")
        self.project_input = QLineEdit()
        self.project_input.setText("runs/val")
        
        output_layout.addWidget(self.project_label, 0, 0)
        output_layout.addWidget(self.project_input, 0, 1)
        
        # 實驗名稱
        self.name_label = QLabel("實驗名稱:")
        self.name_input = QLineEdit()
        self.name_input.setText("exp")
        
        output_layout.addWidget(self.name_label, 1, 0)
        output_layout.addWidget(self.name_input, 1, 1)
        
        layout.addWidget(output_group)
        
        # 控制按鈕群組
        control_group = QGroupBox("測試控制")
        control_layout = QVBoxLayout(control_group)
        
        # 按鈕區域
        button_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("開始測試")
        self.test_btn.setMinimumHeight(40)
        self.test_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.stop_btn = QPushButton("停止測試")
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
        
        button_layout.addWidget(self.test_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.clear_btn)
        
        # 進度顯示
        self.progress_label = QLabel("就緒")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 日誌輸出
        log_label = QLabel("測試日誌:")
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setPlaceholderText("測試日誌將顯示在此...")
        font = QFont("Consolas", 9)
        self.log_text.setFont(font)
        
        control_layout.addLayout(button_layout)
        control_layout.addWidget(self.progress_label)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(log_label)
        control_layout.addWidget(self.log_text)
        
        layout.addWidget(control_group)
        
    def create_results_tab(self):
        """創建測試結果標籤頁"""
        layout = QVBoxLayout(self.results_tab)
        
        # 總體結果群組
        overall_group = QGroupBox("總體結果")
        overall_layout = QGridLayout(overall_group)
        
        # 結果標籤
        self.mAP_label = QLabel("mAP@0.5: -")
        self.mAP_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.mAP50_95_label = QLabel("mAP@0.5:0.95: -")
        self.mAP50_95_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.precision_label = QLabel("Precision: -")
        self.precision_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.recall_label = QLabel("Recall: -")
        self.recall_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.f1_label = QLabel("F1-Score: -")
        self.f1_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        overall_layout.addWidget(self.mAP_label, 0, 0)
        overall_layout.addWidget(self.mAP50_95_label, 0, 1)
        overall_layout.addWidget(self.precision_label, 1, 0)
        overall_layout.addWidget(self.recall_label, 1, 1)
        overall_layout.addWidget(self.f1_label, 2, 0, 1, 2, Qt.AlignCenter)
        
        layout.addWidget(overall_group)
        
        # 分類別結果群組
        class_group = QGroupBox("分類別結果")
        class_layout = QVBoxLayout(class_group)
        
        # 結果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "類別", "圖片數", "實例數", "P", "R", "mAP@0.5", "mAP@0.5:0.95", "F1"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        class_layout.addWidget(self.results_table)
        
        layout.addWidget(class_group)
        
        # 混淆矩陣群組
        confusion_group = QGroupBox("混淆矩陣")
        confusion_layout = QVBoxLayout(confusion_group)
        
        self.confusion_text = QTextEdit()
        self.confusion_text.setMaximumHeight(150)
        self.confusion_text.setPlaceholderText("混淆矩陣將顯示在此...")
        self.confusion_text.setFont(QFont("Consolas", 9))
        
        confusion_layout.addWidget(self.confusion_text)
        
        layout.addWidget(confusion_group)
        
        # 速度測試結果群組
        speed_group = QGroupBox("速度測試結果")
        speed_layout = QGridLayout(speed_group)
        
        self.preprocess_time_label = QLabel("預處理時間: -")
        self.inference_time_label = QLabel("推理時間: -")
        self.nms_time_label = QLabel("NMS時間: -")
        self.total_time_label = QLabel("總時間: -")
        
        speed_layout.addWidget(self.preprocess_time_label, 0, 0)
        speed_layout.addWidget(self.inference_time_label, 0, 1)
        speed_layout.addWidget(self.nms_time_label, 1, 0)
        speed_layout.addWidget(self.total_time_label, 1, 1)
        
        layout.addWidget(speed_group)
        
    def connect_signals(self):
        """連接信號和槽"""
        # 瀏覽按鈕
        self.browse_weights_btn.clicked.connect(self.browse_weights)
        self.browse_data_config_btn.clicked.connect(self.browse_data_config)
        
        # 控制按鈕
        self.test_btn.clicked.connect(self.start_testing)
        self.stop_btn.clicked.connect(self.stop_testing)
        self.clear_btn.clicked.connect(self.clear_settings)
        
        # 輸入驗證
        self.weights_input.textChanged.connect(self.validate_inputs)
        self.data_config_input.textChanged.connect(self.validate_inputs)
        
    def browse_weights(self):
        """瀏覽權重檔案"""
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇權重檔案", "",
            "PyTorch權重 (*.pt *.pth);;所有檔案 (*)"
        )
        
        if path:
            self.weights_input.setText(path)
            self.current_weights = path
            
    def browse_data_config(self):
        """瀏覽資料配置檔案"""
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇資料配置檔案", "",
            "YAML檔案 (*.yaml *.yml);;所有檔案 (*)"
        )
        
        if path:
            self.data_config_input.setText(path)
            self.current_data_config = path
            
    def validate_inputs(self):
        """驗證輸入參數"""
        weights_valid = bool(self.weights_input.text().strip())
        data_valid = bool(self.data_config_input.text().strip())
        
        self.test_btn.setEnabled(weights_valid and data_valid)
        
    def get_test_params(self):
        """獲取測試參數"""
        params = {
            'weights': self.weights_input.text().strip(),
            'data': self.data_config_input.text().strip(),
            'imgsz': self.imgsz_spinbox.value(),
            'batch_size': self.batch_size_spinbox.value(),
            'conf_thres': self.conf_thres_spinbox.value(),
            'iou_thres': self.iou_thres_spinbox.value(),
            'max_det': self.max_det_spinbox.value(),
            'device': self.device_combo.currentText(),
            'workers': self.workers_spinbox.value(),
            'task': self.task_combo.currentText(),
            'save_txt': self.save_txt_checkbox.isChecked(),
            'save_conf': self.save_conf_checkbox.isChecked(),
            'save_json': self.save_json_checkbox.isChecked(),
            'verbose': self.verbose_checkbox.isChecked(),
            'project': self.project_input.text().strip(),
            'name': self.name_input.text().strip(),
        }
        
        return params
        
    def start_testing(self):
        """開始測試"""
        try:
            params = self.get_test_params()
            
            # 驗證必要參數
            if not os.path.exists(params['weights']):
                QMessageBox.warning(self, "錯誤", "權重檔案不存在！")
                return
                
            if not os.path.exists(params['data']):
                QMessageBox.warning(self, "錯誤", "資料配置檔案不存在！")
                return
            
            # 更新UI狀態
            self.test_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 無限進度條
            self.progress_label.setText("測試進行中...")
            
            # 清除之前的結果和日誌
            self.clear_results()
            self.log_text.clear()
            
            self.add_log("開始測試...")
            self.add_log(f"權重: {params['weights']}")
            self.add_log(f"資料配置: {params['data']}")
            self.add_log(f"任務類型: {params['task']}")
            self.add_log(f"批次大小: {params['batch_size']}")
            
            # 發送測試開始信號
            self.testing_started.emit()
            
            # 這裡將在後續實現與核心測試模組的連接
            self.add_log("注意: 核心測試功能尚未實現")
            
        except Exception as e:
            self.add_log(f"錯誤: {str(e)}")
            self.reset_ui_state()
            
    def stop_testing(self):
        """停止測試"""
        self.add_log("正在停止測試...")
        self.reset_ui_state()
        
    def reset_ui_state(self):
        """重置UI狀態"""
        self.test_btn.setEnabled(True)
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
            self.weights_input.clear()
            self.data_config_input.clear()
            
            # 重置為預設值
            self.imgsz_spinbox.setValue(640)
            self.batch_size_spinbox.setValue(32)
            self.conf_thres_spinbox.setValue(0.001)
            self.iou_thres_spinbox.setValue(0.6)
            self.max_det_spinbox.setValue(300)
            self.workers_spinbox.setValue(8)
            
            # 重置下拉選單
            self.device_combo.setCurrentIndex(0)
            self.task_combo.setCurrentText("val")
            
            # 重置核取方塊
            self.save_txt_checkbox.setChecked(False)
            self.save_conf_checkbox.setChecked(False)
            self.save_json_checkbox.setChecked(False)
            self.verbose_checkbox.setChecked(True)
            
            # 重置專案設定
            self.project_input.setText("runs/val")
            self.name_input.setText("exp")
            
            # 清除結果和日誌
            self.clear_results()
            self.log_text.clear()
            
            self.add_log("設定已清除")
            
    def clear_results(self):
        """清除測試結果"""
        # 清除總體結果
        self.mAP_label.setText("mAP@0.5: -")
        self.mAP50_95_label.setText("mAP@0.5:0.95: -")
        self.precision_label.setText("Precision: -")
        self.recall_label.setText("Recall: -")
        self.f1_label.setText("F1-Score: -")
        
        # 清除分類別結果表格
        self.results_table.setRowCount(0)
        
        # 清除混淆矩陣
        self.confusion_text.clear()
        
        # 清除速度測試結果
        self.preprocess_time_label.setText("預處理時間: -")
        self.inference_time_label.setText("推理時間: -")
        self.nms_time_label.setText("NMS時間: -")
        self.total_time_label.setText("總時間: -")
        
        self.test_results = {}
        
    def add_log(self, message):
        """添加日誌信息"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        self.log_message.emit(formatted_message)
        
    @pyqtSlot(int)
    def update_progress(self, progress):
        """更新進度條"""
        if self.progress_bar.maximum() != 0:
            self.progress_bar.setValue(progress)
            
    @pyqtSlot(dict)
    def update_results(self, results):
        """更新測試結果"""
        self.test_results = results
        
        # 更新總體結果
        if 'metrics/mAP_0.5' in results:
            self.mAP_label.setText(f"mAP@0.5: {results['metrics/mAP_0.5']:.4f}")
        if 'metrics/mAP_0.5:0.95' in results:
            self.mAP50_95_label.setText(f"mAP@0.5:0.95: {results['metrics/mAP_0.5:0.95']:.4f}")
        if 'metrics/precision' in results:
            self.precision_label.setText(f"Precision: {results['metrics/precision']:.4f}")
        if 'metrics/recall' in results:
            self.recall_label.setText(f"Recall: {results['metrics/recall']:.4f}")
        if 'metrics/f1' in results:
            self.f1_label.setText(f"F1-Score: {results['metrics/f1']:.4f}")
        
        # 更新分類別結果
        if 'class_results' in results:
            self.update_class_results(results['class_results'])
            
        # 更新混淆矩陣
        if 'confusion_matrix' in results:
            self.update_confusion_matrix(results['confusion_matrix'])
            
        # 更新速度測試結果
        if 'speed' in results:
            speed = results['speed']
            if 'preprocess' in speed:
                self.preprocess_time_label.setText(f"預處理時間: {speed['preprocess']:.2f}ms")
            if 'inference' in speed:
                self.inference_time_label.setText(f"推理時間: {speed['inference']:.2f}ms")
            if 'nms' in speed:
                self.nms_time_label.setText(f"NMS時間: {speed['nms']:.2f}ms")
            if 'total' in speed:
                self.total_time_label.setText(f"總時間: {speed['total']:.2f}ms")
                
    def update_class_results(self, class_results):
        """更新分類別結果表格"""
        self.results_table.setRowCount(len(class_results))
        
        for i, (class_name, metrics) in enumerate(class_results.items()):
            self.results_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(metrics.get('images', 0))))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(metrics.get('instances', 0))))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{metrics.get('precision', 0):.4f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{metrics.get('recall', 0):.4f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{metrics.get('mAP_0.5', 0):.4f}"))
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{metrics.get('mAP_0.5_0.95', 0):.4f}"))
            self.results_table.setItem(i, 7, QTableWidgetItem(f"{metrics.get('f1', 0):.4f}"))
            
    def update_confusion_matrix(self, confusion_matrix):
        """更新混淆矩陣顯示"""
        # 這裡可以實現混淆矩陣的文字或圖形顯示
        if isinstance(confusion_matrix, str):
            self.confusion_text.setPlainText(confusion_matrix)
        else:
            # 如果是數字矩陣，轉換為文字格式
            matrix_text = "混淆矩陣:\n"
            try:
                import numpy as np
                if isinstance(confusion_matrix, np.ndarray):
                    matrix_text += str(confusion_matrix)
                else:
                    matrix_text += str(confusion_matrix)
            except:
                matrix_text += str(confusion_matrix)
            
            self.confusion_text.setPlainText(matrix_text)
            
    @pyqtSlot(dict)
    def on_testing_finished(self, results):
        """測試完成處理"""
        self.reset_ui_state()
        self.update_results(results)
        self.add_log("測試完成！")
        self.testing_finished.emit(results)
        
        # 切換到結果頁籤
        self.tab_widget.setCurrentWidget(self.results_tab)
        
    @pyqtSlot(str)
    def on_testing_error(self, error_message):
        """測試錯誤處理"""
        self.reset_ui_state()
        self.add_log(f"測試錯誤: {error_message}")
        QMessageBox.critical(self, "測試錯誤", error_message)
        
    def export_results(self):
        """匯出測試結果"""
        if not self.test_results:
            QMessageBox.information(self, "提示", "沒有可匯出的測試結果")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "匯出測試結果", "",
            "JSON檔案 (*.json);;CSV檔案 (*.csv);;文字檔案 (*.txt)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    import json
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.test_results, f, indent=2, ensure_ascii=False)
                elif file_path.endswith('.csv'):
                    self.export_to_csv(file_path)
                else:  # txt
                    self.export_to_txt(file_path)
                    
                self.add_log(f"測試結果已匯出至: {file_path}")
                QMessageBox.information(self, "成功", f"測試結果已匯出至:\n{file_path}")
                
            except Exception as e:
                self.add_log(f"匯出失敗: {str(e)}")
                QMessageBox.critical(self, "錯誤", f"匯出失敗:\n{str(e)}")
                
    def export_to_csv(self, file_path):
        """匯出為CSV格式"""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 寫入標頭
            writer.writerow(['Metric', 'Value'])
            
            # 寫入總體結果
            for key, value in self.test_results.items():
                if not isinstance(value, dict):
                    writer.writerow([key, value])
                    
    def export_to_txt(self, file_path):
        """匯出為文字格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("YOLO-PCB 測試結果報告\n")
            f.write("=" * 50 + "\n\n")
            
            # 總體結果
            f.write("總體結果:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.test_results.items():
                if not isinstance(value, dict):
                    f.write(f"{key}: {value}\n")
            
            f.write("\n")
            
            # 分類別結果
            if 'class_results' in self.test_results:
                f.write("分類別結果:\n")
                f.write("-" * 20 + "\n")
                for class_name, metrics in self.test_results['class_results'].items():
                    f.write(f"\n{class_name}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value}\n")
