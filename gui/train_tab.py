"""
訓練頁籤模組
提供YOLO-PCB訓練功能的GUI介面
"""

import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox,
                             QFileDialog, QProgressBar, QComboBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QTabWidget, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QPixmap
import yaml


class TrainTab(QWidget):
    """訓練功能頁籤"""
      # 信號定義
    training_started = pyqtSignal()
    training_finished = pyqtSignal(str)  # 模型路徑
    training_stopped = pyqtSignal()
    training_progress = pyqtSignal(int)  # 進度百分比
    log_message = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.current_data_config = ""
        self.current_model_config = ""
        self.current_hyp_config = ""
        
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        """初始化使用者介面"""
        layout = QVBoxLayout(self)
        
        # 創建標籤頁
        self.tab_widget = QTabWidget()
        
        # 資料配置標籤頁
        self.data_tab = QWidget()
        self.create_data_config_tab()
        self.tab_widget.addTab(self.data_tab, "資料配置")
        
        # 模型配置標籤頁
        self.model_tab = QWidget()
        self.create_model_config_tab()
        self.tab_widget.addTab(self.model_tab, "模型配置")
        
        # 訓練參數標籤頁
        self.training_tab = QWidget()
        self.create_training_config_tab()
        self.tab_widget.addTab(self.training_tab, "訓練參數")
        
        # 控制面板標籤頁
        self.control_tab = QWidget()
        self.create_control_tab()
        self.tab_widget.addTab(self.control_tab, "訓練控制")
        
        layout.addWidget(self.tab_widget)
        
    def create_data_config_tab(self):
        """創建資料配置標籤頁"""
        layout = QVBoxLayout(self.data_tab)
        
        # 資料集配置群組
        dataset_group = QGroupBox("資料集配置")
        dataset_layout = QGridLayout(dataset_group)
        
        # 資料配置檔案
        self.data_config_label = QLabel("資料配置檔:")
        self.data_config_input = QLineEdit()
        self.data_config_input.setPlaceholderText("選擇.yaml資料配置檔案")
        self.browse_data_config_btn = QPushButton("瀏覽...")
        
        dataset_layout.addWidget(self.data_config_label, 0, 0)
        dataset_layout.addWidget(self.data_config_input, 0, 1)
        dataset_layout.addWidget(self.browse_data_config_btn, 0, 2)
        
        # 資料集路徑
        self.train_path_label = QLabel("訓練集路徑:")
        self.train_path_input = QLineEdit()
        self.browse_train_path_btn = QPushButton("瀏覽...")
        
        dataset_layout.addWidget(self.train_path_label, 1, 0)
        dataset_layout.addWidget(self.train_path_input, 1, 1)
        dataset_layout.addWidget(self.browse_train_path_btn, 1, 2)
        
        self.val_path_label = QLabel("驗證集路徑:")
        self.val_path_input = QLineEdit()
        self.browse_val_path_btn = QPushButton("瀏覽...")
        
        dataset_layout.addWidget(self.val_path_label, 2, 0)
        dataset_layout.addWidget(self.val_path_input, 2, 1)
        dataset_layout.addWidget(self.browse_val_path_btn, 2, 2)
        
        # 類別配置
        self.nc_label = QLabel("類別數量:")
        self.nc_spinbox = QSpinBox()
        self.nc_spinbox.setRange(1, 1000)
        self.nc_spinbox.setValue(6)  # PCB預設6個類別
        
        dataset_layout.addWidget(self.nc_label, 3, 0)
        dataset_layout.addWidget(self.nc_spinbox, 3, 1)
        
        # 類別名稱表格
        self.names_label = QLabel("類別名稱:")
        self.names_table = QTableWidget(6, 2)
        self.names_table.setHorizontalHeaderLabels(["類別ID", "類別名稱"])
        self.names_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 預設PCB類別
        default_names = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
        for i, name in enumerate(default_names):
            self.names_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.names_table.setItem(i, 1, QTableWidgetItem(name))
        
        dataset_layout.addWidget(self.names_label, 4, 0)
        dataset_layout.addWidget(self.names_table, 5, 0, 1, 3)
        
        layout.addWidget(dataset_group)
        
        # 資料預處理群組
        preprocess_group = QGroupBox("資料預處理")
        preprocess_layout = QGridLayout(preprocess_group)
        
        # 圖片大小
        self.img_size_label = QLabel("圖片大小:")
        self.img_size_spinbox = QSpinBox()
        self.img_size_spinbox.setRange(320, 1280)
        self.img_size_spinbox.setSingleStep(32)
        self.img_size_spinbox.setValue(640)
        
        preprocess_layout.addWidget(self.img_size_label, 0, 0)
        preprocess_layout.addWidget(self.img_size_spinbox, 0, 1)
        
        # 快取選項
        self.cache_checkbox = QCheckBox("啟用圖片快取")
        self.cache_checkbox.setToolTip("將圖片載入記憶體以加速訓練")
        
        preprocess_layout.addWidget(self.cache_checkbox, 1, 0, 1, 2)
        
        layout.addWidget(preprocess_group)
        layout.addStretch()
        
    def create_model_config_tab(self):
        """創建模型配置標籤頁"""
        layout = QVBoxLayout(self.model_tab)
        
        # 模型選擇群組
        model_group = QGroupBox("模型選擇")
        model_layout = QGridLayout(model_group)
        
        # 預訓練權重
        self.pretrained_label = QLabel("預訓練權重:")
        self.pretrained_combo = QComboBox()
        self.pretrained_combo.addItems([
            "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", 
            "yolov5l.pt", "yolov5x.pt", "自定義..."
        ])
        self.pretrained_combo.setCurrentText("yolov5s.pt")
        
        model_layout.addWidget(self.pretrained_label, 0, 0)
        model_layout.addWidget(self.pretrained_combo, 0, 1)
        
        # 自定義權重
        self.custom_weights_label = QLabel("自定義權重:")
        self.custom_weights_input = QLineEdit()
        self.custom_weights_input.setEnabled(False)
        self.browse_custom_weights_btn = QPushButton("瀏覽...")
        self.browse_custom_weights_btn.setEnabled(False)
        
        model_layout.addWidget(self.custom_weights_label, 1, 0)
        model_layout.addWidget(self.custom_weights_input, 1, 1)
        model_layout.addWidget(self.browse_custom_weights_btn, 1, 2)
        
        # 模型配置檔案
        self.model_config_label = QLabel("模型配置:")
        self.model_config_combo = QComboBox()
        self.model_config_combo.addItems([
            "yolov5n.yaml", "yolov5s.yaml", "yolov5m.yaml",
            "yolov5l.yaml", "yolov5x.yaml", "yolov5s_fpn.yaml", "自定義..."
        ])
        self.model_config_combo.setCurrentText("yolov5s.yaml")
        
        model_layout.addWidget(self.model_config_label, 2, 0)
        model_layout.addWidget(self.model_config_combo, 2, 1)
        
        # 自定義模型配置
        self.custom_model_label = QLabel("自定義配置:")
        self.custom_model_input = QLineEdit()
        self.custom_model_input.setEnabled(False)
        self.browse_custom_model_btn = QPushButton("瀏覽...")
        self.browse_custom_model_btn.setEnabled(False)
        
        model_layout.addWidget(self.custom_model_label, 3, 0)
        model_layout.addWidget(self.custom_model_input, 3, 1)
        model_layout.addWidget(self.browse_custom_model_btn, 3, 2)
        
        layout.addWidget(model_group)
        
        # 模型參數群組
        params_group = QGroupBox("模型參數")
        params_layout = QGridLayout(params_group)
        
        # 凍結層數
        self.freeze_label = QLabel("凍結層數:")
        self.freeze_spinbox = QSpinBox()
        self.freeze_spinbox.setRange(0, 24)
        self.freeze_spinbox.setValue(0)
        self.freeze_spinbox.setToolTip("凍結前N層的權重，0表示不凍結")
        
        params_layout.addWidget(self.freeze_label, 0, 0)
        params_layout.addWidget(self.freeze_spinbox, 0, 1)
        
        layout.addWidget(params_group)
        layout.addStretch()
        
    def create_training_config_tab(self):
        """創建訓練參數標籤頁"""
        layout = QVBoxLayout(self.training_tab)
        
        # 基本訓練參數群組
        basic_group = QGroupBox("基本訓練參數")
        basic_layout = QGridLayout(basic_group)
        
        # 訓練週期
        self.epochs_label = QLabel("訓練週期:")
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(300)
        
        basic_layout.addWidget(self.epochs_label, 0, 0)
        basic_layout.addWidget(self.epochs_spinbox, 0, 1)
        
        # 批次大小
        self.batch_size_label = QLabel("批次大小:")
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 128)
        self.batch_size_spinbox.setValue(16)
        
        basic_layout.addWidget(self.batch_size_label, 1, 0)
        basic_layout.addWidget(self.batch_size_spinbox, 1, 1)
        
        # 學習率
        self.lr_label = QLabel("初始學習率:")
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 1.0)
        self.lr_spinbox.setSingleStep(0.001)
        self.lr_spinbox.setValue(0.01)
        self.lr_spinbox.setDecimals(4)
        
        basic_layout.addWidget(self.lr_label, 2, 0)
        basic_layout.addWidget(self.lr_spinbox, 2, 1)
        
        # 最終學習率比例
        self.lrf_label = QLabel("最終學習率比例:")
        self.lrf_spinbox = QDoubleSpinBox()
        self.lrf_spinbox.setRange(0.01, 1.0)
        self.lrf_spinbox.setSingleStep(0.01)
        self.lrf_spinbox.setValue(0.1)
        self.lrf_spinbox.setDecimals(3)
        
        basic_layout.addWidget(self.lrf_label, 3, 0)
        basic_layout.addWidget(self.lrf_spinbox, 3, 1)
        
        # 動量
        self.momentum_label = QLabel("動量:")
        self.momentum_spinbox = QDoubleSpinBox()
        self.momentum_spinbox.setRange(0.0, 1.0)
        self.momentum_spinbox.setSingleStep(0.01)
        self.momentum_spinbox.setValue(0.937)
        self.momentum_spinbox.setDecimals(3)
        
        basic_layout.addWidget(self.momentum_label, 4, 0)
        basic_layout.addWidget(self.momentum_spinbox, 4, 1)
        
        # 權重衰減
        self.weight_decay_label = QLabel("權重衰減:")
        self.weight_decay_spinbox = QDoubleSpinBox()
        self.weight_decay_spinbox.setRange(0.0, 0.01)
        self.weight_decay_spinbox.setSingleStep(0.0001)
        self.weight_decay_spinbox.setValue(0.0005)
        self.weight_decay_spinbox.setDecimals(5)
        
        basic_layout.addWidget(self.weight_decay_label, 5, 0)
        basic_layout.addWidget(self.weight_decay_spinbox, 5, 1)
        
        layout.addWidget(basic_group)
        
        # 進階參數群組
        advanced_group = QGroupBox("進階參數")
        advanced_layout = QGridLayout(advanced_group)
        
        # Warmup週期
        self.warmup_epochs_label = QLabel("Warmup週期:")
        self.warmup_epochs_spinbox = QSpinBox()
        self.warmup_epochs_spinbox.setRange(0, 10)
        self.warmup_epochs_spinbox.setValue(3)
        
        advanced_layout.addWidget(self.warmup_epochs_label, 0, 0)
        advanced_layout.addWidget(self.warmup_epochs_spinbox, 0, 1)
        
        # Warmup動量
        self.warmup_momentum_label = QLabel("Warmup動量:")
        self.warmup_momentum_spinbox = QDoubleSpinBox()
        self.warmup_momentum_spinbox.setRange(0.0, 1.0)
        self.warmup_momentum_spinbox.setSingleStep(0.01)
        self.warmup_momentum_spinbox.setValue(0.8)
        self.warmup_momentum_spinbox.setDecimals(3)
        
        advanced_layout.addWidget(self.warmup_momentum_label, 1, 0)
        advanced_layout.addWidget(self.warmup_momentum_spinbox, 1, 1)
        
        # Warmup偏差學習率
        self.warmup_bias_lr_label = QLabel("Warmup偏差學習率:")
        self.warmup_bias_lr_spinbox = QDoubleSpinBox()
        self.warmup_bias_lr_spinbox.setRange(0.0, 1.0)
        self.warmup_bias_lr_spinbox.setSingleStep(0.01)
        self.warmup_bias_lr_spinbox.setValue(0.1)
        self.warmup_bias_lr_spinbox.setDecimals(3)
        
        advanced_layout.addWidget(self.warmup_bias_lr_label, 2, 0)
        advanced_layout.addWidget(self.warmup_bias_lr_spinbox, 2, 1)
        
        layout.addWidget(advanced_group)
        
        # 設備和優化群組
        device_group = QGroupBox("設備和優化")
        device_layout = QGridLayout(device_group)
        
        # 計算設備
        self.device_label = QLabel("計算設備:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda:0"])
        
        device_layout.addWidget(self.device_label, 0, 0)
        device_layout.addWidget(self.device_combo, 0, 1)
        
        # 工作線程數
        self.workers_label = QLabel("工作線程數:")
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setRange(0, 16)
        self.workers_spinbox.setValue(8)
        
        device_layout.addWidget(self.workers_label, 1, 0)
        device_layout.addWidget(self.workers_spinbox, 1, 1)
        
        # 混合精度訓練
        self.amp_checkbox = QCheckBox("啟用混合精度訓練 (AMP)")
        self.amp_checkbox.setToolTip("使用FP16混合精度加速訓練")
        
        device_layout.addWidget(self.amp_checkbox, 2, 0, 1, 2)
        
        layout.addWidget(device_group)
        layout.addStretch()
        
    def create_control_tab(self):
        """創建控制標籤頁"""
        layout = QVBoxLayout(self.control_tab)
        
        # 輸出設定群組
        output_group = QGroupBox("輸出設定")
        output_layout = QGridLayout(output_group)
        
        # 專案名稱
        self.project_label = QLabel("專案名稱:")
        self.project_input = QLineEdit()
        self.project_input.setText("runs/train")
        
        output_layout.addWidget(self.project_label, 0, 0)
        output_layout.addWidget(self.project_input, 0, 1)
        
        # 實驗名稱
        self.name_label = QLabel("實驗名稱:")
        self.name_input = QLineEdit()
        self.name_input.setText("exp")
        
        output_layout.addWidget(self.name_label, 1, 0)
        output_layout.addWidget(self.name_input, 1, 1)
        
        # 保存選項
        self.save_period_label = QLabel("保存週期:")
        self.save_period_spinbox = QSpinBox()
        self.save_period_spinbox.setRange(-1, 1000)
        self.save_period_spinbox.setValue(-1)
        self.save_period_spinbox.setToolTip("-1表示僅保存最後一個epoch，0表示不保存")
        
        output_layout.addWidget(self.save_period_label, 2, 0)
        output_layout.addWidget(self.save_period_spinbox, 2, 1)
        
        layout.addWidget(output_group)
        
        # 控制按鈕群組
        control_group = QGroupBox("訓練控制")
        control_layout = QVBoxLayout(control_group)
        
        # 按鈕區域
        button_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("開始訓練")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.setStyleSheet("""
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
        
        self.stop_btn = QPushButton("停止訓練")
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
        
        self.resume_btn = QPushButton("恢復訓練")
        self.resume_btn.setMinimumHeight(40)
        self.resume_btn.setToolTip("從上次中斷的地方恢復訓練")
        
        self.clear_btn = QPushButton("清除設定")
        self.clear_btn.setMinimumHeight(40)
        
        button_layout.addWidget(self.train_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.resume_btn)
        button_layout.addWidget(self.clear_btn)
        
        # 進度顯示
        self.progress_label = QLabel("就緒")
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setVisible(False)
        
        self.overall_progress_label = QLabel("總體進度:")
        self.overall_progress = QProgressBar()
        self.overall_progress.setVisible(False)
        
        # 訓練統計
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(100)
        self.stats_text.setPlaceholderText("訓練統計信息將顯示在此...")
        
        # 日誌輸出
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setPlaceholderText("訓練日誌將顯示在此...")
        font = QFont("Consolas", 9)
        self.log_text.setFont(font)
        
        control_layout.addLayout(button_layout)
        control_layout.addWidget(self.progress_label)
        control_layout.addWidget(self.epoch_progress)
        control_layout.addWidget(self.overall_progress_label)
        control_layout.addWidget(self.overall_progress)
        control_layout.addWidget(QLabel("訓練統計:"))
        control_layout.addWidget(self.stats_text)
        control_layout.addWidget(QLabel("訓練日誌:"))
        control_layout.addWidget(self.log_text)
        
        layout.addWidget(control_group)
        
    def connect_signals(self):
        """連接信號和槽"""
        # 瀏覽按鈕
        self.browse_data_config_btn.clicked.connect(self.browse_data_config)
        self.browse_train_path_btn.clicked.connect(self.browse_train_path)
        self.browse_val_path_btn.clicked.connect(self.browse_val_path)
        self.browse_custom_weights_btn.clicked.connect(self.browse_custom_weights)
        self.browse_custom_model_btn.clicked.connect(self.browse_custom_model)
        
        # 下拉選單變化
        self.pretrained_combo.currentTextChanged.connect(self.on_pretrained_changed)
        self.model_config_combo.currentTextChanged.connect(self.on_model_config_changed)
        
        # 類別數量變化
        self.nc_spinbox.valueChanged.connect(self.update_names_table)
        
        # 控制按鈕
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.resume_btn.clicked.connect(self.resume_training)
        self.clear_btn.clicked.connect(self.clear_settings)
        
        # 輸入驗證
        self.data_config_input.textChanged.connect(self.validate_inputs)
        
    def browse_data_config(self):
        """瀏覽資料配置檔案"""
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇資料配置檔案", "",
            "YAML檔案 (*.yaml *.yml);;所有檔案 (*)"
        )
        
        if path:
            self.data_config_input.setText(path)
            self.load_data_config(path)
            
    def browse_train_path(self):
        """瀏覽訓練集路徑"""
        path = QFileDialog.getExistingDirectory(
            self, "選擇訓練集資料夾", ""
        )
        
        if path:
            self.train_path_input.setText(path)
            
    def browse_val_path(self):
        """瀏覽驗證集路徑"""
        path = QFileDialog.getExistingDirectory(
            self, "選擇驗證集資料夾", ""
        )
        
        if path:
            self.val_path_input.setText(path)
            
    def browse_custom_weights(self):
        """瀏覽自定義權重檔案"""
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇權重檔案", "",
            "PyTorch權重 (*.pt *.pth);;所有檔案 (*)"
        )
        
        if path:
            self.custom_weights_input.setText(path)
            
    def browse_custom_model(self):
        """瀏覽自定義模型配置"""
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇模型配置檔案", "",
            "YAML檔案 (*.yaml *.yml);;所有檔案 (*)"
        )
        
        if path:
            self.custom_model_input.setText(path)
            
    def on_pretrained_changed(self, text):
        """處理預訓練權重選擇變化"""
        is_custom = text == "自定義..."
        self.custom_weights_input.setEnabled(is_custom)
        self.browse_custom_weights_btn.setEnabled(is_custom)
        
    def on_model_config_changed(self, text):
        """處理模型配置選擇變化"""
        is_custom = text == "自定義..."
        self.custom_model_input.setEnabled(is_custom)
        self.browse_custom_model_btn.setEnabled(is_custom)
        
    def update_names_table(self, count):
        """更新類別名稱表格"""
        self.names_table.setRowCount(count)
        
        for i in range(count):
            if not self.names_table.item(i, 0):
                self.names_table.setItem(i, 0, QTableWidgetItem(str(i)))
            if not self.names_table.item(i, 1):
                self.names_table.setItem(i, 1, QTableWidgetItem(f"class_{i}"))
                
    def load_data_config(self, config_path):
        """載入資料配置檔案"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # 更新UI
            if 'train' in config:
                self.train_path_input.setText(str(config['train']))
            if 'val' in config:
                self.val_path_input.setText(str(config['val']))
            if 'nc' in config:
                self.nc_spinbox.setValue(config['nc'])
            if 'names' in config:
                names = config['names']
                self.update_names_table(len(names))
                for i, name in enumerate(names):
                    self.names_table.setItem(i, 1, QTableWidgetItem(name))
                    
            self.add_log(f"已載入資料配置: {config_path}")
            
        except Exception as e:
            self.add_log(f"載入資料配置失敗: {str(e)}")
            QMessageBox.warning(self, "錯誤", f"無法載入資料配置檔案:\n{str(e)}")
            
    def validate_inputs(self):
        """驗證輸入參數"""
        data_config_valid = bool(self.data_config_input.text().strip())
        self.train_btn.setEnabled(data_config_valid)
        
    def get_training_params(self):
        """獲取訓練參數"""
        # 獲取類別名稱
        names = []
        for i in range(self.names_table.rowCount()):
            name_item = self.names_table.item(i, 1)
            if name_item:
                names.append(name_item.text())
            else:
                names.append(f"class_{i}")
        
        # 獲取權重路徑
        if self.pretrained_combo.currentText() == "自定義...":
            weights = self.custom_weights_input.text().strip()
        else:
            weights = self.pretrained_combo.currentText()
            
        # 獲取模型配置
        if self.model_config_combo.currentText() == "自定義...":
            cfg = self.custom_model_input.text().strip()
        else:
            cfg = self.model_config_combo.currentText()
        
        params = {
            # 資料參數
            'data': self.data_config_input.text().strip(),
            'train_path': self.train_path_input.text().strip(),
            'val_path': self.val_path_input.text().strip(),
            'nc': self.nc_spinbox.value(),
            'names': names,
            
            # 模型參數
            'weights': weights,
            'cfg': cfg,
            'freeze': self.freeze_spinbox.value(),
            
            # 訓練參數
            'epochs': self.epochs_spinbox.value(),
            'batch_size': self.batch_size_spinbox.value(),
            'imgsz': self.img_size_spinbox.value(),
            'lr0': self.lr_spinbox.value(),
            'lrf': self.lrf_spinbox.value(),
            'momentum': self.momentum_spinbox.value(),
            'weight_decay': self.weight_decay_spinbox.value(),
            'warmup_epochs': self.warmup_epochs_spinbox.value(),
            'warmup_momentum': self.warmup_momentum_spinbox.value(),
            'warmup_bias_lr': self.warmup_bias_lr_spinbox.value(),
            
            # 設備參數
            'device': self.device_combo.currentText(),
            'workers': self.workers_spinbox.value(),
            'amp': self.amp_checkbox.isChecked(),
            
            # 輸出參數
            'project': self.project_input.text().strip(),
            'name': self.name_input.text().strip(),
            'save_period': self.save_period_spinbox.value(),
            
            # 其他選項
            'cache': self.cache_checkbox.isChecked(),
        }
        
        return params
        
    def start_training(self):
        """開始訓練"""
        try:
            params = self.get_training_params()
            
            # 驗證必要參數
            if not params['data'] or not os.path.exists(params['data']):
                QMessageBox.warning(self, "錯誤", "請選擇有效的資料配置檔案！")
                return
            
            # 更新UI狀態
            self.train_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.epoch_progress.setVisible(True)
            self.overall_progress.setVisible(True)
            self.epoch_progress.setRange(0, params['epochs'])
            self.overall_progress.setRange(0, 100)
            self.progress_label.setText("正在準備訓練...")
            
            # 清除之前的日誌
            self.log_text.clear()
            self.stats_text.clear()
            
            self.add_log("開始訓練...")
            self.add_log(f"資料配置: {params['data']}")
            self.add_log(f"模型: {params['cfg']}")
            self.add_log(f"權重: {params['weights']}")
            self.add_log(f"訓練週期: {params['epochs']}")
            self.add_log(f"批次大小: {params['batch_size']}")
            
            # 發送訓練開始信號
            self.training_started.emit()
            
            # 這裡將在後續實現與核心訓練模組的連接
            self.add_log("注意: 核心訓練功能尚未實現")
            
        except Exception as e:
            self.add_log(f"錯誤: {str(e)}")
            self.reset_ui_state()
            
    def stop_training(self):
        """停止訓練"""
        self.add_log("正在停止訓練...")
        self.training_stopped.emit()
        self.reset_ui_state()
        
    def resume_training(self):
        """恢復訓練"""
        # 尋找最近的checkpoint
        project_path = self.project_input.text().strip()
        name = self.name_input.text().strip()
        
        if project_path and name:
            resume_path = os.path.join(project_path, name, "weights", "last.pt")
            if os.path.exists(resume_path):
                self.add_log(f"從checkpoint恢復訓練: {resume_path}")
                # 這裡實現恢復訓練的邏輯
            else:
                QMessageBox.information(self, "提示", "未找到可恢復的訓練checkpoint")
        else:
            QMessageBox.warning(self, "錯誤", "請設定專案名稱和實驗名稱")
            
    def reset_ui_state(self):
        """重置UI狀態"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.epoch_progress.setVisible(False)
        self.overall_progress.setVisible(False)
        self.progress_label.setText("就緒")
        
    def clear_settings(self):
        """清除所有設定"""
        reply = QMessageBox.question(
            self, "確認", "確定要清除所有設定嗎？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 清除輸入
            self.data_config_input.clear()
            self.train_path_input.clear()
            self.val_path_input.clear()
            self.custom_weights_input.clear()
            self.custom_model_input.clear()
            
            # 重置為預設值
            self.nc_spinbox.setValue(6)
            self.img_size_spinbox.setValue(640)
            self.epochs_spinbox.setValue(300)
            self.batch_size_spinbox.setValue(16)
            self.lr_spinbox.setValue(0.01)
            self.lrf_spinbox.setValue(0.1)
            self.momentum_spinbox.setValue(0.937)
            self.weight_decay_spinbox.setValue(0.0005)
            self.warmup_epochs_spinbox.setValue(3)
            self.warmup_momentum_spinbox.setValue(0.8)
            self.warmup_bias_lr_spinbox.setValue(0.1)
            self.freeze_spinbox.setValue(0)
            self.workers_spinbox.setValue(8)
            self.save_period_spinbox.setValue(-1)
            
            # 重置下拉選單
            self.pretrained_combo.setCurrentText("yolov5s.pt")
            self.model_config_combo.setCurrentText("yolov5s.yaml")
            self.device_combo.setCurrentIndex(0)
            
            # 重置核取方塊
            self.cache_checkbox.setChecked(False)
            self.amp_checkbox.setChecked(False)
            
            # 重置專案設定
            self.project_input.setText("runs/train")
            self.name_input.setText("exp")
            
            # 重置類別表格
            self.update_names_table(6)
            default_names = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
            for i, name in enumerate(default_names):
                self.names_table.setItem(i, 1, QTableWidgetItem(name))
            
            # 清除日誌
            self.log_text.clear()
            self.stats_text.clear()
            
            self.add_log("設定已清除")
            
    def add_log(self, message):
        """添加日誌信息"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        self.log_message.emit(formatted_message)
        
    @pyqtSlot(int)
    def update_epoch_progress(self, epoch):
        """更新epoch進度"""
        self.epoch_progress.setValue(epoch)
        self.progress_label.setText(f"訓練中... Epoch {epoch}/{self.epochs_spinbox.value()}")
        
    @pyqtSlot(int)
    def update_overall_progress(self, progress):
        """更新總體進度"""
        self.overall_progress.setValue(progress)
        
    @pyqtSlot(dict)
    def update_training_stats(self, stats):
        """更新訓練統計"""
        stats_text = ""
        for key, value in stats.items():
            stats_text += f"{key}: {value}\n"
        self.stats_text.setPlainText(stats_text)
        
    @pyqtSlot(str)
    def on_training_finished(self, model_path):
        """訓練完成處理"""
        self.reset_ui_state()
        self.add_log(f"訓練完成！模型保存至: {model_path}")
        self.training_finished.emit(model_path)
        
    @pyqtSlot(str)
    def on_training_error(self, error_message):
        """訓練錯誤處理"""
        self.reset_ui_state()
        self.add_log(f"訓練錯誤: {error_message}")
        QMessageBox.critical(self, "訓練錯誤", error_message)
