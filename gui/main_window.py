#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主視窗模組
YOLO-PCB GUI 應用程式的主視窗

作者: AI Assistant
版本: 1.0.0
"""

from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QMenuBar, QStatusBar, 
                            QAction, QMessageBox, QVBoxLayout, QWidget,
                            QToolBar, QLabel, QProgressBar, QHBoxLayout, QShortcut)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QKeySequence
# 暫時註解掉，稍後動態導入
# from gui.detect_tab import DetectTab
# from gui.train_tab import TrainTab  
# from gui.test_tab import TestTab
from yolo_gui_utils.config_manager import ConfigManager
import logging

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """主視窗類別"""
    
    # 自訂信號
    config_changed = pyqtSignal(str, object)  # 配置變更信號
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化主視窗
        
        Args:
            config_manager: 配置管理器
        """
        super().__init__()
        self.config_manager = config_manager
        self.detect_tab = None
        self.train_tab = None
        self.test_tab = None
        self.tab_widget = None
        
        # 狀態列元件
        self.status_label = None
        self.progress_bar = None
        self.memory_label = None
        
        # 計時器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)  # 每2秒更新一次狀態
        
        self.init_ui()
        self.init_menu()
        self.init_toolbar()
        self.init_statusbar()
        self.load_settings()
        # 全局 ESC 鍵快捷鍵，用於中斷檢測
        esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        esc_shortcut.setContext(Qt.ApplicationShortcut)
        esc_shortcut.activated.connect(self.on_global_escape)
        
        logger.info("主視窗初始化完成")
    
    def init_ui(self):
        """初始化使用者介面"""
        try:
            # 設置視窗基本屬性
            self.setWindowTitle("YOLO-PCB GUI - PCB 缺陷檢測系統")
            self.setMinimumSize(1000, 700)
            
            # 載入視窗幾何設定
            geometry = self.config_manager.get('app.window_geometry', [100, 100, 1200, 800])
            self.setGeometry(*geometry)
            
            # 創建中央頁籤控件
            self.tab_widget = QTabWidget()
            self.setCentralWidget(self.tab_widget)
            
            # 設置頁籤樣式
            self.tab_widget.setTabPosition(QTabWidget.North)
            self.tab_widget.setTabsClosable(False)
            self.tab_widget.setMovable(False)
            
            # 初始化各個功能頁籤
            self.init_tabs()            
            # 連接信號
            self.tab_widget.currentChanged.connect(self.on_tab_changed)
            
        except Exception as e:
            logger.error(f"初始化UI失敗: {str(e)}")
            raise
    
    def init_tabs(self):
        """初始化功能頁籤"""
        try:
            # 動態導入頁籤模組
            from gui.detect_tab import DetectTab
            from gui.train_tab import TrainTab  
            from gui.test_tab import TestTab
            
            # 檢測頁籤
            self.detect_tab = DetectTab(self)
            self.tab_widget.addTab(self.detect_tab, "🔍 PCB 檢測")
            
            # 測試頁籤
            self.test_tab = TestTab(self)
            self.tab_widget.addTab(self.test_tab, "📊 模型測試")
            
            # 訓練頁籤
            self.train_tab = TrainTab(self)
            self.tab_widget.addTab(self.train_tab, "🎓 模型訓練")
              # 連接頁籤信號
            self.connect_tab_signals()
            
            logger.info("功能頁籤初始化完成")
            
        except Exception as e:
            logger.error(f"初始化頁籤失敗: {str(e)}")
            # 如果頁籤導入失敗，創建一個簡單的佔位頁籤
            self.create_placeholder_tabs()
    
    def connect_tab_signals(self):
        """連接頁籤信號"""
        try:
            # 檢測頁籤信號
            if self.detect_tab:
                if hasattr(self.detect_tab, 'log_message'):
                    self.detect_tab.log_message.connect(self.show_status_message)
                if hasattr(self.detect_tab, 'detection_started'):
                    self.detect_tab.detection_started.connect(self.on_detection_started)
                if hasattr(self.detect_tab, 'detection_finished'):
                    self.detect_tab.detection_finished.connect(self.on_detection_finished)
            
            # 測試頁籤信號
            if self.test_tab:
                if hasattr(self.test_tab, 'log_message'):
                    self.test_tab.log_message.connect(self.show_status_message)
                if hasattr(self.test_tab, 'test_started'):
                    self.test_tab.test_started.connect(self.on_test_started)
                if hasattr(self.test_tab, 'test_finished'):
                    self.test_tab.test_finished.connect(self.on_test_finished)
            
            # 訓練頁籤信號
            if self.train_tab:
                if hasattr(self.train_tab, 'log_message'):
                    self.train_tab.log_message.connect(self.show_status_message)
                if hasattr(self.train_tab, 'training_started'):
                    self.train_tab.training_started.connect(self.on_training_started)
                if hasattr(self.train_tab, 'training_finished'):
                    self.train_tab.training_finished.connect(self.on_training_finished)
                if hasattr(self.train_tab, 'training_progress'):
                    self.train_tab.training_progress.connect(self.on_training_progress)
            
            logger.info("頁籤信號連接完成")
            
        except Exception as e:
            logger.error(f"連接頁籤信號失敗: {str(e)}")
    
    def on_detection_started(self):
        """檢測開始處理"""
        self.set_busy_status(True)
        self.show_status_message("檢測進行中...")
        if self.progress_bar:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 無限進度條
    
    def on_detection_finished(self, result_path):
        """檢測完成處理"""
        self.set_busy_status(False)
        self.show_status_message(f"檢測完成 - {result_path}")
        if self.progress_bar:
            self.progress_bar.setVisible(False)
        # 重置 ESC 快捷鍵提示
        self.statusBar().clearMessage()
    
    def on_global_escape(self):
        logger.info("全局 ESC 鍵被觸發")
        """全局 ESC 處理：如果正在攝像頭檢測則停止"""
        try:
            if self.detect_tab:
                # 停止檢測
                self.show_status_message("ESC 停止檢測", 3000)
                self.detect_tab.stop_detection()
        except Exception as e:
            logger.error(f"ESC 全局處理錯誤: {e}")
    
    def on_test_started(self):
        """測試開始處理"""
        self.set_busy_status(True)
        self.show_status_message("模型測試中...")
        if self.progress_bar:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
    
    def on_test_finished(self, result_path):
        """測試完成處理"""
        self.set_busy_status(False)
        self.show_status_message(f"測試完成 - {result_path}")
        if self.progress_bar:
            self.progress_bar.setVisible(False)
    
    def on_training_started(self):
        """訓練開始處理"""
        self.set_busy_status(True)
        self.show_status_message("模型訓練中...")
        if self.progress_bar:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
    
    def on_training_finished(self, result_path):
        """訓練完成處理"""
        self.set_busy_status(False)
        self.show_status_message(f"訓練完成 - {result_path}")
        if self.progress_bar:
            self.progress_bar.setVisible(False)
    
    def on_training_progress(self, progress):
        """訓練進度更新"""
        if self.progress_bar and self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(progress)
    
    def set_busy_status(self, busy):
        """設置忙碌狀態"""
        # 可以在這裡添加更多的忙碌狀態指示
        self.setEnabled(not busy) if busy else self.setEnabled(True)
    
    def create_placeholder_tabs(self):
        """創建佔位頁籤（當正常頁籤無法載入時使用）"""
        try:
            # 檢測頁籤佔位
            detect_placeholder = QWidget()
            detect_layout = QVBoxLayout(detect_placeholder)
            detect_label = QLabel("檢測功能模組載入失敗\n請檢查依賴是否正確安裝")
            detect_label.setAlignment(Qt.AlignCenter)
            detect_layout.addWidget(detect_label)
            self.tab_widget.addTab(detect_placeholder, "🔍 PCB 檢測")
            
            # 測試頁籤佔位
            test_placeholder = QWidget()
            test_layout = QVBoxLayout(test_placeholder)
            test_label = QLabel("測試功能模組載入失敗\n請檢查依賴是否正確安裝")
            test_label.setAlignment(Qt.AlignCenter)
            test_layout.addWidget(test_label)
            self.tab_widget.addTab(test_placeholder, "📊 模型測試")
            
            # 訓練頁籤佔位
            train_placeholder = QWidget()
            train_layout = QVBoxLayout(train_placeholder)
            train_label = QLabel("訓練功能模組載入失敗\n請檢查依賴是否正確安裝")
            train_label.setAlignment(Qt.AlignCenter)
            train_layout.addWidget(train_label)
            self.tab_widget.addTab(train_placeholder, "🎓 模型訓練")
            
            logger.warning("使用佔位頁籤替代正常功能頁籤")
            
        except Exception as e:
            logger.error(f"創建佔位頁籤失敗: {str(e)}")
    
    def init_menu(self):
        """初始化選單列"""
        try:
            menubar = self.menuBar()
            
            # 檔案選單
            file_menu = menubar.addMenu('檔案(&F)')
            
            # 新建專案
            new_action = QAction('新建專案(&N)', self)
            new_action.setShortcut(QKeySequence.New)
            new_action.triggered.connect(self.new_project)
            file_menu.addAction(new_action)
            
            # 開啟專案
            open_action = QAction('開啟專案(&O)', self)
            open_action.setShortcut(QKeySequence.Open)
            open_action.triggered.connect(self.open_project)
            file_menu.addAction(open_action)
            
            file_menu.addSeparator()
            
            # 設定
            settings_action = QAction('設定(&S)', self)
            settings_action.triggered.connect(self.open_settings)
            file_menu.addAction(settings_action)
            
            file_menu.addSeparator()
            
            # 退出
            exit_action = QAction('退出(&X)', self)
            exit_action.setShortcut(QKeySequence.Quit)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            
            # 檢視選單
            view_menu = menubar.addMenu('檢視(&V)')
            
            # 工具列
            toolbar_action = QAction('工具列(&T)', self)
            toolbar_action.setCheckable(True)
            toolbar_action.setChecked(self.config_manager.get('ui.show_toolbar', True))
            toolbar_action.triggered.connect(self.toggle_toolbar)
            view_menu.addAction(toolbar_action)
            
            # 狀態列
            statusbar_action = QAction('狀態列(&S)', self)
            statusbar_action.setCheckable(True)
            statusbar_action.setChecked(self.config_manager.get('ui.show_statusbar', True))
            statusbar_action.triggered.connect(self.toggle_statusbar)
            view_menu.addAction(statusbar_action)
            
            # 說明選單
            help_menu = menubar.addMenu('說明(&H)')
            
            # 關於
            about_action = QAction('關於(&A)', self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
            
            logger.info("選單列初始化完成")
            
        except Exception as e:
            logger.error(f"初始化選單失敗: {str(e)}")
    
    def init_toolbar(self):
        """初始化工具列"""
        try:
            if not self.config_manager.get('ui.show_toolbar', True):
                return
            
            toolbar = QToolBar('主工具列')
            self.addToolBar(toolbar)
            
            # 檢測按鈕
            detect_action = QAction('開始檢測', self)
            detect_action.triggered.connect(self.quick_detect)
            toolbar.addAction(detect_action)
            
            toolbar.addSeparator()
            
            # 設定按鈕
            settings_action = QAction('設定', self)
            settings_action.triggered.connect(self.open_settings)
            toolbar.addAction(settings_action)
            
            logger.info("工具列初始化完成")
            
        except Exception as e:
            logger.error(f"初始化工具列失敗: {str(e)}")
    
    def init_statusbar(self):
        """初始化狀態列"""
        try:
            if not self.config_manager.get('ui.show_statusbar', True):
                return
            
            statusbar = self.statusBar()
            
            # 主狀態標籤
            self.status_label = QLabel("就緒")
            statusbar.addWidget(self.status_label)
            
            # 添加進度條
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            self.progress_bar.setMaximumWidth(200)
            statusbar.addPermanentWidget(self.progress_bar)
            
            # 記憶體使用狀態
            self.memory_label = QLabel("記憶體: --")
            statusbar.addPermanentWidget(self.memory_label)
            
            # 版本資訊
            version_label = QLabel(f"v{self.config_manager.get('app.version', '1.0.0')}")
            statusbar.addPermanentWidget(version_label)
            
            logger.info("狀態列初始化完成")
            
        except Exception as e:
            logger.error(f"初始化狀態列失敗: {str(e)}")
    
    def load_settings(self):
        """載入設定"""
        try:
            # 載入主題（如果有的話）
            theme = self.config_manager.get('app.theme', 'default')
            if theme != 'default':
                self.apply_theme(theme)
            
            logger.info("設定載入完成")
            
        except Exception as e:
            logger.error(f"載入設定失敗: {str(e)}")
    
    def apply_theme(self, theme_name: str):
        """應用主題"""
        # TODO: 實現主題功能
        pass
    
    def on_tab_changed(self, index: int):
        """頁籤變更事件"""
        try:
            tab_names = ["檢測", "測試", "訓練"]
            if 0 <= index < len(tab_names):
                self.show_status_message(f"切換到 {tab_names[index]} 頁面")
                logger.info(f"切換到頁籤: {tab_names[index]}")
                
        except Exception as e:
            logger.error(f"頁籤變更處理失敗: {str(e)}")
    
    def show_status_message(self, message: str, timeout: int = 3000):
        """顯示狀態訊息"""
        if self.status_label:
            self.status_label.setText(message)
            if timeout > 0:
                QTimer.singleShot(timeout, lambda: self.status_label.setText("就緒"))
    
    def update_status(self):
        """更新狀態資訊"""
        try:
            # 更新記憶體使用情況
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if self.memory_label:
                self.memory_label.setText(f"記憶體: {memory_percent:.1f}%")
                
        except ImportError:
            # 如果沒有 psutil，顯示簡單資訊
            if self.memory_label:
                self.memory_label.setText("記憶體: --")
        except Exception as e:
            logger.error(f"更新狀態失敗: {str(e)}")
    
    def new_project(self):
        """新建專案"""
        # TODO: 實現新建專案功能
        QMessageBox.information(self, "新建專案", "新建專案功能開發中...")
    
    def open_project(self):
        """開啟專案"""
        # TODO: 實現開啟專案功能
        QMessageBox.information(self, "開啟專案", "開啟專案功能開發中...")
    
    def open_settings(self):
        """開啟設定對話框"""
        # TODO: 實現設定對話框
        QMessageBox.information(self, "設定", "設定功能開發中...")
    
    def quick_detect(self):
        """快速檢測"""
        if self.detect_tab:
            self.tab_widget.setCurrentWidget(self.detect_tab)
            # TODO: 觸發檢測功能
    
    def toggle_toolbar(self, checked: bool):
        """切換工具列顯示"""
        toolbar = self.findChild(QToolBar)
        if toolbar:
            toolbar.setVisible(checked)
        self.config_manager.set('ui.show_toolbar', checked)
    
    def toggle_statusbar(self, checked: bool):
        """切換狀態列顯示"""
        self.statusBar().setVisible(checked)
        self.config_manager.set('ui.show_statusbar', checked)
    
    def show_about(self):
        """顯示關於對話框"""
        about_text = f"""
        <h2>YOLO-PCB GUI</h2>
        <p><b>版本:</b> {self.config_manager.get('app.version', '1.0.0')}</p>
        <p><b>描述:</b> PCB 缺陷檢測系統圖形化介面</p>
        <p><b>基於:</b> YOLOv5 深度學習框架</p>
        <p><b>作者:</b> AI Assistant</p>
        <p><b>日期:</b> 2025-06-14</p>
        """
        QMessageBox.about(self, "關於 YOLO-PCB GUI", about_text)
    
    def closeEvent(self, event):
        """視窗關閉事件"""
        try:
            # 確認退出
            if self.config_manager.get('ui.confirm_exit', True):
                reply = QMessageBox.question(
                    self, '確認退出',
                    '確定要退出 YOLO-PCB GUI 嗎？',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply != QMessageBox.Yes:
                    event.ignore()
                    return
            
            # 保存視窗幾何設定
            geometry = [self.x(), self.y(), self.width(), self.height()]
            self.config_manager.set('app.window_geometry', geometry)
            
            # 停止計時器
            if self.status_timer.isActive():
                self.status_timer.stop()
            
            # 清理資源
            if self.detect_tab and hasattr(self.detect_tab, 'cleanup'):
                self.detect_tab.cleanup()
            if self.train_tab and hasattr(self.train_tab, 'cleanup'):
                self.train_tab.cleanup()
            if self.test_tab and hasattr(self.test_tab, 'cleanup'):
                self.test_tab.cleanup()
            
            logger.info("主視窗正常關閉")
            event.accept()
            
        except Exception as e:
            logger.error(f"關閉視窗時發生錯誤: {str(e)}")
            event.accept()  # 即使有錯誤也要關閉
