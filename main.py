#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-PCB GUI 主程式
PCB 缺陷檢測視窗應用程式

作者: AI Assistant
版本: 1.0.0
日期: 2025-06-14
"""

import sys
import os
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 添加原始 YOLO-PCB 專案路徑
yolo_pcb_root = project_root.parent
sys.path.insert(0, str(yolo_pcb_root))

from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor

try:
    from gui.main_window import MainWindow
    from yolo_gui_utils.config_manager import ConfigManager
    GUI_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"GUI模組導入失敗: {str(e)}")
    GUI_MODULES_AVAILABLE = False
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_pcb_gui.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class YoloPcbApp:
    """YOLO-PCB GUI 應用程式主類"""
    
    def __init__(self):
        self.app = None
        self.main_window = None
        self.config_manager = None
        self.splash = None
    
    def create_splash_screen(self):
        """創建啟動畫面"""
        # 創建啟動畫面圖像
        pixmap = QPixmap(400, 300)
        pixmap.fill(QColor(45, 45, 45))
        
        # 在圖像上繪製文字
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 255, 255))
        
        # 設置字體
        title_font = QFont("Arial", 24, QFont.Bold)
        subtitle_font = QFont("Arial", 12)
        
        # 繪製標題
        painter.setFont(title_font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "YOLO-PCB GUI")
        
        # 繪製副標題
        painter.setFont(subtitle_font)
        painter.drawText(50, 200, "PCB 缺陷檢測系統")
        painter.drawText(50, 220, "正在載入...")
        painter.drawText(50, 260, "版本 1.0.0")
        
        painter.end()
        
        # 創建啟動畫面
        self.splash = QSplashScreen(pixmap)
        self.splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.splash.show()
    
    def init_application(self):
        """初始化應用程式"""
        try:
            logger.info("正在初始化 YOLO-PCB GUI 應用程式...")
            
            # 創建 QApplication
            self.app = QApplication(sys.argv)
            self.app.setApplicationName("YOLO-PCB GUI")
            self.app.setApplicationVersion("1.0.0")
            self.app.setOrganizationName("YOLO-PCB")
            
            # 設置應用程式圖示（如果有的話）
            # self.app.setWindowIcon(QIcon("resources/icons/app_icon.png"))
            
            # 創建啟動畫面
            self.create_splash_screen()
            
            # 處理啟動畫面事件
            self.app.processEvents()
            
            # 載入配置
            self.splash.showMessage("載入配置檔案...", Qt.AlignBottom | Qt.AlignCenter, QColor(255, 255, 255))
            self.app.processEvents()
            
            self.config_manager = ConfigManager()
            
            # 創建主視窗
            self.splash.showMessage("初始化主視窗...", Qt.AlignBottom | Qt.AlignCenter, QColor(255, 255, 255))
            self.app.processEvents()
            
            self.main_window = MainWindow(self.config_manager)
            
            # 延遲顯示主視窗
            QTimer.singleShot(2000, self.show_main_window)
            
            logger.info("應用程式初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"應用程式初始化失敗: {str(e)}")
            if self.splash:
                self.splash.close()
            QMessageBox.critical(None, "初始化錯誤", f"應用程式初始化失敗:\n{str(e)}")
            return False
    
    def show_main_window(self):
        """顯示主視窗"""
        try:
            if self.splash:
                self.splash.close()
            
            self.main_window.show()
            logger.info("主視窗已顯示")
            
        except Exception as e:
            logger.error(f"顯示主視窗失敗: {str(e)}")
            QMessageBox.critical(None, "顯示錯誤", f"無法顯示主視窗:\n{str(e)}")
    
    def run(self):
        """運行應用程式"""
        if not self.init_application():
            return 1
        
        try:
            # 進入事件循環
            return self.app.exec_()
        except Exception as e:
            logger.error(f"應用程式運行錯誤: {str(e)}")
            return 1

def main():
    """主函數"""
    try:
        # 檢查 Python 版本
        if sys.version_info < (3, 7):
            print("錯誤：需要 Python 3.7 或更高版本")
            return 1
        
        # 創建並運行應用程式
        app = YoloPcbApp()
        return app.run()
        
    except KeyboardInterrupt:
        logger.info("使用者中斷程式")
        return 0
    except Exception as e:
        logger.error(f"未處理的錯誤: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
