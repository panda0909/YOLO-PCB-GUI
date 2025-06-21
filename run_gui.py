#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-PCB GUI 啟動腳本
用於啟動和測試GUI應用程式

作者: AI Assistant
版本: 1.1.0
"""

import sys
import os
import traceback
from pathlib import Path

# 確保能找到模組
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def check_dependencies():
    """檢查依賴是否安裝"""
    required_packages = [
        ('PyQt5', 'PyQt5.QtWidgets'),
        ('torch', 'torch'),
        ('cv2', 'cv2'),
        ('numpy', 'numpy'),
        ('yaml', 'yaml'),
    ]
    
    missing_packages = []
    
    print("檢查依賴包...")
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name} 已安裝")
        except ImportError:
            print(f"✗ {package_name} 未安裝")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n缺少以下依賴包: {', '.join(missing_packages)}")
        print("請執行以下命令安裝:")
        print("pip install -r requirements_gui.txt")
        return False
    
    print("✓ 所有依賴檢查通過")
    return True

def create_sample_config():
    """創建範例配置文件"""
    try:
        config_dir = current_dir / 'resources' / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 創建必要目錄
        dirs_to_create = ['weights', 'data', 'runs', 'logs']
        for dir_name in dirs_to_create:
            (current_dir / dir_name).mkdir(exist_ok=True)
        
        print("✓ 目錄結構檢查完成")
        return True
        
    except Exception as e:
        print(f"創建配置失敗: {str(e)}")
        return False

def test_gui_imports():
    """測試GUI模組是否可正常導入"""
    try:
        print("測試GUI模組導入...")
        
        # 測試基本模組
        from yolo_gui_utils.config_manager import ConfigManager
        print("✓ 配置管理器")
        
        from gui.widgets.image_viewer import AnnotatedImageViewer
        print("✓ 圖像顯示組件")
        
        from gui.main_window import MainWindow
        print("✓ 主視窗")
        
        from gui.detect_tab import DetectTab
        from gui.train_tab import TrainTab
        from gui.test_tab import TestTab
        print("✓ 功能頁籤")
        
        from core.detector import DetectionWorker
        from core.trainer import TrainingWorker
        from core.tester import TestingWorker
        print("✓ 核心模組")
        
        print("✓ 所有GUI模組測試通過")
        return True
        
    except Exception as e:
        print(f"✗ GUI模組測試失敗: {str(e)}")
        print("將嘗試以兼容模式啟動...")
        return False

def main():
    """主函數"""
    print("YOLO-PCB GUI 啟動器")
    print("="*50)
    
    # 檢查依賴
    if not check_dependencies():
        return 1
    
    # 創建範例配置
    create_sample_config()
    
    # 測試GUI模組
    test_gui_imports()
    
    try:
        # 導入並啟動GUI
        print("\n正在啟動GUI應用程式...")
        
        from main import YoloPcbApp
        
        # 創建並運行應用程式
        app = YoloPcbApp()
        
        print("✓ GUI應用程式已初始化")
        print("正在顯示主視窗...")
        
        # 運行應用程式
        return app.run()
        
    except ImportError as e:
        print(f"\n✗ 模組導入失敗: {str(e)}")
        print("請確保所有必要的模組都已正確安裝")
        print("建議執行: pip install -r requirements_gui.txt")
        return 1
        
    except Exception as e:
        print(f"\n✗ 應用程式啟動失敗: {str(e)}")
        print("\n詳細錯誤信息:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n用戶中斷程序")
        sys.exit(0)
    except Exception as e:
        print(f"\n未預期的錯誤: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
