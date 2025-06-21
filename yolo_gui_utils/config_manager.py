#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模組
負責應用程式配置的載入、保存和管理

作者: AI Assistant
版本: 1.0.0
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器類別"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置檔案名稱
        """
        self.config_file = Path(config_file)
        self.config = {}
        self.default_config = self._get_default_config()
        self.load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """獲取預設配置"""
        return {
            "app": {
                "name": "YOLO-PCB GUI",
                "version": "1.0.0",
                "window_geometry": [100, 100, 1200, 800],
                "theme": "default",
                "language": "zh-TW"
            },
            "detection": {
                "confidence_threshold": 0.7,
                "iou_threshold": 0.45,
                "image_size": 640,
                "device": "auto",  # auto, cpu, cuda:0, cuda:1
                "save_results": True,
                "show_labels": True,
                "line_thickness": 3,
                "font_size": 12
            },
            "training": {
                "epochs": 100,
                "batch_size": 16,
                "learning_rate": 0.01,
                "image_size": 608,
                "workers": 4,
                "device": "auto",
                "cache": False,
                "save_period": 10,
                "patience": 20
            },
            "testing": {
                "batch_size": 32,
                "image_size": 608,
                "confidence_threshold": 0.001,
                "iou_threshold": 0.6,
                "save_results": True,
                "save_txt": True,
                "save_json": False,
                "verbose": True
            },
            "paths": {
                "weights_dir": "../weights",
                "data_dir": "../data",
                "results_dir": "./results",
                "models_dir": "../models",
                "default_weights": "../weights/baseline_fpn_loss.pt",
                "default_data": "../data/PCB.yaml"
            },
            "camera": {
                "default_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "backend": "auto",  # auto, dshow, v4l2
                "buffer_size": 1
            },
            "ui": {
                "show_toolbar": True,
                "show_statusbar": True,
                "auto_save_config": True,
                "confirm_exit": True,
                "recent_files_count": 10
            }
        }
    
    def load_config(self) -> bool:
        """
        載入配置檔案
        
        Returns:
            bool: 載入是否成功
        """
        try:
            if self.config_file.exists():
                logger.info(f"載入配置檔案: {self.config_file}")
                
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.suffix.lower() == '.yaml':
                        self.config = yaml.safe_load(f)
                    else:
                        self.config = json.load(f)
                
                # 合併預設配置（確保新增的配置項目存在）
                self.config = self._merge_configs(self.default_config, self.config)
                
                logger.info("配置檔案載入成功")
                return True
            else:
                logger.info("配置檔案不存在，使用預設配置")
                self.config = self.default_config.copy()
                self.save_config()
                return True
                
        except Exception as e:
            logger.error(f"載入配置檔案失敗: {str(e)}")
            logger.info("使用預設配置")
            self.config = self.default_config.copy()
            return False
    
    def save_config(self) -> bool:
        """
        保存配置檔案
        
        Returns:
            bool: 保存是否成功
        """
        try:
            logger.info(f"保存配置檔案: {self.config_file}")
            
            # 確保目錄存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.yaml':
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            logger.info("配置檔案保存成功")
            return True
            
        except Exception as e:
            logger.error(f"保存配置檔案失敗: {str(e)}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        獲取配置值
        
        Args:
            key_path: 配置鍵路徑，使用點號分隔，如 'detection.confidence_threshold'
            default: 預設值
            
        Returns:
            配置值
        """
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"獲取配置值失敗: {key_path}, 錯誤: {str(e)}")
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """
        設置配置值
        
        Args:
            key_path: 配置鍵路徑，使用點號分隔
            value: 配置值
            
        Returns:
            bool: 設置是否成功
        """
        try:
            keys = key_path.split('.')
            config = self.config
            
            # 導航到最後一層
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # 設置值
            config[keys[-1]] = value
            
            # 自動保存（如果啟用）
            if self.get('ui.auto_save_config', True):
                self.save_config()
            
            return True
            
        except Exception as e:
            logger.error(f"設置配置值失敗: {key_path}, 錯誤: {str(e)}")
            return False
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """
        合併配置，保留使用者配置並添加缺失的預設值
        
        Args:
            default: 預設配置
            user: 使用者配置
            
        Returns:
            合併後的配置
        """
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def reset_to_default(self) -> bool:
        """
        重置為預設配置
        
        Returns:
            bool: 重置是否成功
        """
        try:
            logger.info("重置配置為預設值")
            self.config = self.default_config.copy()
            return self.save_config()
        except Exception as e:
            logger.error(f"重置配置失敗: {str(e)}")
            return False
    
    def get_detection_config(self) -> Dict[str, Any]:
        """獲取檢測相關配置"""
        return self.get('detection', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """獲取訓練相關配置"""
        return self.get('training', {})
    
    def get_testing_config(self) -> Dict[str, Any]:
        """獲取測試相關配置"""
        return self.get('testing', {})
    
    def get_camera_config(self) -> Dict[str, Any]:
        """獲取攝影機相關配置"""
        return self.get('camera', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """獲取路徑相關配置"""
        return self.get('paths', {})
    
    def update_detection_config(self, config: Dict[str, Any]) -> bool:
        """更新檢測配置"""
        try:
            current_config = self.get_detection_config()
            current_config.update(config)
            return self.set('detection', current_config)
        except Exception as e:
            logger.error(f"更新檢測配置失敗: {str(e)}")
            return False
    
    def update_training_config(self, config: Dict[str, Any]) -> bool:
        """更新訓練配置"""
        try:
            current_config = self.get_training_config()
            current_config.update(config)
            return self.set('training', current_config)
        except Exception as e:
            logger.error(f"更新訓練配置失敗: {str(e)}")
            return False
    
    def update_testing_config(self, config: Dict[str, Any]) -> bool:
        """更新測試配置"""
        try:
            current_config = self.get_testing_config()
            current_config.update(config)
            return self.set('testing', current_config)
        except Exception as e:
            logger.error(f"更新測試配置失敗: {str(e)}")
            return False
