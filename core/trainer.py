"""
訓練器核心模組
提供YOLO-PCB訓練功能的核心實現
"""

import os
import sys
import yaml
import json
import time
import torch
from pathlib import Path
from PyQt5.QtCore import QObject, QThread, pyqtSignal


class TrainingWorker(QThread):
    """訓練工作執行緒"""
    
    # 信號定義
    epoch_finished = pyqtSignal(dict)     # Epoch完成
    training_progress = pyqtSignal(int)   # 訓練進度
    training_stats = pyqtSignal(dict)     # 訓練統計
    log_message = pyqtSignal(str)         # 日誌訊息
    training_finished = pyqtSignal(str)   # 訓練完成
    error_occurred = pyqtSignal(str)      # 錯誤發生
    
    def __init__(self):
        super().__init__()
        self.params = {}
        self.training_active = False
        self.model = None
        self.device = None
        
    def set_parameters(self, params):
        """設置訓練參數"""
        self.params = params.copy()
        
    def run(self):
        """執行訓練"""
        try:
            self.training_active = True
            self.log_message.emit("開始初始化訓練...")
            
            # 驗證參數
            if not self._validate_parameters():
                return
            
            # 設置設備
            self._setup_device()
            
            # 準備資料配置
            if not self._prepare_data_config():
                return
            
            # 載入模型
            if not self._load_model():
                return
            
            # 執行訓練
            self._execute_training()
            
        except Exception as e:
            self.error_occurred.emit(f"訓練執行失敗: {str(e)}")
    
    def _validate_parameters(self):
        """驗證訓練參數"""
        try:
            # 檢查必要參數
            required_params = ['data', 'epochs', 'batch_size']
            for param in required_params:
                if param not in self.params:
                    self.error_occurred.emit(f"缺少必要參數: {param}")
                    return False
            
            # 檢查資料配置檔案
            data_config = self.params['data']
            if not os.path.exists(data_config):
                self.error_occurred.emit(f"資料配置檔案不存在: {data_config}")
                return False
            
            # 檢查權重檔案（如果指定）
            weights = self.params.get('weights', '')
            if weights and weights != 'random' and not os.path.exists(weights):
                self.log_message.emit(f"警告: 權重檔案不存在，將使用隨機初始化: {weights}")
            
            self.log_message.emit("參數驗證通過")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"參數驗證失敗: {str(e)}")
            return False
    
    def _setup_device(self):
        """設置計算設備"""
        try:
            device = self.params.get('device', 'auto')
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            self.log_message.emit(f"使用設備: {self.device}")
            
            # 如果使用GPU，顯示GPU資訊
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.log_message.emit(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                
        except Exception as e:
            self.error_occurred.emit(f"設備設置失敗: {str(e)}")
    
    def _prepare_data_config(self):
        """準備資料配置"""
        try:
            data_config_path = self.params['data']
            
            # 讀取現有配置
            with open(data_config_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            # 如果UI中指定了路徑，則更新配置
            if self.params.get('train_path'):
                data_config['train'] = self.params['train_path']
            if self.params.get('val_path'):
                data_config['val'] = self.params['val_path']
            
            # 更新類別數量和名稱
            if self.params.get('nc'):
                data_config['nc'] = self.params['nc']
            if self.params.get('names'):
                data_config['names'] = self.params['names']
            
            # 保存更新後的配置
            temp_config_path = os.path.join(os.path.dirname(data_config_path), 'temp_data.yaml')
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
            
            self.params['data'] = temp_config_path
            self.log_message.emit("資料配置準備完成")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"資料配置準備失敗: {str(e)}")
            return False
    
    def _load_model(self):
        """載入模型"""
        try:
            self.log_message.emit("正在載入模型...")
            
            # 這裡實現模型載入邏輯
            # 由於無法確定具體的YOLOv5實現，這裡提供一個示例
            
            # 嘗試使用torch.hub載入YOLOv5
            try:
                cfg = self.params.get('cfg', 'yolov5s.yaml')
                if cfg in ['yolov5n.yaml', 'yolov5s.yaml', 'yolov5m.yaml', 'yolov5l.yaml', 'yolov5x.yaml']:
                    model_name = cfg.replace('.yaml', '')
                    self.model = torch.hub.load('ultralytics/yolov5', model_name)
                else:
                    # 自定義配置需要特殊處理
                    self.log_message.emit("自定義模型配置需要進一步實現")
                    return False
                
                self.model.to(self.device)
                self.log_message.emit("模型載入成功")
                return True
                
            except Exception as e:
                self.log_message.emit(f"torch.hub載入失敗: {str(e)}")
                # 這裡可以實現其他載入方式
                self.error_occurred.emit("模型載入失敗，請檢查網路連接或使用本地模型")
                return False
                
        except Exception as e:
            self.error_occurred.emit(f"模型載入失敗: {str(e)}")
            return False
    
    def _execute_training(self):
        """執行訓練過程"""
        try:
            epochs = self.params['epochs']
            batch_size = self.params['batch_size']
            
            self.log_message.emit(f"開始訓練 - Epochs: {epochs}, Batch Size: {batch_size}")
            
            # 創建輸出目錄
            project = self.params.get('project', 'runs/train')
            name = self.params.get('name', 'exp')
            output_dir = os.path.join(project, name)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存訓練配置
            config_file = os.path.join(output_dir, 'training_config.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.params, f, indent=2, ensure_ascii=False)
            
            # 模擬訓練過程（實際實現需要根據具體的YOLOv5版本調整）
            self._simulate_training(epochs, output_dir)
            
        except Exception as e:
            self.error_occurred.emit(f"訓練執行失敗: {str(e)}")
    
    def _simulate_training(self, epochs, output_dir):
        """模擬訓練過程（示例實現）"""
        try:
            self.log_message.emit("注意: 這是訓練過程的模擬實現")
            self.log_message.emit("實際部署時需要整合真實的YOLOv5訓練代碼")
            
            # 模擬訓練統計
            best_map = 0.0
            
            for epoch in range(epochs):
                if not self.training_active:
                    self.log_message.emit("訓練被用戶中止")
                    break
                
                # 模擬訓練一個epoch
                start_time = time.time()
                
                # 模擬損失值（遞減）
                train_loss = max(0.1, 1.0 - (epoch * 0.01))
                val_loss = max(0.05, 0.8 - (epoch * 0.008))
                
                # 模擬精度指標（遞增）
                precision = min(0.95, 0.5 + (epoch * 0.005))
                recall = min(0.92, 0.4 + (epoch * 0.006))
                map50 = min(0.90, 0.3 + (epoch * 0.007))
                map50_95 = min(0.65, 0.2 + (epoch * 0.005))
                
                epoch_time = time.time() - start_time
                
                # 創建epoch統計
                epoch_stats = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'precision': precision,
                    'recall': recall,
                    'mAP@0.5': map50,
                    'mAP@0.5:0.95': map50_95,
                    'epoch_time': epoch_time
                }
                
                # 發送統計信息
                self.epoch_finished.emit(epoch_stats)
                self.training_stats.emit(epoch_stats)
                
                # 更新進度
                progress = int(((epoch + 1) / epochs) * 100)
                self.training_progress.emit(progress)
                
                # 日誌輸出
                self.log_message.emit(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}"
                )
                
                # 保存最佳模型
                if map50 > best_map:
                    best_map = map50
                    self._save_checkpoint(output_dir, epoch_stats, is_best=True)
                
                # 定期保存checkpoint
                if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                    self._save_checkpoint(output_dir, epoch_stats, is_best=False)
                
                # 模擬訓練時間
                self.msleep(100)  # 模擬每個epoch需要的時間
            
            # 訓練完成
            model_path = os.path.join(output_dir, 'weights', 'best.pt')
            self.training_finished.emit(model_path)
            self.log_message.emit(f"訓練完成！最佳模型保存至: {model_path}")
            
        except Exception as e:
            self.error_occurred.emit(f"訓練模擬失敗: {str(e)}")
    
    def _save_checkpoint(self, output_dir, epoch_stats, is_best=False):
        """保存訓練checkpoint"""
        try:
            weights_dir = os.path.join(output_dir, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch_stats['epoch'],
                'model_state_dict': None,  # 實際實現時需要保存模型狀態
                'optimizer_state_dict': None,  # 實際實現時需要保存優化器狀態
                'stats': epoch_stats,
                'params': self.params
            }
            
            # 保存最新checkpoint
            last_path = os.path.join(weights_dir, 'last.pt')
            torch.save(checkpoint, last_path)
            
            # 保存最佳模型
            if is_best:
                best_path = os.path.join(weights_dir, 'best.pt')
                torch.save(checkpoint, best_path)
                self.log_message.emit(f"保存最佳模型: mAP@0.5={epoch_stats['mAP@0.5']:.4f}")
                
        except Exception as e:
            self.log_message.emit(f"保存checkpoint失敗: {str(e)}")
    
    def stop_training(self):
        """停止訓練"""
        self.training_active = False
        self.log_message.emit("正在停止訓練...")


class Trainer(QObject):
    """訓練器主類別"""
    
    # 信號定義
    training_started = pyqtSignal()
    training_finished = pyqtSignal(str)
    training_progress = pyqtSignal(int)
    training_stats = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.worker = None
        
    def start_training(self, params):
        """開始訓練"""
        try:
            # 創建工作執行緒
            self.worker = TrainingWorker()
            self.worker.set_parameters(params)
            
            # 連接信號
            self.worker.epoch_finished.connect(self._on_epoch_finished)
            self.worker.training_progress.connect(self.training_progress.emit)
            self.worker.training_stats.connect(self.training_stats.emit)
            self.worker.log_message.connect(self.log_message.emit)
            self.worker.training_finished.connect(self.training_finished.emit)
            self.worker.error_occurred.connect(self.log_message.emit)
            
            # 開始訓練
            self.worker.start()
            self.training_started.emit()
            
            return True
            
        except Exception as e:
            self.log_message.emit(f"啟動訓練失敗: {str(e)}")
            return False
    
    def stop_training(self):
        """停止訓練"""
        if self.worker:
            self.worker.stop_training()
            if self.worker.isRunning():
                self.worker.quit()
                self.worker.wait(5000)  # 最多等待5秒
            self.worker = None
    
    def _on_epoch_finished(self, epoch_stats):
        """處理epoch完成事件"""
        # 這裡可以添加額外的處理邏輯
        pass


class TrainingConfig:
    """訓練配置管理器"""
    
    @staticmethod
    def create_data_config(config_dict, output_path):
        """創建資料配置檔案"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            print(f"創建資料配置失敗: {str(e)}")
            return False
    
    @staticmethod
    def load_data_config(config_path):
        """載入資料配置檔案"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"載入資料配置失敗: {str(e)}")
            return None
    
    @staticmethod
    def validate_data_paths(config_dict):
        """驗證資料路徑"""
        errors = []
        
        # 檢查訓練集路徑
        if 'train' in config_dict:
            train_path = config_dict['train']
            if not os.path.exists(train_path):
                errors.append(f"訓練集路徑不存在: {train_path}")
        else:
            errors.append("缺少訓練集路徑")
        
        # 檢查驗證集路徑
        if 'val' in config_dict:
            val_path = config_dict['val']
            if not os.path.exists(val_path):
                errors.append(f"驗證集路徑不存在: {val_path}")
        else:
            errors.append("缺少驗證集路徑")
        
        # 檢查類別配置
        if 'nc' not in config_dict:
            errors.append("缺少類別數量配置")
        
        if 'names' not in config_dict:
            errors.append("缺少類別名稱配置")
        elif len(config_dict['names']) != config_dict.get('nc', 0):
            errors.append("類別名稱數量與類別數量不匹配")
        
        return errors
