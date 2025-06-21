"""
簡化的YOLOv5模型載入器
專門處理模組命名衝突問題
"""

import os
import sys
import torch
from pathlib import Path


class YOLOv5Loader:
    """YOLOv5模型載入器，解決模組衝突"""
    
    def __init__(self):
        self.model = None
        self.device = None
    
    def load_model(self, weights_path, device='cpu'):
        """
        載入YOLOv5模型，避免模組衝突
        
        Args:
            weights_path: 權重檔案路徑
            device: 運算設備 ('cpu', 'cuda')
            
        Returns:
            tuple: (success, model, error_message)
        """
        try:
            # 設置設備
            self.device = torch.device(device)
            
            # 方法1: 嘗試直接載入權重檔案
            success, model, error = self._load_direct_weights(weights_path)
            if success:
                self.model = model
                return True, model, None
            
            # 方法2: 嘗試使用原始YOLO-PCB模型載入
            success, model, error = self._load_original_yolo_pcb(weights_path)
            if success:
                self.model = model
                return True, model, None
            
            return False, None, f"所有載入方法都失敗: {error}"
            
        except Exception as e:
            return False, None, str(e)
    
    def _load_direct_weights(self, weights_path):
        """直接載入權重檔案"""
        try:
            # 使用 weights_only=False 來避免PyTorch 2.6的安全限制
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            
            # 嘗試不同的鍵來找到模型
            model_keys = ['model', 'ema', 'model_state_dict', 'net', 'state_dict']
            
            for key in model_keys:
                if isinstance(checkpoint, dict) and key in checkpoint:
                    model = checkpoint[key]
                    
                    # 如果是ema，取其模型部分
                    if hasattr(model, 'ema'):
                        model = model.ema
                    elif hasattr(model, 'model'):
                        model = model.model
                    
                    # 設置模型屬性
                    if hasattr(model, 'float'):
                        model = model.float()
                    if hasattr(model, 'to'):
                        model = model.to(self.device)
                    if hasattr(model, 'eval'):
                        model.eval()
                    
                    return True, model, None
            
            # 如果沒有找到已知的鍵，嘗試直接使用checkpoint
            if hasattr(checkpoint, 'float'):
                model = checkpoint.float().to(self.device)
                if hasattr(model, 'eval'):
                    model.eval()
                return True, model, None
            
            return False, None, f"無法從權重檔案中找到模型，可用鍵: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'non-dict'}"
            
        except Exception as e:
            return False, None, str(e)
    
    def _load_original_yolo_pcb(self, weights_path):
        """嘗試使用原始YOLO-PCB項目的方式載入"""
        try:
            # 檢查是否存在原始YOLO-PCB項目
            yolo_pcb_root = Path(__file__).parent.parent.parent  # 回到YOLO-PCB根目錄
            
            if (yolo_pcb_root / 'models' / 'experimental.py').exists():
                # 添加原始YOLO-PCB路徑
                original_path = sys.path.copy()
                sys.path.insert(0, str(yolo_pcb_root))
                  try:
                    # 導入原始YOLO-PCB模型
                    from models.experimental import attempt_load
                    
                    # 修改torch.load的預設參數以支援舊版權重檔案
                    original_load = torch.load
                    def safe_load(*args, **kwargs):
                        if 'weights_only' not in kwargs:
                            kwargs['weights_only'] = False
                        return original_load(*args, **kwargs)
                    
                    # 臨時替換torch.load
                    torch.load = safe_load
                    
                    try:
                        model = attempt_load(weights_path, map_location=self.device)
                        if hasattr(model, 'float'):
                            model = model.float()
                        if hasattr(model, 'eval'):
                            model.eval()
                    finally:
                        # 恢復原始的torch.load
                        torch.load = original_load
                    
                    return True, model, None
                    
                except Exception as e:
                    return False, None, f"使用原始YOLO-PCB載入失敗: {str(e)}"
                    
                finally:
                    sys.path = original_path
            
            return False, None, "未找到原始YOLO-PCB模型定義"
            
        except Exception as e:
            return False, None, str(e)
