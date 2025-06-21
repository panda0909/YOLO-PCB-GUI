"""
簡化的YOLOv5模型載入器
專門處理模組命名衝突問題
"""

import os
import sys
import torch
from pathlib import Path


class YOLOv5Loader:
    """YOLOv5模型載入器，僅支援官方/YOLO-PCB結構"""
    
    def __init__(self):
        self.model = None
        self.device = None
    
    def load_model(self, weights_path, device='cpu'):
        """
        僅用YOLOv5/YOLO-PCB官方 attempt_load 載入模型
        
        Args:
            weights_path: 權重檔案路徑
            device: 運算設備 ('cpu', 'cuda')
        Returns:
            tuple: (success, model, error_message)
        """
        try:
            # 設置設備
            self.device = torch.device(device)
            # 解決 PyTorch 2.6 權重載入安全限制
            import numpy
            torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
            # Monkey patch torch.load 讓 weights_only=False
            orig_torch_load = torch.load
            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return orig_torch_load(*args, **kwargs)
            torch.load = patched_load
            yolo_pcb_root = Path(__file__).parent.parent  # 回到YOLO-PCB-GUI根目錄
            models_dir = yolo_pcb_root / 'models'
            if (models_dir / 'experimental.py').exists():
                sys.path.insert(0, str(yolo_pcb_root))
                try:
                    from models.experimental import attempt_load
                    model = attempt_load(weights_path, map_location=self.device)
                    if hasattr(model, 'float'):
                        model = model.float()
                    if hasattr(model, 'eval'):
                        model.eval()
                    self.model = model
                    return True, model, None
                except Exception as e:
                    return False, None, f"使用YOLOv5/YOLO-PCB attempt_load載入失敗: {str(e)}"
                finally:
                    sys.path.pop(0)
                    torch.load = orig_torch_load
            else:
                torch.load = orig_torch_load
                return False, None, "未找到YOLOv5/YOLO-PCB模型定義"
        except Exception as e:
            torch.load = orig_torch_load
            return False, None, str(e)
