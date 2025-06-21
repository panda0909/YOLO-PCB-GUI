"""
獨立的YOLOv5模型載入器
專門處理模組命名衝突問題
"""

import os
import sys
import torch
import subprocess
import tempfile
from pathlib import Path


class YOLOv5Loader:
    """YOLOv5模型載入器，解決模組命名衝突"""
    
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
            
            # 方法1: 嘗試使用隔離的Python進程載入
            success, model, error = self._load_with_isolated_process(weights_path)
            if success:
                self.model = model
                return True, model, None
            
            # 方法2: 嘗試直接載入權重檔案
            success, model, error = self._load_direct_weights(weights_path)
            if success:
                self.model = model
                return True, model, None
            
            # 方法3: 嘗試使用原始YOLO-PCB模型載入
            success, model, error = self._load_original_yolo_pcb(weights_path)
            if success:
                self.model = model
                return True, model, None
            
            return False, None, f"所有載入方法都失敗: {error}"            
        except Exception as e:
            return False, None, str(e)
    
    def _load_with_isolated_process(self, weights_path):
        """使用隔離的Python進程載入模型"""
        try:
            # 創建臨時腳本來載入模型
            script_content = f'''
import torch
import sys
import os

# 移除可能衝突的路徑
gui_paths = [p for p in sys.path if 'YOLO-PCB-GUI' in p]
for path in gui_paths:
    if path in sys.path:
        sys.path.remove(path)

# 清除可能的模組衝突
conflict_modules = ['utils', 'models', 'utils.datasets', 'utils.general']
for module in conflict_modules:
    if module in sys.modules:
        del sys.modules[module]

try:
    # 載入YOLOv5
    if os.path.basename(r"{weights_path}") in ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
        model = torch.hub.load('ultralytics/yolov5', os.path.basename(r"{weights_path}").split('.')[0], device='cpu')
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', r"{weights_path}", device='cpu')
    
    # 保存模型到臨時檔案
    temp_path = r"{weights_path}.temp.pt"
    torch.save(model, temp_path)
    print("SUCCESS:" + temp_path)
    
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                temp_script = f.name
            
            try:
                # 設置乾淨的環境變數
                env = os.environ.copy()
                env['PYTHONPATH'] = ''  # 清空PYTHONPATH以避免模組衝突
                
                # 執行隔離腳本
                result = subprocess.run([sys.executable, temp_script], 
                                      capture_output=True, text=True, timeout=120, env=env)
                
                if result.returncode == 0 and "SUCCESS:" in result.stdout:
                    # 從輸出中提取臨時檔案路徑
                    for line in result.stdout.split('\n'):
                        if line.startswith('SUCCESS:'):
                            temp_model_path = line.split('SUCCESS:')[1].strip()
                            break
                    else:
                        temp_model_path = f"{weights_path}.temp.pt"
                    
                    if os.path.exists(temp_model_path):
                        model = torch.load(temp_model_path, map_location=self.device)
                        os.unlink(temp_model_path)  # 清理臨時檔案
                        return True, model, None
                
                error_msg = result.stderr if result.stderr else result.stdout
                return False, None, f"隔離進程載入失敗: {error_msg}"
                
            finally:
                # 清理臨時腳本
                if os.path.exists(temp_script):
                    os.unlink(temp_script)
                # 清理可能的臨時模型檔案
                temp_model_path = f"{weights_path}.temp.pt"
                if os.path.exists(temp_model_path):
                    os.unlink(temp_model_path)
                    
        except Exception as e:
            return False, None, str(e)
    
    def _load_direct_weights(self, weights_path):
        """直接載入權重檔案"""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            # 嘗試不同的鍵來找到模型
            model_keys = ['model', 'model_state_dict', 'net', 'state_dict']
            
            for key in model_keys:
                if key in checkpoint:
                    model = checkpoint[key]
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
            
            if (yolo_pcb_root / 'models' / 'yolo.py').exists():
                # 添加原始YOLO-PCB路徑
                original_path = sys.path.copy()
                sys.path.insert(0, str(yolo_pcb_root))
                
                try:
                    # 導入原始YOLO-PCB模型
                    from models.experimental import attempt_load
                    
                    model = attempt_load(weights_path, map_location=self.device)
                    model.float()
                    model.eval()
                    
                    return True, model, None
                    
                finally:
                    sys.path = original_path
            
            return False, None, "未找到原始YOLO-PCB模型定義"
            
        except Exception as e:
            return False, None, str(e)
