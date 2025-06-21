"""
測試器核心模組
提供YOLO-PCB測試功能的核心實現
"""

import os
import sys
import yaml
import json
import time
import torch
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject, QThread, pyqtSignal


class TestingWorker(QThread):
    """測試工作執行緒"""
    
    # 信號定義
    testing_progress = pyqtSignal(int)    # 測試進度
    testing_stats = pyqtSignal(dict)      # 測試統計
    log_message = pyqtSignal(str)         # 日誌訊息
    testing_finished = pyqtSignal(dict)   # 測試完成
    error_occurred = pyqtSignal(str)      # 錯誤發生
    
    def __init__(self):
        super().__init__()
        self.params = {}
        self.testing_active = False
        self.model = None
        self.device = None
        self.results = {}
        
    def set_parameters(self, params):
        """設置測試參數"""
        self.params = params.copy()
        
    def run(self):
        """執行測試"""
        try:
            self.testing_active = True
            self.log_message.emit("開始初始化測試...")
            
            # 驗證參數
            if not self._validate_parameters():
                return
            
            # 設置設備
            self._setup_device()
            
            # 載入模型
            if not self._load_model():
                return
            
            # 載入資料配置
            if not self._load_data_config():
                return
            
            # 執行測試
            self._execute_testing()
            
        except Exception as e:
            self.error_occurred.emit(f"測試執行失敗: {str(e)}")
    
    def _validate_parameters(self):
        """驗證測試參數"""
        try:
            # 檢查必要參數
            required_params = ['weights', 'data']
            for param in required_params:
                if param not in self.params:
                    self.error_occurred.emit(f"缺少必要參數: {param}")
                    return False
            
            # 檢查權重檔案
            weights = self.params['weights']
            if not os.path.exists(weights):
                self.error_occurred.emit(f"權重檔案不存在: {weights}")
                return False
            
            # 檢查資料配置檔案
            data_config = self.params['data']
            if not os.path.exists(data_config):
                self.error_occurred.emit(f"資料配置檔案不存在: {data_config}")
                return False
            
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
    
    def _load_model(self):
        """載入模型"""
        try:
            weights_path = self.params['weights']
            self.log_message.emit(f"正在載入模型: {weights_path}")
            
            # 嘗試載入YOLOv5模型
            try:
                if os.path.basename(weights_path) in ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    self.model = torch.hub.load('ultralytics/yolov5', os.path.basename(weights_path).split('.')[0])
                else:
                    # 載入自定義權重
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)
                
                self.model.to(self.device)
                self.model.eval()
                self.log_message.emit("模型載入成功")
                return True
                
            except Exception as e:
                self.log_message.emit(f"YOLOv5載入失敗: {str(e)}")
                # 備選方案：直接載入權重檔案
                try:
                    checkpoint = torch.load(weights_path, map_location=self.device)
                    if 'model' in checkpoint:
                        self.model = checkpoint['model'].float()
                    else:
                        self.model = checkpoint
                    
                    self.model.to(self.device)
                    self.model.eval()
                    self.log_message.emit("模型載入成功（直接載入）")
                    return True
                    
                except Exception as e2:
                    self.error_occurred.emit(f"模型載入失敗: {str(e2)}")
                    return False
                    
        except Exception as e:
            self.error_occurred.emit(f"模型載入失敗: {str(e)}")
            return False
    
    def _load_data_config(self):
        """載入資料配置"""
        try:
            data_config_path = self.params['data']
            
            with open(data_config_path, 'r', encoding='utf-8') as f:
                self.data_config = yaml.safe_load(f)
            
            # 驗證資料配置
            task = self.params.get('task', 'val')
            if task == 'val':
                data_path = self.data_config.get('val')
            elif task == 'test':
                data_path = self.data_config.get('test')
            else:
                data_path = self.data_config.get('val')  # 預設使用val
            
            if not data_path or not os.path.exists(data_path):
                self.error_occurred.emit(f"測試資料路徑不存在: {data_path}")
                return False
            
            self.test_data_path = data_path
            self.class_names = self.data_config.get('names', [])
            self.num_classes = self.data_config.get('nc', len(self.class_names))
            
            self.log_message.emit(f"資料配置載入成功，類別數: {self.num_classes}")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"資料配置載入失敗: {str(e)}")
            return False
    
    def _execute_testing(self):
        """執行測試過程"""
        try:
            self.log_message.emit("開始執行測試...")
            
            # 創建輸出目錄
            project = self.params.get('project', 'runs/val')
            name = self.params.get('name', 'exp')
            output_dir = os.path.join(project, name)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存測試配置
            config_file = os.path.join(output_dir, 'testing_config.json')
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.params, f, indent=2, ensure_ascii=False)
            
            # 執行測試
            task = self.params.get('task', 'val')
            if task == '速度測試':
                self._speed_test(output_dir)
            else:
                self._accuracy_test(output_dir)
                
        except Exception as e:
            self.error_occurred.emit(f"測試執行失敗: {str(e)}")
    
    def _accuracy_test(self, output_dir):
        """精度測試"""
        try:
            self.log_message.emit("執行精度測試...")
            
            # 模擬測試過程（實際實現需要根據具體的YOLOv5版本調整）
            self.log_message.emit("注意: 這是測試過程的模擬實現")
            self.log_message.emit("實際部署時需要整合真實的YOLOv5測試代碼")
            
            # 模擬測試統計
            total_images = 100  # 假設有100張測試圖片
            processed_images = 0
            
            # 模擬分類別結果
            class_results = {}
            for i, class_name in enumerate(self.class_names):
                # 模擬每個類別的測試結果
                class_results[class_name] = {
                    'images': np.random.randint(10, 20),
                    'instances': np.random.randint(15, 30),
                    'precision': np.random.uniform(0.7, 0.95),
                    'recall': np.random.uniform(0.6, 0.9),
                    'mAP_0.5': np.random.uniform(0.6, 0.9),
                    'mAP_0.5_0.95': np.random.uniform(0.4, 0.7),
                    'f1': 0.0  # 會在下面計算
                }
                # 計算F1分數
                p = class_results[class_name]['precision']
                r = class_results[class_name]['recall']
                class_results[class_name]['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            # 計算總體統計
            all_precision = [cls['precision'] for cls in class_results.values()]
            all_recall = [cls['recall'] for cls in class_results.values()]
            all_map50 = [cls['mAP_0.5'] for cls in class_results.values()]
            all_map50_95 = [cls['mAP_0.5_0.95'] for cls in class_results.values()]
            all_f1 = [cls['f1'] for cls in class_results.values()]
            
            overall_results = {
                'metrics/precision': np.mean(all_precision),
                'metrics/recall': np.mean(all_recall),
                'metrics/mAP_0.5': np.mean(all_map50),
                'metrics/mAP_0.5:0.95': np.mean(all_map50_95),
                'metrics/f1': np.mean(all_f1)
            }
            
            # 模擬測試進度
            for i in range(total_images):
                if not self.testing_active:
                    self.log_message.emit("測試被用戶中止")
                    return
                
                processed_images += 1
                progress = int((processed_images / total_images) * 100)
                self.testing_progress.emit(progress)
                
                # 模擬處理時間
                self.msleep(10)
            
            # 生成混淆矩陣
            confusion_matrix = self._generate_confusion_matrix()
            
            # 模擬速度統計
            speed_stats = {
                'preprocess': np.random.uniform(1.0, 3.0),
                'inference': np.random.uniform(10.0, 50.0),
                'nms': np.random.uniform(0.5, 2.0),
                'total': 0.0
            }
            speed_stats['total'] = sum([speed_stats['preprocess'], speed_stats['inference'], speed_stats['nms']])
            
            # 組合最終結果
            final_results = {
                **overall_results,
                'class_results': class_results,
                'confusion_matrix': confusion_matrix,
                'speed': speed_stats,
                'total_images': total_images,
                'output_dir': output_dir
            }
            
            # 保存結果
            self._save_results(output_dir, final_results)
            
            # 發送結果
            self.testing_finished.emit(final_results)
            self.log_message.emit("精度測試完成")
            
        except Exception as e:
            self.error_occurred.emit(f"精度測試失敗: {str(e)}")
    
    def _speed_test(self, output_dir):
        """速度測試"""
        try:
            self.log_message.emit("執行速度測試...")
            
            # 模擬速度測試
            test_iterations = 100
            batch_size = self.params.get('batch_size', 1)
            img_size = self.params.get('imgsz', 640)
            
            # 模擬測試圖片
            dummy_img = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
            
            # 預熱
            self.log_message.emit("模型預熱中...")
            for _ in range(10):
                if hasattr(self.model, '__call__'):
                    _ = self.model(dummy_img)
            
            # 測試推理速度
            self.log_message.emit(f"開始速度測試，迭代次數: {test_iterations}")
            
            times = []
            for i in range(test_iterations):
                if not self.testing_active:
                    self.log_message.emit("速度測試被用戶中止")
                    return
                
                start_time = time.time()
                
                # 執行推理
                if hasattr(self.model, '__call__'):
                    with torch.no_grad():
                        _ = self.model(dummy_img)
                        
                        # 如果使用GPU，需要同步
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # 轉換為毫秒
                
                # 更新進度
                progress = int(((i + 1) / test_iterations) * 100)
                self.testing_progress.emit(progress)
            
            # 計算統計
            times = np.array(times)
            speed_results = {
                'batch_size': batch_size,
                'image_size': img_size,
                'iterations': test_iterations,
                'device': str(self.device),
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'fps': float(batch_size * 1000 / np.mean(times)),
                'speed': {
                    'preprocess': 1.0,  # 預處理時間（模擬）
                    'inference': float(np.mean(times)),
                    'nms': 1.0,  # NMS時間（模擬）
                    'total': float(np.mean(times)) + 2.0
                }
            }
            
            # 保存結果
            result_file = os.path.join(output_dir, 'speed_test_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(speed_results, f, indent=2, ensure_ascii=False)
            
            self.log_message.emit(f"平均推理時間: {speed_results['mean_time']:.2f}ms")
            self.log_message.emit(f"推理FPS: {speed_results['fps']:.1f}")
            
            # 發送結果
            self.testing_finished.emit(speed_results)
            self.log_message.emit("速度測試完成")
            
        except Exception as e:
            self.error_occurred.emit(f"速度測試失敗: {str(e)}")
    
    def _generate_confusion_matrix(self):
        """生成混淆矩陣"""
        try:
            num_classes = len(self.class_names)
            
            # 生成模擬混淆矩陣
            confusion_matrix = np.random.randint(0, 20, size=(num_classes, num_classes))
            
            # 確保對角線元素較大（正確分類）
            for i in range(num_classes):
                confusion_matrix[i, i] += np.random.randint(50, 100)
            
            # 轉換為字串格式以便顯示
            matrix_str = "混淆矩陣 (實際 vs 預測):\n"
            matrix_str += "類別: " + " ".join([f"{i:>8}" for i in range(num_classes)]) + "\n"
            
            for i, row in enumerate(confusion_matrix):
                class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                matrix_str += f"{class_name[:8]:>8}: " + " ".join([f"{val:>8}" for val in row]) + "\n"
            
            return matrix_str
            
        except Exception as e:
            self.log_message.emit(f"生成混淆矩陣失敗: {str(e)}")
            return "混淆矩陣生成失敗"
    
    def _save_results(self, output_dir, results):
        """保存測試結果"""
        try:
            # 保存JSON格式結果
            results_file = os.path.join(output_dir, 'test_results.json')
            
            # 處理不可序列化的數據
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (np.ndarray, np.generic)):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (np.ndarray, np.generic)):
                            serializable_results[key][k] = v.tolist()
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # 保存文字格式報告
            report_file = os.path.join(output_dir, 'test_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("YOLO-PCB 測試報告\n")
                f.write("=" * 50 + "\n\n")
                
                # 總體結果
                f.write("總體結果:\n")
                f.write("-" * 20 + "\n")
                if 'metrics/mAP_0.5' in results:
                    f.write(f"mAP@0.5: {results['metrics/mAP_0.5']:.4f}\n")
                if 'metrics/mAP_0.5:0.95' in results:
                    f.write(f"mAP@0.5:0.95: {results['metrics/mAP_0.5:0.95']:.4f}\n")
                if 'metrics/precision' in results:
                    f.write(f"Precision: {results['metrics/precision']:.4f}\n")
                if 'metrics/recall' in results:
                    f.write(f"Recall: {results['metrics/recall']:.4f}\n")
                if 'metrics/f1' in results:
                    f.write(f"F1-Score: {results['metrics/f1']:.4f}\n")
                
                f.write("\n")
                
                # 分類別結果
                if 'class_results' in results:
                    f.write("分類別結果:\n")
                    f.write("-" * 20 + "\n")
                    for class_name, metrics in results['class_results'].items():
                        f.write(f"\n{class_name}:\n")
                        for metric, value in metrics.items():
                            f.write(f"  {metric}: {value:.4f}\n")
                
                # 混淆矩陣
                if 'confusion_matrix' in results:
                    f.write(f"\n{results['confusion_matrix']}\n")
                
                # 速度統計
                if 'speed' in results:
                    f.write("\n速度統計:\n")
                    f.write("-" * 20 + "\n")
                    for metric, value in results['speed'].items():
                        f.write(f"{metric}: {value:.2f}ms\n")
            
            self.log_message.emit(f"測試結果已保存至: {output_dir}")
            
        except Exception as e:
            self.log_message.emit(f"保存測試結果失敗: {str(e)}")
    
    def stop_testing(self):
        """停止測試"""
        self.testing_active = False
        self.log_message.emit("正在停止測試...")


class Tester(QObject):
    """測試器主類別"""
    
    # 信號定義
    testing_started = pyqtSignal()
    testing_finished = pyqtSignal(dict)
    testing_progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.worker = None
        
    def start_testing(self, params):
        """開始測試"""
        try:
            # 創建工作執行緒
            self.worker = TestingWorker()
            self.worker.set_parameters(params)
            
            # 連接信號
            self.worker.testing_progress.connect(self.testing_progress.emit)
            self.worker.testing_stats.connect(self._on_testing_stats)
            self.worker.log_message.connect(self.log_message.emit)
            self.worker.testing_finished.connect(self.testing_finished.emit)
            self.worker.error_occurred.connect(self.log_message.emit)
            
            # 開始測試
            self.worker.start()
            self.testing_started.emit()
            
            return True
            
        except Exception as e:
            self.log_message.emit(f"啟動測試失敗: {str(e)}")
            return False
    
    def stop_testing(self):
        """停止測試"""
        if self.worker:
            self.worker.stop_testing()
            if self.worker.isRunning():
                self.worker.quit()
                self.worker.wait(5000)  # 最多等待5秒
            self.worker = None
    
    def _on_testing_stats(self, stats):
        """處理測試統計事件"""
        # 這裡可以添加額外的處理邏輯
        pass
