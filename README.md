# YOLO-PCB GUI 使用指南

## 快速開始

### 1. 環境要求
- Python 3.7 或更高版本
- Windows 10/11 或 Linux/macOS
- 至少 4GB RAM（推薦 8GB）
- 支援 CUDA 的 GPU（可選，用於加速）

### 2. 安裝依賴
```bash
cd YOLO-PCB-GUI
pip install -r requirements_gui.txt
```

### 3. 啟動應用程式
有三種方式啟動：

**方式一：使用啟動腳本（推薦）**
```bash
python run_gui.py
```

**方式二：直接運行主程式**
```bash
python main.py
```

**方式三：測試並啟動**
```bash
python test_and_run.py
```

### 3.1 使用 venv 建立虛擬環境並啟動

建議使用 Python venv 建立獨立環境，避免依賴衝突：

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安裝依賴
pip install -r requirements_gui.txt

# 啟動應用程式
python run_gui.py
```
### 4. 模組測試
如果遇到問題，可以先運行模組測試：
```bash
python test_modules.py
```

## 主要功能

### 🔍 PCB 檢測
1. **輸入來源**：支援圖片、影片、資料夾和攝像頭
2. **模型配置**：載入 YOLO 權重檔案
3. **參數調整**：信心度、IoU 閾值、最大檢測數量等
4. **結果顯示**：即時標註、統計資訊、結果保存

**使用步驟：**
1. 選擇檢測頁籤
2. 設置輸入來源（瀏覽選擇檔案或資料夾）
3. 載入模型權重檔案（.pt 格式）
4. 調整檢測參數
5. 點擊「開始檢測」按鈕
6. 查看檢測結果和統計資訊

### 🎓 模型訓練
1. **資料配置**：設置訓練、驗證和測試資料集
2. **模型配置**：選擇 YOLO 模型架構
3. **超參數設定**：學習率、批次大小、訓練輪數等
4. **訓練監控**：即時顯示訓練進度和損失

**使用步驟：**
1. 選擇訓練頁籤
2. 在「資料配置」分頁設置資料集路徑和類別
3. 在「模型配置」分頁選擇模型架構
4. 在「超參數配置」分頁調整訓練參數
5. 在「訓練控制」分頁開始訓練
6. 監控訓練進度和結果

### 📊 模型測試
1. **測試配置**：設置測試資料集和模型
2. **評估指標**：計算 mAP、精確度、召回率等
3. **結果分析**：詳細的性能報告和可視化
4. **結果匯出**：保存測試報告

**使用步驟：**
1. 選擇測試頁籤
2. 在「測試配置」分頁設置模型和資料
3. 在「測試控制」分頁開始測試
4. 在「結果顯示」分頁查看詳細結果
5. 匯出測試報告

## 目錄結構

```
YOLO-PCB-GUI/
├── main.py                 # 主程式入口
├── run_gui.py              # 啟動腳本
├── test_modules.py         # 模組測試腳本
├── requirements_gui.txt    # 依賴列表
├── config.json            # 配置檔案
├── gui/                   # GUI 模組
│   ├── main_window.py     # 主視窗
│   ├── detect_tab.py      # 檢測頁籤
│   ├── train_tab.py       # 訓練頁籤
│   ├── test_tab.py        # 測試頁籤
│   └── widgets/           # 自定義控件
│       └── image_viewer.py # 圖像顯示控件
├── core/                  # 核心模組
│   ├── detector.py        # 檢測核心
│   ├── trainer.py         # 訓練核心
│   └── tester.py          # 測試核心
├── utils/                 # 工具模組
│   └── config_manager.py  # 配置管理
├── data/                  # 資料目錄
├── weights/               # 模型權重
├── runs/                  # 執行結果
│   ├── detect/           # 檢測結果
│   ├── train/            # 訓練結果
│   └── test/             # 測試結果
└── logs/                  # 日誌檔案
```

## 配置說明

### 配置檔案位置
- 主配置：`config.json`
- 資料配置：`data/*.yaml`
- 模型配置：`models/*.yaml`

### 主要配置項目
```json
{
  "app": {
    "name": "YOLO-PCB GUI",
    "version": "1.0.0",
    "window_geometry": [100, 100, 1200, 800],
    "theme": "default"
  },
  "detection": {
    "default_weights": "./weights/best.pt",
    "default_conf": 0.25,
    "default_iou": 0.45,
    "default_device": "auto"
  },
  "training": {
    "default_epochs": 100,
    "default_batch_size": 16,
    "default_lr": 0.01
  }
}
```

## 故障排除

### 常見問題

**1. 模組導入失敗**
```
ImportError: cannot import name 'xxx' from 'xxx'
```
**解決方案：**
- 檢查 Python 版本：`python --version`
- 重新安裝依賴：`pip install -r requirements_gui.txt`
- 運行模組測試：`python test_modules.py`

**2. PyQt5 相關錯誤**
```
ImportError: No module named 'PyQt5'
```
**解決方案：**
```bash
pip install PyQt5 PyQt5-tools
```

**3. CUDA/GPU 相關問題**
```
CUDA out of memory / No CUDA devices
```
**解決方案：**
- 設置 device 為 "cpu"
- 減少批次大小
- 檢查 GPU 記憶體使用情況

**4. 權重檔案載入失敗**
```
FileNotFoundError: No such file or directory
```
**解決方案：**
- 確認權重檔案路徑正確
- 下載對應的 YOLO 權重檔案
- 檢查檔案格式（需要 .pt 格式）

### 日誌檔案
- 應用程式日誌：`yolo_pcb_gui.log`
- 測試日誌：`gui_test.log`

查看日誌可以獲得更詳細的錯誤資訊。

## 開發資訊

### 版本資訊
- 當前版本：1.0.0
- 開發狀態：Beta
- 最後更新：2025-06-14

### 技術棧
- GUI 框架：PyQt5
- 深度學習：PyTorch
- 圖像處理：OpenCV
- 資料處理：NumPy, Pandas
- 配置管理：YAML, JSON

### 支援的 YOLO 版本
- YOLOv5（主要支援）
- YOLOv8（規劃中）
- 自定義 YOLO 模型

## 聯繫資訊

如果遇到問題或需要幫助，請：
1. 查看日誌檔案
2. 運行模組測試腳本
3. 檢查環境配置
4. 參考 GitHub Issues

---

**祝使用愉快！**
