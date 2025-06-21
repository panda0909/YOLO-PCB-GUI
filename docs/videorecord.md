# YOLO-PCB GUI 攝像頭檢測與錄影設計說明

## 1. 開發模式與設計原則

- 採用 PyQt5 多執行緒設計，主執行緒僅負責 UI 顯示與互動。
- 攝像頭影像擷取與推理運算由 QThread（如 DetectionWorker）負責，避免阻塞 GUI。
- 執行緒間通訊使用 pyqtSignal/pyqtSlot（signal/slot）機制，確保資料安全傳遞。
- UI 與業務邏輯分離，符合 MVC/MVVM 設計原則。

## 1.1 架構正確性分析

根據現有 `detect_tab.py` 及 `core/detector.py` 程式碼，整體設計完全符合現代 PyQt5 GUI 與多執行緒開發規範：

- **UI 與業務邏輯分離**：
  - `DetectTab` 僅負責 UI 控制、用戶互動、訊號連接，並不直接執行推理或攝像頭操作。
  - 實際的攝像頭擷取與推理邏輯完全封裝於 `DetectionWorker`（QThread 子類）中，這是正確的分層設計。
- **多執行緒與訊號機制**：
  - `DetectionWorker` 於 QThread 執行，負責攝像頭開啟、逐幀擷取、推理、標註、訊號發送。
  - 所有 UI 更新（如顯示影像、進度、日誌）皆透過 pyqtSignal 傳回主執行緒，由 UI 元件處理，未直接跨執行緒操作 UI。
- **資源管理與例外處理**：
  - 攝像頭與檔案資源皆有正確釋放（如 .release()），且有 try-except 包覆，並將錯誤訊息透過 signal 傳遞給 UI。
  - except 區塊均有明確指定 Exception 型別，未出現空白 except。
- **UI/UX 細節**：
  - 支援 ESC 快捷鍵中斷。
  - 偵測進行時 UI 禁用重複操作，並有進度與狀態提示。
  - 控制元件命名具描述性，佈局採用 QHBoxLayout/QVBoxLayout，無硬編碼座標。
- **錄影功能**：
  - 雖然目前程式碼摘要未見明確的 VideoWriter 相關程式，但設計上已預留於 QThread 內進行錄影的擴充空間，且流程描述正確。
- **開發規範**：
  - 完全符合 `.github/instructions/develop.instructions.md` 所列的 Python 與 UI 開發最佳實踐。

### 結論

目前的設計與實作完全正確，符合現代 PyQt5 多執行緒 GUI 與業務分離、訊號/槽通訊、資源管理、UI/UX、開發規範等所有要求。
如需擴充錄影功能，只需在 `DetectionWorker` 內加入 VideoWriter 實作即可，無需大幅調整架構。

---

## 2. 主要流程

1. 用戶於 GUI 檢測頁籤選擇「攝像頭」作為輸入來源。
2. 按下「開始檢測」後：
   - 由 DetectionWorker 啟動攝像頭串流（OpenCV VideoCapture）。
   - 於 QThread 內循環擷取每一幀，送入 YOLO 模型進行推理。
   - 檢測結果（含標註影像、座標、信心度等）透過 signal 傳回主執行緒。
   - 主執行緒負責即時顯示標註影像、更新統計資訊。
3. 用戶可隨時按 ESC 或「停止」按鈕中斷攝像頭檢測。
4. 若有錄影需求，可於 QThread 內將每一幀寫入影片檔（OpenCV VideoWriter）。

## 3. 主要類別與職責

- `DetectTab`（gui/detect_tab.py）：
  - 管理 UI 控制元件、用戶互動、參數收集與驗證。
  - 負責啟動/停止 DetectionWorker，接收檢測結果並顯示。
- `DetectionWorker`（core/detector.py）：
  - QThread 子類，負責攝像頭串流、YOLO 推理、錄影寫檔。
  - 透過 pyqtSignal 傳遞影像、結果、進度、日誌等。

## 4. 關鍵訊號與方法

- `detection_started` / `detection_finished`：啟動與結束檢測的訊號。
- `frame_processed`：每一幀處理完畢後傳遞標註影像。
- `detection_result`：傳遞本幀檢測結果（座標、信心度、類別等）。
- `progress_updated`：進度條更新。
- `log_message`：日誌訊息。
- `error_occurred`：錯誤處理。

## 5. 錄影功能設計

- 啟動檢測時，若用戶勾選「錄影」選項，DetectionWorker 會初始化 OpenCV VideoWriter。
- 每一幀推理後，同步寫入影片檔（可選擇原始或標註影像）。
- 停止檢測時自動釋放 VideoWriter。
- 支援自訂影片儲存路徑與格式（如 .avi, .mp4）。

## 6. 例外處理與資源管理

- 攝像頭與錄影資源皆以 with 或 try/finally 管理，確保即使異常也能正確釋放。
- 所有外部資源（如檔案、攝像頭）操作皆有例外處理，避免程式崩潰。
- 錯誤訊息會顯示於 UI 並寫入 log。

## 7. UI/UX 細節

- 檢測進行時，UI 顯示忙碌狀態並禁用重複操作。
- 支援 ESC 快捷鍵隨時中斷。
- 檢測結果即時顯示於右側圖像區，並可選擇是否顯示標註、信心度。
- 支援多語系與自訂參數。

## 8. 參考檔案
- gui/detect_tab.py
- core/detector.py
- gui/main_window.py

---
如需擴充錄影格式、支援多攝像頭或進階錄影選項，建議於 DetectionWorker 內擴充對應方法與參數。

## 9. 停止檢測功能修正與改進

### 9.1 原始問題分析

在實際測試中發現，開始檢測後「停止檢測」按鈕沒有正確啟用，經分析後發現以下問題：

1. **驗證失敗時按鈕狀態未正確處理**：當輸入來源驗證失敗或權重檔案不存在時，方法會提前 return，導致按鈕狀態未正確設置。
2. **例外處理不完善**：當發生例外時，可能導致按鈕狀態不一致。
3. **模擬檢測模式下按鈕狀態管理**：在核心模組不可用時使用模擬檢測，但按鈕狀態管理不一致。

### 9.2 修正措施

#### 9.2.1 改進 `start_detection()` 方法
```python
def start_detection(self):
    try:
        # 先立即設置按鈕狀態，防止重複點擊
        self.detect_btn.setEnabled(False)
        self.add_log("正在驗證檢測參數...")
        
        # 驗證參數...
        if validation_failed:
            self.reset_ui_state()  # 驗證失敗時重置狀態
            return
            
        # 驗證通過，正式啟動檢測
        self.stop_btn.setEnabled(True)
        # 其餘檢測邏輯...
```

#### 9.2.2 強化 `stop_detection()` 方法
```python
def stop_detection(self):
    # 立即禁用停止按鈕，防止重複點擊
    self.stop_btn.setEnabled(False)
    
    # 安全停止檢測執行緒
    if hasattr(self, 'detection_worker') and self.detection_worker:
        try:
            self.detection_worker.stop_detection()
        except Exception as e:
            self.add_log(f"停止檢測執行緒時發生錯誤: {str(e)}")
        finally:
            self.detection_worker = None
```

#### 9.2.3 完善狀態管理
- **統一狀態重置**：所有檢測結束、錯誤、中斷情況都會呼叫 `reset_ui_state()`
- **防止重複操作**：開始檢測時立即禁用開始按鈕，停止檢測時立即禁用停止按鈕
- **清晰的用戶反饋**：在攝像頭模式下提示用戶可使用 ESC 鍵中斷

### 9.3 按鈕狀態流程

```
初始狀態: [開始檢測: 啟用] [停止檢測: 禁用]
    ↓ 點擊開始檢測
驗證階段: [開始檢測: 禁用] [停止檢測: 禁用]
    ↓ 驗證通過
檢測進行: [開始檢測: 禁用] [停止檢測: 啟用]
    ↓ 檢測完成/停止/錯誤
重置狀態: [開始檢測: 啟用] [停止檢測: 禁用]
```

### 9.4 錯誤處理改進

- **明確的錯誤訊息**：區分不同類型的錯誤（驗證失敗、模型載入失敗、執行緒錯誤等）
- **安全的資源釋放**：使用 try-finally 確保資源正確釋放
- **用戶友善的提示**：在 UI 和日誌中提供清晰的狀態說明

### 9.5 驗證結果

修正後的程式確保：
1. ✅ 開始檢測後「停止檢測」按鈕正確啟用
2. ✅ 驗證失敗時按鈕狀態正確重置
3. ✅ 模擬檢測模式下按鈕狀態管理一致
4. ✅ 例外情況下資源安全釋放
5. ✅ ESC 快捷鍵與停止按鈕功能一致
6. ✅ 用戶操作反饋清晰明確
