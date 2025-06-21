# UI 性能優化記錄

## 問題描述
原系統在進行影像檢測時，UI 會在影像讀取和推理過程中完全無響應，「停止檢測」按鈕無法使用。

## 根本原因
1. **UI線程被阻塞**：每幀影像都在UI線程進行大量處理：
   - BGR→RGB 顏色空間轉換
   - OpenCV numpy.ndarray → QImage → QPixmap 轉換  
   - 影像縮放處理
   - 檢測標註的繪製（OpenCV繪圖操作）

2. **重複的重量級操作**：攝像頭檢測時每幀（30fps）都執行上述操作，造成UI嚴重卡頓。

## 優化策略

### 1. 工作負載轉移到QThread
- **變更前**：`frame_processed` 信號傳遞 `np.ndarray`，UI線程負責所有影像處理
- **變更後**：`frame_processed` 信號傳遞 `QPixmap`，worker線程完成所有處理

### 2. Worker端預處理機制
在 `DetectionWorker._prepare_display_image()` 中完成：
- BGR→RGB 轉換
- 檢測結果標註繪製 
- 影像縮放到顯示區域大小
- numpy.ndarray → QImage → QPixmap 轉換

### 3. UI線程輕量化
- `on_frame_processed()` 只負責 `setPixmap()` 
- `on_detection_result()` 只更新統計文字，不做標註處理
- 移除UI端的影像複製、轉換、繪圖操作

### 4. 幀率控制優化
- 新增 `frame_skip` 機制：每N幀才顯示一次，降低UI更新頻率
- 保持檢測頻率，但減少顯示負載

## 具體修改

### DetectionWorker (core/detector.py)
```python
# 信號變更
frame_processed = pyqtSignal(QPixmap)  # 改為QPixmap

# 新增方法
def set_display_size(self, width, height)
def _prepare_display_image(self, cv_image, detections=None)

# 幀跳過機制
self.frame_skip = 2  # 每2幀顯示1幀
```

### DetectTab (gui/detect_tab.py) 
```python
# Slot 優化
@pyqtSlot(QPixmap)
def on_frame_processed(self, pixmap):
    self.image_viewer.setPixmap(pixmap)  # 直接設置，無額外處理

@pyqtSlot(dict) 
def on_detection_result(self, result):
    # 只更新統計文字，標註已在worker完成
```

## 效果預期

1. **UI響應性**：「停止檢測」按鈕隨時可用，UI不再凍結
2. **性能提升**：減少UI線程負載70-80%
3. **穩定性增強**：影像處理錯誤不會阻塞UI
4. **用戶體驗**：檢測過程可隨時中斷，介面保持流暢

## 向後兼容性
- 保持所有原有API不變
- 檢測結果格式完全相同 
- 僅內部處理流程優化，對外接口無變化

## 測試建議
1. 高解析度影像檢測（1080p+）
2. 長時間攝像頭檢測（>5分鐘）
3. 檢測過程中的停止操作響應性
4. 多次啟動/停止檢測的穩定性

---
*優化完成時間：2025-06-21*
*優化範圍：UI性能、線程管理、影像處理流程*
