# LSTM 降雨逕流預測模型

本專案旨在使用長短期記憶（LSTM）類神經網路，根據歷史降雨和流量資料，預測集水區的未來逕流量。專案現在提供一個完整的解決方案，從原始資料處理、模型訓練，到一個互動式的網頁應用程式，讓使用者可以輕鬆上傳資料並獲得預測結果。

## 功能特色

- **資料前處理**：提供腳本自動合併、對齊不同來源的雨量和流量文字檔。
- **自動資料補遺**：在資料載入時，自動偵測缺漏值（如-1）並使用線性內插法進行填補。
- **模型訓練**：使用歷史資料訓練一個 PyTorch LSTM 模型，並將訓練好的模型參數與資料正規化設定儲存起來。
- **暖啟動訓練**：在訓練前自動檢查是否存在參數相符的舊模型，若存在則載入權重進行接續訓練，節省時間。
- **互動式網頁介面**：透過 Streamlit 建立了一個使用者友善的網頁應用，功能包括：
  - 檔案上傳
  - 日期範圍篩選
  - 點擊按鈕執行預測
  - 即時顯示預測結果、評估指標與視覺化圖表
- **參數化執行**：透過命令列參數，可以靈活地調整模型超參數（如 `lookback`, `epochs` 等）和執行模式（訓練或驗證）。
- **視覺化分析**：
  - 產生訓練過程中的損失（Loss）曲線圖。
  - 繪製觀測值與預測值的時間序列對比圖及 XY 散佈圖。
  - 繪製降雨與逕流的歷線圖。

## 專案結構

```plain
lstm-flow1.1/
├── raw_data/                     # 存放原始資料檔案
│   ├── 09_RstAvgRain.txt         # 原始雨量資料
│   └── RFETS_flow.txt            # 原始流量資料
├── data/
│   ├── rainfall_flow_aligned.csv # 經過前處理後合併的資料
│   └── 2023pred1.csv             # (範例) 用於預測的資料
├── Result/                       # 存放所有輸出的圖表與 CSV 檔案
├── src/
│   ├── FlowLSTM.py               # LSTM 模型架構定義
│   ├── ModelEV.py                # 模型評估指標計算函式
│   ├── plot_chart.py             # 繪圖相關函式
│   └── utils.py                  # 通用輔助函式
├── flow_prediction_model.pth     # 訓練後儲存的模型檔案
├── app.py                        # Streamlit 互動式預測應用程式
├── preprocess_data.py            # 資料前處理腳本
├── pred1.py                      # 執行預測的腳本
├── train1.py                     # 執行訓練與驗證的腳本
├── requirements.txt              # 專案必要套件
└── README.md                     # 專案說明文件
```

## 安裝

1. 確認您已安裝 Python 3.8 或更高版本。
2. 複製此專案到您的本地端。
3. 安裝必要的 Python 套件：

```bash
    pip install -r requirements.txt
```

## 使用方法

本專案主要透過 `train1.py` 和 `pred1.py` 兩個腳本執行。

### 1. 訓練與驗證模型 (`train1.py`)

此腳本用於訓練新的 LSTM 模型或驗證現有模型。

**基本指令格式：**

```bash
python train1.py --action [train|validate] [其他參數...]
```

**常用指令範例：**

- **從頭開始訓練模型並驗證** (預設行為):

```bash
  python train1.py --action train
```

- **使用不同的超參數進行訓練** (例如，增加回看時長和訓練週期):

```bash
  python train1.py --action train --lookback 12 --epochs 100 --lr 0.001
```

  訓練完成後，模型 (`flow_prediction_model.pth`)、訓練損失歷史 (`training_loss_history.csv`) 和相關圖表將儲存於 `Result` 目錄中。

- **僅使用現有模型進行驗證**:

```bash
  python train1.py --action validate
```

  此指令會載入 `flow_prediction_model.pth` 並在測試集上執行評估。

- **查看所有可調整參數**:

```bash
  python train1.py --help
```

### 2. 執行疊代預測(或稱單步預測)

- 原始碼: pred1.py
- 此腳本用於載入已訓練好的模型，對一份新的資料集進行未來 6 小時的迭代預測。

**執行預測：**

```bash
python pred1.py
```

腳本會自動載入 `flow_prediction_model.pth` 和 `data/2023pred1.csv` 檔案，並將預測結果的對照表和評估圖表儲存於 `Result` 目錄中。

您可以修改 `pred1.py` 中的 `DATA_PATH` 和 `MODEL_PATH` 變數來指定不同的輸入檔案。
