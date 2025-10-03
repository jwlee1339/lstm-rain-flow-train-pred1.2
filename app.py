# app.py
import streamlit as st
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 匯入您專案中既有的函式
# 確保這些檔案與 app.py 在同一個 Python 環境中可以被找到
from FlowLSTM import FlowLSTM
from src.plot_chart import plot_prediction_vs_observation
from src.ModelEV import ModelTest
from src.utils import create_dataset, inverse_transform_flow

# --- 將 pred1.py 的核心函式複製或匯入到這裡 ---

# 使用 Streamlit 快取來避免重複載入模型，提升效能
@st.cache_resource
def load_model(model_path):
    """
    從指定路徑加載已訓練好的 PyTorch 模型、MinMaxScaler 和模型參數。
    """
    st.info(f"正在從 {model_path} 載入模型...")
    # 注意weights_only=False，因為我們需要載入 scaler 的 numpy 陣列
    checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))

    model_params = checkpoint.get('model_params', {})
    model = FlowLSTM(
        input_size=model_params.get('input_size', 2),
        hidden_size=model_params.get('hidden_size', 64),
        num_layers=model_params.get('num_layers', 2),
        output_size=model_params.get('output_size', 1)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 重建scaler
    scaler = MinMaxScaler()
    scaler.min_ = checkpoint['scaler_min']
    scaler.scale_ = checkpoint['scaler_scale']
    
    st.success("模型載入成功！")
    return model, scaler, model_params.get('lookback'), model_params.get('forecast_horizon')

def predict_future_flow(model, scaler, last_observed_data):
    """
    對單一 lookback 窗口的數據進行一次流量預測。
    """
    scaled_input = scaler.transform(last_observed_data)
    device = next(model.parameters()).device
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        scaled_predictions = model(input_tensor).cpu().numpy()[0]
    prediction = inverse_transform_flow(scaled_predictions, scaler)
    return prediction

def load_and_prepare_data(uploaded_file, shift_hours):
    """
    從上傳的檔案加載並預處理數據。包含線性內插功能來填補缺漏值。
    """
    if uploaded_file is None:
        return None
        
    df = pd.read_csv(uploaded_file)
    # 確保 'obstime' 欄位存在
    if 'obstime' not in df.columns:
        st.error("上傳的 CSV 檔案缺少 'obstime' 欄位，無法繼續處理。")
        return None

    df['date_column'] = pd.to_datetime(df['obstime'])
    df = df.set_index('date_column')
    
    # --- 新增：資料補遺功能 ---
    # 1. 將代表缺漏的負值（如 -1）替換為 NaN，以便進行內插
    st.write("---")
    st.write("🔍 **資料補遺檢查:**")
    for col in ['rainfall', 'flow']:
        if col in df.columns:
            missing_count = (df[col] < 0).sum()
            if missing_count > 0:
                df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)
                st.info(f"在 '{col}' 欄位中發現 {missing_count} 筆缺漏值，將進行線性內插處理。")
    
    # 2. 執行線性內插
    df.interpolate(method='linear', inplace=True)

    # 製作預測雨量特徵
    df[f'fcstrain_{shift_hours}h'] = df['rainfall'].shift(periods=-shift_hours, fill_value=0)
    return df

def perform_iterative_forecast(df, model, scaler, X, lookback, forecast_horizon, lead_hours):
    """
    執行迭代式（滾動）預測。
    """
    all_predictions = []
    initial_times = []
    
    progress_bar = st.progress(0, text="正在執行迭代預測...")

    for index, initial_input in enumerate(X):
        if lookback + index + lead_hours > len(df):
            st.warning(f"數據長度不足，於索引 {index} 停止預測。")
            break

        current_input = initial_input.copy()
        step_predictions = []

        for i in range(0, lead_hours, forecast_horizon):
            pred_values = predict_future_flow(model, scaler, current_input)
            num_preds = len(pred_values)
            step_predictions.extend(pred_values)
            current_input = np.roll(current_input, -num_preds, axis=0)
            for j in range(num_preds):
                next_rainfall_index = lookback + index + i + j
                next_rainfall = df['fcstrain_1h'].iloc[next_rainfall_index]
                current_input[-(num_preds-j)] = [pred_values[j], next_rainfall]

        all_predictions.append(step_predictions[:lead_hours])
        initial_time = df.index[lookback + index]
        initial_times.append(initial_time)
        
        # 更新進度條
        progress_bar.progress((index + 1) / len(X), text=f"正在預測起始時間: {initial_time.strftime('%Y-%m-%d %H:%M')}")

    progress_bar.empty() # 完成後移除進度條
    
    pred_columns = [f'h{i+1}' for i in range(min(lead_hours, len(all_predictions[0])))]
    df_pred = pd.DataFrame(all_predictions, columns=pred_columns, index=initial_times)
    return df_pred

# --- Streamlit App 主體 ---

st.set_page_config(page_title="降雨逕流預測系統", layout="wide")

st.title("🌊 LSTM 降雨逕流預測應用")
st.markdown("""
這是一個使用 LSTM 模型進行降雨逕流預測的展示應用。
請在左側側邊欄上傳您的 CSV 資料檔案，然後點擊按鈕開始預測。
""")

# --- 側邊欄設定 ---
with st.sidebar:
    st.header("⚙️ 參數設定")
    
    # 檔案上傳
    uploaded_file = st.file_uploader(
        "請上傳 CSV 格式的預測資料",
        type="csv",
        help="CSV 檔案必須包含 'obstime', 'rainfall', 'flow' 三個欄位。"
    )
    
    # 新增：範例展示按鈕
    if st.button("載入範例資料"):
        st.session_state.run_demo = True
        st.session_state.uploaded_file = None # 清除已上傳的檔案狀態
        # 使用 st.rerun() 來立即刷新應用狀態
        st.rerun()

    # 如果使用者上傳新檔案，則取消範例模式
    if uploaded_file is not None:
        st.session_state.run_demo = False

    # 預測參數
    LEAD_HOURS_TO_FORECAST = st.slider("預測未來時數 (Lead Hours)", 1, 24, 6)
    LEAD_TIMES_TO_EVALUATE = [1, 3, 6]
    MODEL_PATH = 'flow_prediction_model.pth'

# --- 主頁面邏輯 ---
if uploaded_file is not None:
    st.session_state.run_demo = False
    data_source = uploaded_file
elif 'run_demo' in st.session_state and st.session_state.run_demo:
    st.info("您正在使用範例資料進行展示。")
    data_source = 'data/rainfall_flow_aligned.csv'
else:
    data_source = None

if data_source is not None:
    # 載入並準備資料
    # 這次先載入完整資料以取得日期範圍
    full_df = load_and_prepare_data(data_source, shift_hours=1)

    if full_df is not None:
        with st.sidebar:
            st.markdown("---")
            st.header("🗓️ 日期範圍篩選")
            min_date = full_df.index.min().date()
            max_date = full_df.index.max().date()

            # 如果是範例模式，設定預設日期
            if 'run_demo' in st.session_state and st.session_state.run_demo:
                start_date_default = pd.to_datetime("2022-09-01").date()
                end_date_default = pd.to_datetime("2022-09-30").date()
            else:
                start_date_default = min_date
                end_date_default = max_date

            start_date = st.date_input("開始日期", start_date_default, min_value=min_date, max_value=max_date)
            end_date = st.date_input("結束日期", end_date_default, min_value=start_date, max_value=max_date)

    # 載入模型 (會被快取)
    model, scaler, lookback, forecast_horizon = load_model(MODEL_PATH)
    
    # 顯示模型資訊
    st.sidebar.info(f"""
    **模型資訊:**
    - **Lookback:** `{lookback}` 小時
    - **Forecast Horizon:** `{forecast_horizon}` 小時
    """)

    # 根據選擇的日期範圍篩選資料，用於顯示預覽
    start_datetime_preview = pd.to_datetime(start_date)
    end_datetime_preview = pd.to_datetime(end_date) + pd.Timedelta(days=1, seconds=-1)
    df_preview = full_df.loc[start_datetime_preview:end_datetime_preview]

    st.subheader(f"資料預覽 (從 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')})")
    st.dataframe(df_preview.head())

    # 執行預測按鈕
    if st.button("🚀 執行預測", type="primary"):
        if df_preview.empty:
            st.error("選定的日期範圍內沒有資料，請重新選擇。")
        else:
            with st.spinner("正在準備資料並執行預測，請稍候..."):
                # 準備預測數據集
                # **重要：使用完整的 full_df 來創建 X，以確保有足夠的未來數據**
                FEATURE_COLS = ['flow', 'fcstrain_1h']
                raw_data_values = full_df[FEATURE_COLS].values
                X, y = create_dataset(raw_data_values, lookback, forecast_horizon)

                # 執行迭代預測
                df_pred_full = perform_iterative_forecast(full_df, model, scaler, X, lookback, forecast_horizon, LEAD_HOURS_TO_FORECAST)

                st.subheader("📊 預測結果")
                # 從完整的預測結果中，篩選出使用者想看的日期範圍
                df_pred = df_pred_full.loc[start_datetime_preview:end_datetime_preview]
                st.dataframe(df_pred)

                st.subheader("📈 評估與視覺化")
                
                # 針對要評估的 lead time 進行迭代
                for pred_hour in LEAD_TIMES_TO_EVALUATE:
                    if f'h{pred_hour}' not in df_pred.columns:
                        continue

                    st.markdown(f"--- \n ### 領先時間 (Lead Time): {pred_hour} 小時")
                    
                    # 準備觀測與預測數據以進行比對
                    pred_series = df_pred[f'h{pred_hour}'].copy()
                    pred_series.index += pd.to_timedelta(pred_hour, unit='h')
                    pred_series.name = 'pred_flow'
                    obs_series = full_df[['flow', 'rainfall']].copy() # 使用 full_df 來匹配觀測值
                    eval_df = pd.merge(obs_series, pred_series, left_index=True, right_index=True, how='inner')

                    if eval_df.empty:
                        st.warning(f"無法對齊 h{pred_hour} 的數據，跳過評估。")
                        continue

                    # 計算評估指標
                    obs_flow = eval_df['flow'].to_numpy()
                    pred_flow = eval_df['pred_flow'].to_numpy()
                    model_eva = ModelTest(qo=obs_flow, preq=pred_flow)

                    # 在網頁上顯示評估指標
                    col1, col2, col3 = st.columns(3)
                    col1.metric("效率係數 (CE)", f"{model_eva.get('CE', 0):.3f}")
                    col2.metric("相關係數 (COR)", f"{model_eva.get('COR', 0):.3f}")
                    col3.metric("洪峰誤差 (EQP %)", f"{model_eva.get('EQP', 0):.2f}%")

                    # 繪製圖表
                    fig = plot_prediction_vs_observation(
                        obs_df=eval_df[['flow', 'rainfall']],
                        pred_df=eval_df[['pred_flow']],
                        pred_hour=pred_hour,
                        metrics=model_eva
                    )
                    if fig:
                        st.pyplot(fig)

else:
    st.info("請從左側側邊欄上傳資料檔案以開始。")
