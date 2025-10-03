import pandas as pd
import numpy as np
import os

def parse_data_file(filepath: str, value_col_name: str) -> pd.DataFrame:
    """
    解析特定格式的資料檔案（如 RFETS_flow.txt）。

    Args:
        filepath (str): 檔案路徑。
        value_col_name (str): 要指定的數值欄位名稱 (例如 'flow' 或 'rainfall')。

    Returns:
        pd.DataFrame: 一個以 datetime 為索引的 DataFrame。
    """
    try:
        # 使用 read_csv 讀取資料，並進行以下設定：
        # - comment=';': 忽略以 ';' 開頭的註解行
        # - sep='\s+': 使用一個或多個空白字元作為分隔符
        # - header=None: 檔案沒有標頭行
        # - usecols=[1, 2]: 只讀取第2欄（時間）和第3欄（數值）
        # - names=[...]: 指定欄位名稱
        df = pd.read_csv(
            filepath,
            comment=';',
            sep='\s+',
            header=None,
            usecols=[1, 2],
            names=['obstime', value_col_name],
            encoding='utf-8'
        )
        # 將 'obstime' 欄位轉換為 datetime 物件
        df['obstime'] = pd.to_datetime(df['obstime'])
        # 將 'obstime' 設為索引
        df.set_index('obstime', inplace=True)
        # 處理重複的索引（時間戳），保留第一個出現的值
        df = df[~df.index.duplicated(keep='first')]
        print(f"成功讀取並解析檔案: {filepath}, 共 {len(df)} 筆資料。")
        return df
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"讀取檔案 {filepath} 時發生錯誤: {e}")
        return pd.DataFrame()

def main():
    """
    主執行函式：讀取、對齊、填補並儲存資料。
    """
    # --- 1. 設定檔案路徑 ---
    rainfall_file = 'raw_data/09_RstAvgRain.txt'
    flow_file = 'raw_data/RFETS_flow.txt'
    output_dir = 'data'
    output_file = os.path.join(output_dir, 'rainfall_flow_aligned.csv')

    # --- 2. 讀取並解析資料 ---
    rainfall_df = parse_data_file(rainfall_file, 'rainfall')
    flow_df = parse_data_file(flow_file, 'flow')

    if rainfall_df.empty or flow_df.empty:
        print("因檔案讀取失敗，處理中止。")
        return

    # --- 3. 合併與對齊資料 ---
    # 使用 'outer' 合併，保留所有時間點，缺失處會自動填上 NaN
    merged_df = pd.merge(rainfall_df, flow_df, left_index=True, right_index=True, how='outer')

    # --- 4. 處理缺漏值 ---
    # 將所有 NaN 值填補為 -1
    merged_df.fillna(-1, inplace=True)
    print(f"\n資料合併完成。總時間範圍從 {merged_df.index.min()} 到 {merged_df.index.max()}。")

    # --- 5. 格式化並儲存 ---
    # 將索引 'obstime' 變回欄位
    merged_df.reset_index(inplace=True)
    # 新增一個從1開始的 index 欄位
    merged_df.insert(0, 'index', np.arange(1, len(merged_df) + 1))

    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"\n處理完成！已將對齊後的資料儲存至: {output_file}")

if __name__ == '__main__':
    main()