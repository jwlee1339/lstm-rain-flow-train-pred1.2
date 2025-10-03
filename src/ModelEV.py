# ModelEV.py
# 計算降雨逕流模式評估指標 CE

# ### 降雨逕流模式評估指標

import math
import numpy as np


def ModelTest(qo:np.ndarray, preq:np.ndarray) -> dict:
    """
    計算觀測(observed)與預測(predicted)時間序列的評估指標。
    此函式已優化，使用 NumPy 向量化操作以提升效率，並加入錯誤處理。

    Args:
        qo (np.ndarray): 觀測流量的時間序列陣列。
        preq (np.ndarray): 模式預測(模擬)流量的時間序列陣列。

    Returns:
        dict: 包含以下評估指標的字典：
              - CE (float): 效率係數 (Nash-Sutcliffe Efficiency)
              - VER (float): 體積誤差係數 (Volume Error Ratio)
              - COR (float): 相關係數 (Correlation Coefficient)
              - ETP (int): 洪峰時間誤差 (Error in Time to Peak)
              - EQP (float): 洪峰流量誤差百分比 (Error in Peak Flow)
              - pwMSE (float): 洪峰流量加權均方根誤差 (Peak-Weighted Root Mean Square Error)
    """
    # 初始化回傳結果的字典，設定預設值以防計算失敗
    res = {'CE': -999.0, 'VER': -999.0, 'COR': -999.0, 'ETP': -999, 'EQP': -999.0, 'pwMSE': -999.0}

    try:
        # 確保兩個序列長度一致
        if qo.size != preq.size:
            print(f'[WARN] 觀測與預測序列長度不符: qo.size={qo.size}, preq.size={preq.size}. 將以較短長度計算。')
            min_len = min(qo.size, preq.size)
            qo = qo[:min_len]
            preq = preq[:min_len]

        if qo.size == 0:
            print('[WARN] 輸入序列為空，無法計算評估指標。')
            return res

        # --- 基本統計量 ---
        qsum = np.sum(qo)
        esum = np.sum(preq)
        Qavg = np.mean(qo)
        Qeavg = np.mean(preq)
        Qp = np.max(qo)
        Qp1 = np.max(preq)

        # --- 計算各項指標所需的分量 ---
        # 均方差 (Mean Squared Error) 的分子部分
        xsum = np.sum(np.power(qo - preq, 2))
        # 觀測值與其平均值的離差平方和 (用於CE計算)
        ysum = np.sum(np.power(qo - Qavg, 2))
        
        # 相關係數 (COR) 的分子與分母部分
        Nsum = np.sum((qo - Qavg) * (preq - Qeavg))
        Dsum = ysum  # Dsum 與 ysum 相同
        Dsum1 = np.sum(np.power(preq - Qeavg, 2))

        # 洪峰加權均方根誤差 (pwMSE) 的加權平方和
        # 權重 w 使用觀測流量與平均流量的比值
        weights = qo / Qavg
        pwSum = np.sum(np.power(qo - preq, 2) * weights)

        # --- 計算最終指標 ---
        # 1. 效率係數 (Nash-Sutcliffe Efficiency, CE)
        #   - 衡量模型預測值與觀測值擬合程度。越接近1越好。
        res['CE'] = 1.0 - (xsum / ysum)

        # 2. 體積誤差係數 (Volume Error Ratio, VER)
        #   - 衡量總模擬體積與總觀測體積的差異百分比。越接近0越好。
        res['VER'] = ((esum - qsum) / qsum) * 100.0

        # 3. 相關係數 (Correlation Coefficient, COR)
        #   - 衡量預測值與觀測值的線性相關程度。越接近1越好。
        res['COR'] = Nsum / math.sqrt(Dsum * Dsum1)

        # 4. 洪峰時間誤差 (Error in Time to Peak, ETP)
        #   - 預測洪峰發生的時間與觀測洪峰時間的差距（單位：時間步）。
        res['ETP'] = int(np.argmax(preq) - np.argmax(qo))

        # 5. 洪峰流量誤差百分比 (Error in Peak Flow, EQP)
        #   - 預測洪峰流量與觀測洪峰流量的差異百分比。越接近0越好。
        res['EQP'] = ((Qp1 - Qp) / Qp) * 100.0

        # 6. 洪峰流量加權均方根誤差 (Peak-Weighted Root Mean Square Error, pwMSE)
        #   - 對高流量給予更高權重的誤差指標。越小越好。
        res['pwMSE'] = math.sqrt(pwSum) / qo.size

    except ZeroDivisionError:
        print("[ERROR] 計算評估指標時發生除以零錯誤。可能是因為所有觀測值都相同或為零。")
        # 回傳預設的錯誤值
        return res
    except Exception as e:
        print(f"[ERROR] 計算評估指標時發生未知錯誤: {e}")
        # 回傳預設的錯誤值
        return res

    return res


# Print Model Test Results
def PrintEvaluationFactors(res):
    print('* 模式評估指標 :')
    s = f"CE= {res['CE']:8.2f}, COR ={res['COR']:8.2f}, VER ={res['VER']:8.2f}, "
    s += f"HecObj ={res['pwMSE']:8.2f}"
    print(s)

'''
 ---------------------------------------------
 Model Objective Value
 input
   Qobs     : observed flow in cms
   Qt       : computed flow in cms
   iObjFunc : objective function index
 output
 return obj, eva_res
 ----------------------------------------------
'''
def ModelObjectiveValue(Qobs, Qcomp, iObjFunc=5):
    eva_res = ModelTest(Qobs, Qcomp)
    w = 0.55
    obj = -9999

    if iObjFunc == 1:
        obj = abs(1.0 - eva_res['CE'])
    elif iObjFunc == 2:
        obj = abs(eva_res['EQP'])
    elif iObjFunc == 3:
        obj = abs(1 - eva_res['COR'])
    elif iObjFunc == 4:      
        obj = w * abs(1 - eva_res['CE']) + (1.0 - w) * abs(eva_res['EQ'] / 100.0)
    elif iObjFunc == 5:
        obj = eva_res['pwMSE']
    return obj, eva_res


# main()
def main():
    obsFlow = np.array([0.0, 0.0, 50, 310, 590, 720, 850, 820, 600, 410, 270, 177, 114, 93, 88])
    compFlow = np.array([0.7687499999999999, 1.2365625, 55.71987890625001, 316.22973093749994, 
               597.6484270456054, 734.5773155272685, 865.7250035369163, 895.1267948904416,
                623.8426152741733, 452.8424095736908, 338.185670627656, 248.72905447369232, 
                187.4674284953931, 174.95154425023685, 161.6783751157142])

    res = ModelTest(obsFlow, compFlow)
    PrintEvaluationFactors(res)

    result = ModelObjectiveValue(obsFlow, compFlow, 1)
    print(f'目標函數值 : 1-CE = {result:9.4f}')


if __name__ == '__main__':
    main()	