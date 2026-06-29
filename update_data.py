import yfinance as yf
import numpy as np
import pandas as pd
import json
from datetime import datetime

# 設定想要追蹤的商品清單（可自由增減）
tickers = ["QQQ", "00662.TW", "QLD", "VOO", "2330.TW"]

result_data = {}

for ticker in tickers:
    try:
        print(f"正在處理: {ticker}...")
        # 抓取 5 年數據，再精準篩選出最後 3.5 年（約 882 個交易日）
        df = yf.download(ticker, period="5y")
        if df.empty:
            continue
            
        # 處理 yfinance 可能產生的 MultiIndex 欄位結構
        if isinstance(df.columns, pd.MultiIndex):
            close_data = df['Close'][ticker]
        else:
            close_data = df['Close']
            
        close_data = close_data.dropna().tail(882)
        
        prices = close_data.values.flatten().tolist()
        dates = close_data.index.strftime('%Y-%m-%d').tolist()
        
        N = len(prices)
        X = np.arange(N)
        Y = np.array(prices)
        
        # 1. 線性迴歸計算：中線 TL (y = ax + b)
        a, b = np.polyfit(X, Y, 1)
        TL = a * X + b
        
        # 2. 計算殘差的標準差 (SD)
        residuals = Y - TL
        SD = np.std(residuals)
        
        # 3. 計算五線譜的五條線
        tl_p2 = TL + 2 * SD  # 極度樂觀
        tl_p1 = TL + 1 * SD  # 樂觀
        tl_m1 = TL - 1 * SD  # 悲觀
        tl_m2 = TL - 2 * SD  # 極度悲觀
        
        # 計算決定係數 R² (趨勢強度)
        r_squared = 1 - (np.sum(residuals**2) / np.sum((Y - np.mean(Y))**2))
        
        # 封裝該商品的數據
        result_data[ticker] = {
            "dates": dates,
            "actual": Y.tolist(),
            "tl": TL.tolist(),
            "tl_p2": tl_p2.tolist(),
            "tl_p1": tl_p1.tolist(),
            "tl_m1": tl_m1.tolist(),
            "tl_m2": tl_m2.tolist(),
            "last_price": float(Y[-1]),
            "r_squared": float(r_squared)
        }
    except Exception as e:
        print(f"處理 {ticker} 時發生錯誤: {e}")

# 儲存為 JSON 檔案
output = {
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data": result_data
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
print("data.json 更新成功！")
