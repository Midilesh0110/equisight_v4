import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime

def generate_daily_ledger():
    print("📊 [LEDGER] Generating daily evaluation bounds for the universe...")
    
    # Load universe and apply strict exclusions
    universe_path = 'config/nifty_universe.txt'
    if not os.path.exists(universe_path):
        print("⚠️ [LEDGER] Universe file missing.")
        return

    with open(universe_path, 'r') as f:
        # Exclude HDFC stocks from the scan per system rules
        tickers = [line.strip() for line in f if line.strip() and 'HDFC' not in line.upper()]

    ledger_data = []

    for ticker in tickers:
        try:
            # Fetch last 20 days to calculate baseline ATR
            data = yf.download(ticker, period="1mo", progress=False)
            if data.empty or len(data) < 20:
                continue
            
            # Basic ATR calculation for the cones
            data['High-Low'] = data['High'] - data['Low']
            data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
            data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
            data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
            atr = data['TR'].rolling(window=14).mean().iloc[-1]
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate P10, P50, P90
            p50 = current_price
            p90 = current_price + (atr * 1.5)  # 90th Percentile Bull
            p10 = current_price - (atr * 1.5)  # 10th Percentile Bear
            
            ledger_data.append({
                "Date": datetime.now().strftime('%Y-%m-%d'),
                "Ticker": ticker,
                "Current_Price": round(float(current_price), 2),
                "P10_Bear": round(float(p10), 2),
                "P50_Base": round(float(p50), 2),
                "P90_Bull": round(float(p90), 2),
                "Action": "LOGGED" # Baseline log; the actual agent will overwrite this if traded
            })
        except Exception as e:
            continue

    if ledger_data:
        df = pd.DataFrame(ledger_data)
        os.makedirs('data/ledger', exist_ok=True)
        filename = f"data/ledger/v4_evaluation_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ [LEDGER] Successfully saved {len(ledger_data)} stocks to {filename}")

if __name__ == "__main__":
    generate_daily_ledger()