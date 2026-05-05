import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime

def generate_daily_ledger():
    print("📊 [LEDGER] Generating daily evaluation bounds for the universe...")
    
    universe_path = 'config/nifty_universe.txt'
    if not os.path.exists(universe_path):
        print("⚠️ [LEDGER] Universe file missing.")
        return

    with open(universe_path, 'r') as f:
        tickers = [line.strip() for line in f if line.strip() and 'HDFC' not in line.upper()]

    ledger_data = []
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')

    for ticker in tickers:
        try:
            # FIX 1: Fetch 2 months to guarantee we bypass holiday shortages
            data = yf.download(ticker, period="2mo", progress=False)
            
            # FIX 2: We only need 14 days for our ATR, so 15 is a safer floor
            if data.empty or len(data) < 15:
                print(f"⚠️ [LEDGER] Insufficient data for {ticker}. Days fetched: {len(data)}")
                continue
            
            data['High-Low'] = data['High'] - data['Low']
            data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
            data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
            data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
            atr = data['TR'].rolling(window=14).mean().iloc[-1]
            
            current_price = data['Close'].iloc[-1]
            
            p50 = current_price
            p90 = current_price + (atr * 1.5)
            p10 = current_price - (atr * 1.5)
            
            ledger_data.append({
                "Date": current_date,
                "Time": current_time,
                "Ticker": ticker,
               # Use .iloc[0] to ensure we are grabbing the scalar value before converting to float
                "Current Price": round(float(current_price.iloc[0]), 2),
                "Master Siganl": "LOGGED",
                "P10 Target": round(float(p10.iloc[0]), 2),
                "P50 Target": round(float(p50.iloc[0]), 2),
                "P90 Target": round(float(p90.iloc[0]), 2),
                "Reason": "Baseline bounds recorded."
            })
        except Exception as e:
            # FIX 3: Un-silence the errors so we can catch IP blocks
            print(f"❌ [LEDGER] Error processing {ticker}: {e}")
            continue

    if ledger_data:
        df = pd.DataFrame(ledger_data)
        ledger_file = "equisight_v4_ledger.csv"
        
        if os.path.exists(ledger_file):
            df.to_csv(ledger_file, mode='a', header=False, index=False)
        else:
            df.to_csv(ledger_file, index=False)
            
        print(f"✅ [LEDGER] Successfully appended {len(ledger_data)} records to {ledger_file}")
    else:
        print("❌ [LEDGER] CRITICAL: No data was generated for any ticker.")

if __name__ == "__main__":
    generate_daily_ledger()
