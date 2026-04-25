import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def calculate_atr(df, period=14):
    """Calculates the Average True Range (ATR) ensuring columns are numeric."""
    # Force columns to numeric, turning errors into 'NaN'
    for col in ['High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows that failed the conversion
    df = df.dropna(subset=['High', 'Low', 'Close'])
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    # 14-day rolling average of the True Range
    atr = true_range.rolling(window=period).mean()
    return atr

def generate_probability_cones():
    today_str = datetime.now().strftime('%Y%m%d')
    target_file = os.path.join('data', 'active_targets', f'top_5_alpha_{today_str}.csv')
    
    if not os.path.exists(target_file):
        print("⚠️ [BOUNCER] No active targets file found. Standing down.")
        sys.exit(1)
        
    targets_df = pd.read_csv(target_file)
    if targets_df.empty:
        sys.exit(0)
        
    tickers = targets_df['Ticker'].tolist()
    print("🛡️ [BOUNCER] Calculating dynamic risk cones for active targets...")
    
    cone_results = []
    
    for ticker in tickers:
        data_path = os.path.join('data', 'mtf_data', f'{ticker}_1d.csv')
        if not os.path.exists(data_path):
            continue
            
        # skip_rows=2 is often needed for yfinance CSVs to bypass multi-index headers
        try:
            df = pd.read_csv(data_path, skiprows=2) 
            # If the CSV has specific headers like 'Price' or 'Ticker' in row 1, 
            # we re-assign them to match our logic
            df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        except:
            # Fallback for standard CSVs
            df = pd.read_csv(data_path)

        df['ATR'] = calculate_atr(df)
        
        # Get the most recent data point
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        current_atr = float(latest['ATR'])
        
        # Build the dynamic cones (1 ATR for standard deviation)
        p50_base = current_price
        p90_bull = current_price + current_atr
        p10_bear = current_price - current_atr
        
        cone_results.append({
            'Ticker': ticker,
            'Current_Price': round(current_price, 2),
            'ATR_14': round(current_atr, 2),
            'P10_Bear': round(p10_bear, 2),
            'P50_Base': round(p50_base, 2),
            'P90_Bull': round(p90_bull, 2)
        })
        
        print(f"   -> {ticker} | Price: ₹{current_price:.2f} | ATR: ₹{current_atr:.2f}")
        print(f"      [P10: ₹{p10_bear:.2f}] <--- [P50: ₹{p50_base:.2f}] ---> [P90: ₹{p90_bull:.2f}]")

    # Save the risk boundaries
    cones_df = pd.DataFrame(cone_results)
    cones_path = os.path.join('data', 'active_targets', f'risk_cones_{today_str}.csv')
    cones_df.to_csv(cones_path, index=False)
    print(f"✅ [BOUNCER] Dynamic probability cones locked and saved.")

if __name__ == "__main__":
    generate_probability_cones()