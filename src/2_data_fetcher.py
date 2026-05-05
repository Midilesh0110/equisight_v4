import pandas as pd
import yfinance as yf
import numpy as np
import os

def validate_ohlcv_integrity(data, ticker):
    """Executes critical data sanity checks. Fails bad data immediately."""
    try:
        # 1. No Nulls
        if data.isnull().values.any():
            raise AssertionError("Null values detected in OHLCV array.")
            
        # 2. No Circuit Locks (Zero Volume)
        if (data['Volume'] <= 0).any():
            raise AssertionError("Zero-volume bars detected (Possible circuit lock).")
            
        # 3. Meaningful Volatility
        data['High-Low'] = data['High'] - data['Low']
        data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
        data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
        data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        
        atr_series = data['TR'].rolling(window=14).mean().dropna()
        if atr_series.empty:
            raise AssertionError("Insufficient data for ATR calculation.")
            
        current_atr = atr_series.iloc[-1]
        if current_atr <= 0.001:
            raise AssertionError(f"ATR is practically zero ({current_atr}). Division by zero risk.")
            
        # 4. No Extreme Black Swan Outliers
        log_atr = np.log(atr_series[atr_series > 0]) 
        if np.std(log_atr) >= 3:
            raise AssertionError(f"Extreme volatility outlier detected.")

        return True 

    except AssertionError as e:
        print(f"⚠️ [DATA SHIELD] {ticker} dropped: {e}")
        return False

def fetch_active_market_data():
    print("📡 [FETCHER] Initiating robust data ingestion...")
    
    # Read universe, strictly bypassing HDFC
    universe_path = 'config/nifty_universe.txt'
    with open(universe_path, 'r') as f:
        tickers = [line.strip() for line in f if line.strip() and 'HDFC' not in line.upper()]

    valid_data = {}
    for ticker in tickers:
        # Fetching 2 months to prevent holiday gaps from breaking the 14-day ATR
        data = yf.download(ticker, period="2mo", progress=False)
        
        if data.empty or len(data) < 15:
            print(f"⚠️ [FETCHER] {ticker} dropped: Insufficient trading days.")
            continue
            
        if validate_ohlcv_integrity(data, ticker):
            valid_data[ticker] = data
            print(f"✅ [FETCHER] {ticker} passed all integrity checks.")

    # Save validated data for Phase 3
    if not os.path.exists('data/active_targets'):
        os.makedirs('data/active_targets')
        
    for ticker, df in valid_data.items():
        df.to_csv(f"data/active_targets/{ticker}_clean.csv")
        
    print(f"🏁 [FETCHER] Pipeline secured. {len(valid_data)} targets ready for AI processing.")

if __name__ == "__main__":
    fetch_active_market_data()