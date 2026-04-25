import pandas as pd
import yfinance as yf
import os
import sys
from datetime import datetime

def fetch_deep_data():
    """Reads the active targets and downloads multi-timeframe data for them."""
    
    today_str = datetime.now().strftime('%Y%m%d')
    target_file = os.path.join('data', 'active_targets', f'top_5_alpha_{today_str}.csv')
    
    # 1. Check if the screener actually ran and produced a file
    if not os.path.exists(target_file):
        print(f"⚠️ [FETCHER] Target file not found. Did the screener run? Exiting.")
        sys.exit(1)
        
    targets_df = pd.read_csv(target_file)
    
    # 2. Check if the bouncer rejected everything
    if targets_df.empty:
        print("⚠️ [FETCHER] No active targets today. Standing down to preserve capital.")
        sys.exit(0)
        
    tickers = targets_df['Ticker'].tolist()
    print(f"📥 [FETCHER] Waking up. Pulling deep multi-timeframe data for: {tickers}")
    
    # Ensure our data directory exists
    os.makedirs(os.path.join('data', 'mtf_data'), exist_ok=True)
    
    # 3. Download the high-resolution data
    for ticker in tickers:
        print(f"   -> Fetching {ticker} (15-min Intraday & Daily)...")
        try:
            # Fetching 15m data for the last 60 days (max allowed by Yahoo for 15m)
            intraday = yf.download(ticker, period="60d", interval="15m", progress=False)
            # Fetching 1d data for the last 2 years for macro context
            daily = yf.download(ticker, period="2y", interval="1d", progress=False)
            
            # Save them to the local data lake
            intraday_path = os.path.join('data', 'mtf_data', f'{ticker}_15m.csv')
            daily_path = os.path.join('data', 'mtf_data', f'{ticker}_1d.csv')
            
            intraday.to_csv(intraday_path)
            daily.to_csv(daily_path)
        except Exception as e:
            print(f"   ❌ Failed to fetch data for {ticker}: {e}")
            
    print("✅ [FETCHER] High-resolution data secured. Ready for the Deep Inference engine.")

if __name__ == "__main__":
    fetch_deep_data()