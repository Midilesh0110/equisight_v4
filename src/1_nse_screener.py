import pandas as pd
import yfinance as yf
import os
from datetime import datetime

def load_universe(filepath):
    """Loads the ticker universe from a config file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def calculate_volatility_score(tickers):
    """Scans the universe for pure mathematical momentum."""
    print(f"🔄 Scanning {len(tickers)} tickers for alpha...")
    selected_targets = []
    
    # Fetch 1 month of daily data for the baseline
    data = yf.download(tickers, period="1mo", interval="1d", group_by="ticker", progress=False)
    
    for ticker in tickers:
        try:
            # Extract data for the specific ticker
            df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
            
            if len(df) < 20:
                continue
                
            # Calculate True Range proxy (High-Low % for normalization)
            df['Daily_Range_%'] = ((df['High'] - df['Low']) / df['Low']) * 100
            
            # Short-term (last 3 days) vs Baseline Volatility (whole month)
            recent_volatility = df['Daily_Range_%'].tail(3).mean()
            baseline_volatility = df['Daily_Range_%'].mean()
            
            # Short-term vs Baseline Volume
            recent_volume = df['Volume'].tail(3).mean()
            baseline_volume = df['Volume'].mean()
            
            # Strict Filter: Must be 20% more volatile than norm AND have 50% more volume
            if (recent_volatility > baseline_volatility * 1.2) and (recent_volume > baseline_volume * 1.5):
                selected_targets.append({
                    'Ticker': ticker,
                    'Volatility_Score': recent_volatility,
                    'Volume_Multiplier': recent_volume / baseline_volume,
                    'Close_Price': df['Close'].iloc[-1]
                })
        except Exception as e:
            continue
            
    # Rank strictly by mathematical volatility score
    results_df = pd.DataFrame(selected_targets)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Volatility_Score', ascending=False).head(5)
    return results_df

def export_targets(df):
    """Saves the top targets for Module 2 to ingest."""
    # Ensure the target directory exists
    os.makedirs(os.path.join('data', 'active_targets'), exist_ok=True)
    
    today_str = datetime.now().strftime('%Y%m%d')
    filepath = os.path.join('data', 'active_targets', f'top_5_alpha_{today_str}.csv')
    
    if df.empty:
        print("⚠️ [SCREENER] No stocks met the aggressive momentum criteria today. Engine stands down.")
        # Create an empty file to signal downstream modules to skip inference
        pd.DataFrame(columns=['Ticker']).to_csv(filepath, index=False)
    else:
        print(f"🎯 [SCREENER] Top High-Alpha Targets Locked:\n{df[['Ticker', 'Volatility_Score', 'Volume_Multiplier']]}")
        df.to_csv(filepath, index=False)
        print(f"📁 Targets saved to {filepath}")

if __name__ == "__main__":
    universe_path = os.path.join('config', 'nifty_universe.txt')
    
    # Fallback to hardcoded list if config is missing
    if not os.path.exists(universe_path):
        print(f"⚠️ {universe_path} not found. Using fallback list.")
        tickers = ['RELIANCE.NS', 'TCS.NS', 'BAJFINANCE.NS', 'ZOMATO.NS', 'TATAMOTORS.NS']
    else:
        tickers = load_universe(universe_path)
        
    targets = calculate_volatility_score(tickers)
    export_targets(targets)