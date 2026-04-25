import pandas as pd
import numpy as np
from hmmlearn import hmm
import os
import sys
from datetime import datetime
import warnings

# Suppress standard hmmlearn warnings for clean terminal output
warnings.filterwarnings("ignore")

def detect_regime():
    today_str = datetime.now().strftime('%Y%m%d')
    target_file = os.path.join('data', 'active_targets', f'top_5_alpha_{today_str}.csv')
    
    # 1. Check if we have active targets from Phase 1
    if not os.path.exists(target_file):
        print("⚠️ [REGIME] No active targets file found. Engine stands down.")
        sys.exit(0)
        
    targets_df = pd.read_csv(target_file)
    if targets_df.empty:
        sys.exit(0)
        
    tickers = targets_df['Ticker'].tolist()
    print(f"🧠 [REGIME] Waking up the HMM. Analyzing hidden market states for {tickers}...")
    
    regime_results = []
    
    for ticker in tickers:
        data_path = os.path.join('data', 'mtf_data', f'{ticker}_1d.csv')
        if not os.path.exists(data_path):
            print(f"   ❌ Missing 1d data for {ticker}. Skipping.")
            continue
            
        # 2. Load and clean the daily data we fetched in Phase 2
        try:
            df = pd.read_csv(data_path, skiprows=2)
            df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        except:
            df = pd.read_csv(data_path)
            
        # Force numeric conversion to prevent the string error we hit yesterday
        for col in ['Close', 'High', 'Low']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Close', 'High', 'Low'])

        # 3. Calculate the Features the HMM needs: Log Returns and Volatility
        # We use Log Returns instead of simple returns for better mathematical stability
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        # Range acts as a proxy for intraday volatility
        df['Range'] = (df['High'] - df['Low']) / df['Close']
        df = df.dropna()

        # Isolate the feature columns into an array for the model
        X = df[['Returns', 'Range']].values
        
        # 4. Initialize and Train the 3-State HMM
        # We use GaussianHMM because financial returns often resemble bell curves
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X)
        
        # Predict the hidden states for every day in the dataset
        states = model.predict(X)
        
        # We only care about the state of the *most recent* trading day
        current_state = states[-1]
        
        # 5. Interpret the States
        # The HMM assigns random IDs (0, 1, 2). We must determine what those IDs actually mean.
        # We do this by looking at the average 'Returns' associated with each state.
        state_means = model.means_[:, 0] 
        bull_state_id = np.argmax(state_means) # The state with the highest average return
        bear_state_id = np.argmin(state_means) # The state with the lowest average return
        
        # Assign the human-readable label
        if current_state == bull_state_id:
            regime_label = "BULLISH 📈"
        elif current_state == bear_state_id:
            regime_label = "BEARISH 📉"
        else:
            regime_label = "SIDEWAYS ↔️"
            
        print(f"   -> {ticker} is currently locked in a {regime_label} regime (HMM State: {current_state})")
        
        regime_results.append({
            'Ticker': ticker,
            'Regime': regime_label.split(" ")[0], # Strip the emoji for clean saving
            'State_ID': current_state
        })

    # 6. Save the findings
    regime_df = pd.DataFrame(regime_results)
    regime_path = os.path.join('data', 'active_targets', f'regime_states_{today_str}.csv')
    regime_df.to_csv(regime_path, index=False)
    print(f"✅ [REGIME] Market states securely logged to {regime_path}")

if __name__ == "__main__":
    detect_regime()