import pandas as pd
import yfinance as yf
import numpy as np
import os
import pickle
import warnings
from datetime import datetime

# Ignore pandas warnings for a clean terminal output
warnings.filterwarnings('ignore')

def extract_features_for_hmm(data):
    """
    Calculates the technical features required by the HMM.
    """
    df = data.copy()
    
    # Calculate standard daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate ATR (Average True Range) for volatility
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Drop empty rows caused by rolling windows and pct_change
    df = df.dropna()
    
    # Isolate the exact features your HMM was trained on.
    # Note: If your training script used multiple columns (like Returns and ATR), 
    # adjust the array below to match your training shape (e.g., df[['Returns', 'ATR']].values)
    features = df[['Returns']].values 
    
    return features

def generate_daily_ledger():
    print("📊 [LEDGER] Running AI evaluation for the Nifty universe...")
    
    universe_path = 'config/nifty_universe.txt'
    if not os.path.exists(universe_path):
        print("⚠️ [LEDGER] Universe file missing at config/nifty_universe.txt")
        return

    # Read tickers, automatically skipping HDFC
    with open(universe_path, 'r') as f:
        tickers = [line.strip() for line in f if line.strip() and 'HDFC' not in line.upper()]

    # ==========================================
    # 1. LOAD YOUR V4 AI MODELS
    # ==========================================
    hmm_path = 'equisight_hmm_v4.pkl'
    q_table_path = 'equisight_q_table_v4.npy'

    if not os.path.exists(hmm_path) or not os.path.exists(q_table_path):
        print(f"❌ [LEDGER] CRITICAL: Models missing! Ensure {hmm_path} and {q_table_path} are in this directory.")
        return

    with open(hmm_path, 'rb') as f:
        hmm_model = pickle.load(f)
    
    q_table = np.load(q_table_path)

    ledger_data = []
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')

    for ticker in tickers:
        try:
            # Fetch recent data
            data = yf.download(ticker, period="2mo", progress=False)
            
            if data.empty or len(data) < 15:
                print(f"⚠️ [LEDGER] Insufficient data for {ticker}. Skipping.")
                continue
            
            # Safely extract scalar current price
            if isinstance(data['Close'].iloc[-1], pd.Series):
                current_price = float(data['Close'].iloc[-1].iloc[0])
            else:
                current_price = float(data['Close'].iloc[-1])

            # ==========================================
            # 2. AI STATE EVALUATION (The Eyes)
            # ==========================================
            features = extract_features_for_hmm(data)
            
            if len(features) == 0:
                continue

            # The HMM reads the recent data and identifies the current market regime
            hidden_states = hmm_model.predict(features)
            current_state = hidden_states[-1]

            # ==========================================
            # 3. Q-LEARNING DECISION (The Brain)
            # ==========================================
            # Lookup the expected value for this specific regime in the Q-table
            q_values = q_table[current_state]
            action_index = np.argmax(q_values)
            
            # Assuming standard Action Space: 0 = HOLD, 1 = BUY, 2 = SELL
            if action_index == 1:
                master_signal = "BUY"
                reason = f"AI Edge Confirmed (State {current_state})"
            elif action_index == 2:
                master_signal = "SELL"
                reason = f"AI Edge Confirmed (State {current_state})"
            else:
                master_signal = "HOLD"
                reason = f"Agent Directive: HOLD. (State {current_state})"

            # ==========================================
            # 4. LOG PURE MATH
            # ==========================================
            ledger_data.append({
                "Date": current_date,
                "Time": current_time,
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "Master Signal": master_signal,
                "Reason": reason
            })
            
        except Exception as e:
            print(f"❌ [LEDGER] Error evaluating {ticker}: {e}")
            continue

    # ==========================================
    # 5. SAVE TO LEDGER CSV
    # ==========================================
    if ledger_data:
        df = pd.DataFrame(ledger_data)
        ledger_file = "equisight_v4_ledger.csv"
        
        # Save file (Append if exists, create if new)
        if os.path.exists(ledger_file):
            df.to_csv(ledger_file, mode='a', header=False, index=False)
        else:
            df.to_csv(ledger_file, index=False)
            
        print(f"✅ [LEDGER] Successfully evaluated and appended {len(ledger_data)} records to {ledger_file}")
        
        # Print a quick summary of today's actions to the terminal
        action_counts = df['Master Signal'].value_counts()
        print(f"🎯 [SUMMARY] {action_counts.to_dict()}")
    else:
        print("❌ [LEDGER] CRITICAL: No data was generated for any ticker.")

if __name__ == "__main__":
    generate_daily_ledger()
    
