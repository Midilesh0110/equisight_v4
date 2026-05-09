import pandas as pd
import yfinance as yf
import numpy as np
import os
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def extract_features_for_hmm(df):
    """
    Extracts features for the HMM dynamically without dropping NaNs 
    in the main dataframe so ATR calculations remain intact.
    """
    temp_df = df.copy()
    temp_df['Returns'] = temp_df['Close'].pct_change()
    temp_df = temp_df.dropna()
    return temp_df[['Returns']].values

def generate_daily_ledger():
    print("📊 [LEDGER] Running AI evaluation and calculating ATR bounds...")
    
    universe_path = 'config/nifty_universe.txt'
    if not os.path.exists(universe_path):
        print("⚠️ [LEDGER] Universe file missing at config/nifty_universe.txt")
        return

    with open(universe_path, 'r') as f:
        tickers = [line.strip() for line in f if line.strip() and 'HDFC' not in line.upper()]

    # Load V4 Models
    hmm_path = 'models/equisight_hmm_v4.pkl'
    q_table_path = 'models/equisight_q_table_v4.npy'

    if not os.path.exists(hmm_path) or not os.path.exists(q_table_path):
        print(f"❌ [LEDGER] CRITICAL: Models missing! Ensure {hmm_path} and {q_table_path} exist.")
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
            data = yf.download(ticker, period="2mo", progress=False)
            
            if data.empty or len(data) < 15:
                continue
            
            # Safely extract scalar price
            if isinstance(data['Close'].iloc[-1], pd.Series):
                current_price = float(data['Close'].iloc[-1].iloc[0])
            else:
                current_price = float(data['Close'].iloc[-1])

            # ==========================================
            # 1. CALCULATE ORIGINAL ATR TARGET BOUNDS
            # ==========================================
            data['High-Low'] = data['High'] - data['Low']
            data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
            data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
            data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
            
            atr_series = data['TR'].rolling(window=14).mean()
            
            if isinstance(atr_series.iloc[-1], pd.Series):
                atr = float(atr_series.iloc[-1].iloc[0])
            else:
                atr = float(atr_series.iloc[-1])

            p50 = current_price
            p90 = current_price + (atr * 1.5)
            p10 = current_price - (atr * 1.5)

            # ==========================================
            # 2. AI STATE EVALUATION (The Brain)
            # ==========================================
            features = extract_features_for_hmm(data)
            if len(features) == 0:
                continue

            hidden_states = hmm_model.predict(features)
            current_state = hidden_states[-1]

            q_values = q_table[current_state]
            action_index = np.argmax(q_values)
            
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
            # 3. CONSTRUCT ORIGINAL LEDGER FORMAT
            # ==========================================
            ledger_data.append({
                "Date": current_date,
                "Time": current_time,
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "Master Signal": master_signal,
                "P10 Target": round(p10, 2),
                "P50 Target": round(p50, 2),
                "P90 Target": round(p90, 2),
                "Reason": reason
            })
            
        except Exception as e:
            print(f"❌ [LEDGER] Error evaluating {ticker}: {e}")
            continue

    if ledger_data:
        df = pd.DataFrame(ledger_data)
        ledger_file = "equisight_v4_ledger.csv"
        
        if os.path.exists(ledger_file):
            df.to_csv(ledger_file, mode='a', header=False, index=False)
        else:
            df.to_csv(ledger_file, index=False)
            
        print(f"✅ [LEDGER] Evaluated and saved {len(ledger_data)} records to {ledger_file}")
        
if __name__ == "__main__":
    generate_daily_ledger()
    
