import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# --- Institutional Execution Parameters ---
CAPITAL = 100000.0
RISK_PER_TRADE = 0.05 # 5% Kelly Fraction
MAX_ALLOCATION = 0.20 # 20% Hard Drawdown Cap
Q_TABLE_PATH = 'models/q_table.csv'

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def get_price_location(price, p10, p50, p90):
    dist_p10, dist_p50, dist_p90 = abs(price - p10), abs(price - p50), abs(price - p90)
    min_dist = min(dist_p10, dist_p50, dist_p90)
    if min_dist == dist_p10: return 'NEAR_P10'
    elif min_dist == dist_p90: return 'NEAR_P90'
    else: return 'NEAR_P50'

def determine_current_regime(df):
    """Dynamically calculates the current market regime using the last 200 days."""
    if len(df) < 50:
        return "SIDEWAYS" # Default safe state if data is missing
        
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    train_df = df.dropna().copy()
    
    X = train_df[['Returns', 'Range']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X_scaled)
    current_state = model.predict(X_scaled)[-1] # Get the very last state
    
    state_means = model.means_[:, 0]
    bull_id, bear_id = np.argmax(state_means), np.argmin(state_means)
    
    if current_state == bull_id: return "BULLISH"
    elif current_state == bear_id: return "BEARISH"
    else: return "SIDEWAYS"

def run_live_inference():
    print("🚀 [EXECUTION] Booting Live Inference Engine...")
    
    if not os.path.exists(Q_TABLE_PATH):
        print("🚨 [FATAL] Q-Table missing! Cannot execute trades without an authorized brain.")
        return

    # Load the Authorized Brain
    q_table = pd.read_csv(Q_TABLE_PATH, index_col=0)
    print("🧠 [EXECUTION] Authorized Q-Table loaded successfully.")
    
    # Read our active universe (strictly skipping HDFC)
    universe_path = 'config/nifty_universe.txt'
    with open(universe_path, 'r') as f:
        tickers = [line.strip() for line in f if line.strip() and 'HDFC' not in line.upper()]

    todays_orders = []
    
    print(f"📡 [EXECUTION] Scanning market conditions for {len(tickers)} targets...")

    for ticker in tickers:
        # Fetch the last 200 days to get accurate ATR and Regime data
        df = yf.download(ticker, period="200d", progress=False)
        
        if df.empty or len(df) < 20:
            continue
            
        # 🛠️ THE FIX: Flatten yfinance's new MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Clean data
        for col in ['Close', 'High', 'Low']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        # Calculate Current State Variables
        df['ATR'] = calculate_atr(df)
        today = df.iloc[-1]
        price, atr = today['Close'], today['ATR']
        
        if pd.isna(atr) or atr <= 0.001:
            continue # Drop illiquid/halted stocks
            
        p50 = price
        p90 = price + atr
        p10 = price - atr
        
        location = get_price_location(price, p10, p50, p90)
        regime = determine_current_regime(df)
        current_state = f"{regime}_{location}"
        
        # --- THE AI DECISION ---
        if current_state in q_table.index:
            # Exploit the Q-Table
            action = q_table.loc[current_state].idxmax()
            confidence = q_table.loc[current_state].max()
            
            # If the state has no mathematical edge, default to HOLD
            if confidence <= 0:
                action = 'HOLD'
        else:
            # If the AI has never seen this state before, protect capital
            action = 'HOLD'
            confidence = 0.0

        # --- POSITION SIZING & DRAWDOWN SHIELD ---
        shares_to_buy = 0
        capital_allocated = 0
        
        if action in ['BUY', 'SELL']:
            capital_at_risk = CAPITAL * RISK_PER_TRADE
            max_investment_allowed = CAPITAL * MAX_ALLOCATION
            
            raw_shares = int(capital_at_risk / atr)
            max_shares = int(max_investment_allowed / price)
            
            shares_to_buy = max(1, min(raw_shares, max_shares))
            capital_allocated = round(shares_to_buy * price, 2)
            
        # Log the order
        todays_orders.append({
            'Ticker': ticker,
            'Action': action,
            'State': current_state,
            'Price': round(price, 2),
            'Shares': shares_to_buy,
            'Capital_Req': capital_allocated
        })

    # Filter out HOLDS, sort, and STRICTLY CAP AT TOP 5 TRADES (Max 100% Capital)
    final_orders_df = pd.DataFrame(todays_orders)
    actionable_trades = final_orders_df[final_orders_df['Action'] != 'HOLD'].sort_values(by='Capital_Req', ascending=False).head(5)
    
    # Save the execution list
    os.makedirs('data', exist_ok=True)
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_file = f'data/execution_orders_{today_str}.csv'
    actionable_trades.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print(f"📊 EQUISIGHT V4 - FINAL EXECUTION DIRECTIVE")
    print("="*50)
    if actionable_trades.empty:
        print("🛑 No mathematically valid edges found today. Capital protected. HOLD.")
    else:
        print(actionable_trades.to_string(index=False))
        total_deployment = actionable_trades['Capital_Req'].sum()
        print("-" * 50)
        print(f"💰 Total Capital Deployment Required: ₹{total_deployment:,.2f}")
    print("="*50)
    print(f"✅ Execution orders saved to {output_file}")

if __name__ == "__main__":
    run_live_inference()