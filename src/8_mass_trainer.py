import pandas as pd
import numpy as np
from hmmlearn import hmm
import yfinance as yf
import os
import random
import warnings

warnings.filterwarnings("ignore")

# --- Q-Learning Hyperparameters ---
ALPHA = 0.1      # Learning Rate
GAMMA = 0.9      # Discount Factor
EPSILON = 0.1    # 10% chance to explore
ACTIONS = ['HOLD', 'BUY', 'SELL']

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return np.max(ranges, axis=1).rolling(window=period).mean()

def get_price_location(price, p10, p50, p90):
    dist_p10, dist_p50, dist_p90 = abs(price - p10), abs(price - p50), abs(price - p90)
    min_dist = min(dist_p10, dist_p50, dist_p90)
    if min_dist == dist_p10: return 'NEAR_P10'
    elif min_dist == dist_p90: return 'NEAR_P90'
    else: return 'NEAR_P50'

def run_mass_training():
    # 1. Load the active universe
    universe_path = os.path.join('config', 'nifty_universe.txt')
    if os.path.exists(universe_path):
        with open(universe_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        print("⚠️ Universe file missing. Fallback to default heavyweights.")
        tickers = ['INFY.NS', 'TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS']
        
    print(f"🏋️ [MASS TRAINER] Booting 10-Year simulation for {len(tickers)} tickers...")
    
    # 2. Initialize a blank Master Q-Table
    regimes = ['BULLISH', 'BEARISH', 'SIDEWAYS']
    locations = ['NEAR_P10', 'NEAR_P50', 'NEAR_P90']
    states = [f"{r}_{l}" for r in regimes for l in locations]
    q_table = pd.DataFrame(np.zeros((len(states), len(ACTIONS))), index=states, columns=ACTIONS)
    
    total_wins = 0
    total_losses = 0

    for ticker in tickers:
        print(f"   -> Downloading and simulating 10 years of data for {ticker}...")
        try:
            # Download directly into RAM (avoids filling your hard drive)
            df = yf.download(ticker, period="10y", interval="1d", progress=False)
            if df.empty or len(df) < 100:
                continue
                
            df = df.dropna()
            
            # Fix multi-index columns from yfinance's recent updates
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            # 3. Calculate technicals
            df['ATR'] = calculate_atr(df)
            df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Range'] = (df['High'] - df['Low']) / df['Close']
            df = df.dropna().reset_index(drop=True)
            
            # 4. Fit the HMM
            X = df[['Returns', 'Range']].values
            model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            model.fit(X)
            df['HMM_State'] = model.predict(X)
            
            state_means = model.means_[:, 0]
            bull_id, bear_id = np.argmax(state_means), np.argmin(state_means)
            
            def map_regime(state):
                if state == bull_id: return "BULLISH"
                elif state == bear_id: return "BEARISH"
                else: return "SIDEWAYS"
            
            df['Regime'] = df['HMM_State'].apply(map_regime)
            
            # 5. The Matrix Training Loop
            for i in range(len(df) - 1):
                today = df.iloc[i]
                tomorrow = df.iloc[i + 1]
                
                price, atr = float(today['Close']), float(today['ATR'])
                p50, p90, p10 = price, price + atr, price - atr
                location = get_price_location(price, p10, p50, p90)
                current_state = f"{today['Regime']}_{location}"
                
                if random.uniform(0, 1) < EPSILON:
                    action = random.choice(ACTIONS)
                else:
                    action = q_table.loc[current_state].idxmax()
                    if q_table.loc[current_state].max() == 0:
                        action = 'HOLD'

                price_change = float(tomorrow['Close']) - float(today['Close'])
                reward = 0
                
                if action == 'BUY':
                    reward = 1 if price_change > 0 else -1
                elif action == 'SELL':
                    reward = 1 if price_change < 0 else -1
                    
                if reward > 0: total_wins += 1
                elif reward < 0: total_losses += 1

                next_price, next_atr = float(tomorrow['Close']), float(tomorrow['ATR'])
                next_location = get_price_location(next_price, next_price - next_atr, next_price, next_price + next_atr)
                next_state = f"{tomorrow['Regime']}_{next_location}"

                old_value = q_table.loc[current_state, action]
                next_max = q_table.loc[next_state].max()
                new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
                
                q_table.loc[current_state, action] = new_value

        except Exception as e:
            print(f"      ❌ Failed on {ticker}: {e}")
            
    # 6. Save the fully educated Master Q-Table
    os.makedirs('models', exist_ok=True)
    q_table.to_csv('models/q_table.csv')
    
    total_trades = total_wins + total_losses
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"\n✅ [MASS TRAINER] 10-Year Macro-Training Complete!")
    print(f"📊 Total Actions Evaluated: {total_trades:,}")
    print(f"📈 Master Win Rate: {win_rate:.2f}% (Exploration vs Exploitation included)")

if __name__ == "__main__":
    run_mass_training()