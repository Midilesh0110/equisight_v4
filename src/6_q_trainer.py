import pandas as pd
import numpy as np
from hmmlearn import hmm
import os
import random
import warnings

warnings.filterwarnings("ignore")

# --- Q-Learning Hyperparameters ---
ALPHA = 0.1      # Learning Rate
GAMMA = 0.9      # Discount Factor
EPSILON = 0.1    # 10% chance to take a random action to discover new strategies
ACTIONS = ['HOLD', 'BUY', 'SELL']

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

def train_agent():
    ticker = 'INFY.NS'
    data_path = os.path.join('data', 'mtf_data', f'{ticker}_1d.csv')
    
    if not os.path.exists(data_path):
        print(f"⚠️ [TRAINER] Missing historical data for {ticker}.")
        return

    print(f"⚙️ [TRAINER] Booting historical simulator for {ticker}...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(data_path, skiprows=2)
        df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    except:
        df = pd.read_csv(data_path)
    
    for col in ['Close', 'High', 'Low']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)

    # 2. Pre-calculate ATR and HMM Regimes for the whole timeline
    print("   -> Calculating historical ATR and HMM states...")
    df['ATR'] = calculate_atr(df)
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    df = df.dropna().reset_index(drop=True)

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

    # 3. Initialize Q-Table
    regimes = ['BULLISH', 'BEARISH', 'SIDEWAYS']
    locations = ['NEAR_P10', 'NEAR_P50', 'NEAR_P90']
    states = [f"{r}_{l}" for r in regimes for l in locations]
    q_table = pd.DataFrame(np.zeros((len(states), len(ACTIONS))), index=states, columns=ACTIONS)

    # 4. The Training Loop (Stepping through time)
    print("   -> Initiating Bellman Equation training loop...")
    wins, losses = 0, 0
    
    for i in range(len(df) - 1): # We stop 1 day early so we can see "tomorrow's" reward
        today = df.iloc[i]
        tomorrow = df.iloc[i + 1]
        
        # Build Current State
        price, atr = today['Close'], today['ATR']
        p50, p90, p10 = price, price + atr, price - atr
        location = get_price_location(price, p10, p50, p90)
        current_state = f"{today['Regime']}_{location}"
        
        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < EPSILON:
            action = random.choice(ACTIONS) # Explore
        else:
            action = q_table.loc[current_state].idxmax() # Exploit best known move
            if q_table.loc[current_state].max() == 0:
                action = 'HOLD'

        # Calculate Reward based on tomorrow's price action
        price_change = tomorrow['Close'] - today['Close']
        reward = 0
        
        if action == 'BUY':
            reward = 1 if price_change > 0 else -1
        elif action == 'SELL':
            reward = 1 if price_change < 0 else -1
            
        if reward > 0: wins += 1
        elif reward < 0: losses += 1

        # Build Next State
        next_price, next_atr = tomorrow['Close'], tomorrow['ATR']
        next_location = get_price_location(next_price, next_price - next_atr, next_price, next_price + next_atr)
        next_state = f"{tomorrow['Regime']}_{next_location}"

        # Bellman Equation Update
        old_value = q_table.loc[current_state, action]
        next_max = q_table.loc[next_state].max()
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        
        q_table.loc[current_state, action] = new_value

    # 5. Save the trained brain
    os.makedirs('models', exist_ok=True)
    q_table.to_csv('models/q_table.csv')
    
    win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
    print(f"✅ [TRAINER] Training complete. Q-Table updated.")
    print(f"📊 Training Simulation Win Rate: {win_rate:.2f}% (Exploration vs Exploitation)")

if __name__ == "__main__":
    train_agent()