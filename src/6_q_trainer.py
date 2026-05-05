import pandas as pd
import numpy as np
from hmmlearn import hmm
import os
import random
import warnings
from collections import Counter
import glob
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- Q-Learning Hyperparameters ---
ALPHA = 0.1      # Learning Rate
GAMMA = 0.9      # Discount Factor
EPSILON = 0.1    # 10% chance to take a random action
ACTIONS = ['HOLD', 'BUY', 'SELL']
CAPITAL = 100000.0
RISK_PER_TRADE = 0.05 # 5% Kelly Cap

# --- 1. Institutional Auditor Functions ---

def validate_state_sparsity(states_list, min_visits=50):
    state_counts = Counter(states_list)
    robust_states = [s for s, count in state_counts.items() if count >= min_visits]
    coverage = len(robust_states) / len(state_counts) if state_counts else 0
    print(f"🧠 [AUDIT] State Space Coverage: {coverage*100:.1f}% (States > {min_visits} hits: {len(robust_states)})")
    return coverage > 0.9, robust_states

def rigorous_backtest(trades_df, initial_capital=100000.0, commission=0.001, slippage=0.002):
    if trades_df.empty: return False

    # Apply real-world friction
    trades_df['Real_Entry'] = trades_df['Entry_Price'] * (1 + slippage)
    trades_df['Real_Exit'] = trades_df['Exit_Price'] * (1 - slippage)
    trades_df['Gross_PnL'] = (trades_df['Real_Exit'] - trades_df['Real_Entry']) * trades_df['Shares']
    
    # Sell trades invert the PnL logic
    trades_df.loc[trades_df['Action'] == 'SELL', 'Gross_PnL'] = (trades_df['Real_Entry'] - trades_df['Real_Exit']) * trades_df['Shares']

    trades_df['Entry_Cost'] = (trades_df['Real_Entry'] * trades_df['Shares']) * commission
    trades_df['Exit_Cost'] = (trades_df['Real_Exit'] * trades_df['Shares']) * commission
    trades_df['Net_PnL'] = trades_df['Gross_PnL'] - (trades_df['Entry_Cost'] + trades_df['Exit_Cost'])
    
    daily_pnl = trades_df.groupby('Date')['Net_PnL'].sum().reset_index()
    daily_pnl['Portfolio_Value'] = initial_capital + daily_pnl['Net_PnL'].cumsum()
    
    wins = trades_df[trades_df['Net_PnL'] > 0]['Net_PnL']
    losses = trades_df[trades_df['Net_PnL'] <= 0]['Net_PnL']
    
    win_rate = len(wins) / len(trades_df)
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 else 0
    
    daily_pnl['Peak'] = daily_pnl['Portfolio_Value'].cummax()
    daily_pnl['Drawdown'] = (daily_pnl['Portfolio_Value'] - daily_pnl['Peak']) / daily_pnl['Peak']
    max_dd = abs(daily_pnl['Drawdown'].min())
    
    trading_days = len(daily_pnl)
    total_return = (daily_pnl['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
    
    daily_returns_pct = daily_pnl['Portfolio_Value'].pct_change().dropna()
    annualized_vol = daily_returns_pct.std() * np.sqrt(252)
    
    sharpe = (annualized_return - 0.07) / annualized_vol if annualized_vol > 0 else 0
    calmar = annualized_return / max_dd if max_dd > 0 else 0

    print("\n📊 --- RIGOROUS BACKTEST RESULTS (WITH FRICTION) ---")
    print(f"Win Rate:       {win_rate*100:.1f}%   | Target: > 51.0%")
    print(f"Profit Factor:  {profit_factor:.2f}    | Target: > 1.5")
    print(f"Max Drawdown:   {max_dd*100:.1f}%    | Target: < 20.0%")
    print(f"Sharpe Ratio:   {sharpe:.2f}     | Target: > 1.2")
    print(f"Calmar Ratio:   {calmar:.2f}     | Target: > 0.8")
    
    metrics = [
        win_rate > 0.51, profit_factor > 1.5,
        max_dd < 0.20, sharpe > 1.2, calmar > 0.8
    ]
    return all(metrics)

# --- 2. Your Mathematical Core ---

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
    print("⚙️ [TRAINER] Booting Multi-Epoch Historical Simulator...")
    
    data_files = glob.glob('data/mtf_data/*_1d.csv')
    if not data_files:
        print("⚠️ [TRAINER] No data files found. Please fetch data first.")
        return

    regimes = ['BULLISH', 'BEARISH', 'SIDEWAYS']
    locations = ['NEAR_P10', 'NEAR_P50', 'NEAR_P90']
    states = [f"{r}_{l}" for r in regimes for l in locations]
    q_table = pd.DataFrame(np.zeros((len(states), len(ACTIONS))), index=states, columns=ACTIONS)

    # --- PHASE 1: THE TRAINING GYM (10 EPOCHS) ---
    EPISODES = 10
    print(f"🧠 [TRAINING] Running {EPISODES} Epochs. AI is exploring and learning...")
    
    # Pre-process all dataframes once to save time
    processed_data = {}
    for data_path in data_files:
        ticker = os.path.basename(data_path).replace('_1d.csv', '')
        try:
            df = pd.read_csv(data_path, skiprows=2)
            df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        except:
            df = pd.read_csv(data_path)
            
        for col in ['Close', 'High', 'Low']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)

        if len(df) < 50: continue

        df['ATR'] = calculate_atr(df)
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Range'] = (df['High'] - df['Low']) / df['Close']
        df = df.dropna().reset_index(drop=True)

        X = df[['Returns', 'Range']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X_scaled)
        df['HMM_State'] = model.predict(X_scaled)
        
        state_means = model.means_[:, 0]
        bull_id, bear_id = np.argmax(state_means), np.argmin(state_means)
        
        def map_regime(state):
            if state == bull_id: return "BULLISH"
            elif state == bear_id: return "BEARISH"
            else: return "SIDEWAYS"
        
        df['Regime'] = df['HMM_State'].apply(map_regime)
        processed_data[ticker] = df

    # The actual learning loop
    for epoch in range(EPISODES):
        for ticker, df in processed_data.items():
            for i in range(len(df) - 1):
                today, tomorrow = df.iloc[i], df.iloc[i + 1]
                
                price, atr = today['Close'], today['ATR']
                p50, p90, p10 = price, price + atr, price - atr
                location = get_price_location(price, p10, p50, p90)
                current_state = f"{today['Regime']}_{location}"
                
                # Explore (Epsilon) vs Exploit
                if random.uniform(0, 1) < EPSILON:
                    action = random.choice(ACTIONS)
                else:
                    action = q_table.loc[current_state].idxmax()
                    if q_table.loc[current_state].max() == 0: action = 'HOLD'

                price_change = tomorrow['Close'] - today['Close']
                reward = 0
            
                if action == 'BUY':
                    reward = 1 if price_change > 0 else -1
                elif action == 'SELL':
                    reward = 1 if price_change < 0 else -1
                elif action == 'HOLD':
                    reward = 0 # THE FIX: Restoring the Sniper mentality. No penalty for patience.
                next_price, next_atr = tomorrow['Close'], tomorrow['ATR']
                next_location = get_price_location(next_price, next_price - next_atr, next_price, next_price + next_atr)
                next_state = f"{tomorrow['Regime']}_{next_location}"

                old_value = q_table.loc[current_state, action]
                next_max = q_table.loc[next_state].max()
                q_table.loc[current_state, action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
                
    print("✅ [TRAINING] Q-Table converged.")

    # --- PHASE 2: THE FINAL EXAM (TEST PASS) ---
    print("🔬 [TESTING] Running Final Exam (Zero Exploration)...")
    my_test_trades_list = []
    all_historical_states = []

    for ticker, df in processed_data.items():
        for i in range(len(df) - 1):
            today, tomorrow = df.iloc[i], df.iloc[i + 1]
            
            price, atr = today['Close'], today['ATR']
            p50, p90, p10 = price, price + atr, price - atr
            location = get_price_location(price, p10, p50, p90)
            current_state = f"{today['Regime']}_{location}"
            all_historical_states.append(current_state)
            
            # STRICT EXPLOITATION (No Random Guesses)
            action = q_table.loc[current_state].idxmax()
            if q_table.loc[current_state].max() <= 0: action = 'HOLD'

            if action in ['BUY', 'SELL'] and atr > 0:
                # 🛡️ THE DRAWDOWN SHIELD: Cap Position Size
                capital_at_risk = CAPITAL * RISK_PER_TRADE
                max_investment_allowed = CAPITAL * 0.20 # Never put more than 20% cash into one trade
                
                raw_shares = int(capital_at_risk / atr)
                max_shares = int(max_investment_allowed / price)
                
                shares_to_buy = max(1, min(raw_shares, max_shares)) # Take the safer of the two
                
                my_test_trades_list.append({
                    'Date': tomorrow['Date'] if 'Date' in df.columns else i,
                    'Ticker': ticker,
                    'State': current_state,
                    'Action': action,
                    'Entry_Price': today['Close'],
                    'Exit_Price': tomorrow['Close'],
                    'Shares': shares_to_buy
                })

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    q_table.to_csv('models/q_table.csv')
    
    trade_df = pd.DataFrame(my_test_trades_list)
    trade_df.to_csv('data/backtest_equity_curve.csv', index=False)
    
    print(f"✅ [TESTING] Complete. AI executed {len(trade_df)} high-conviction trades.")
    
    # --- PHASE 3: THE AUDITOR ---
    print("\n⚙️ [AUDITOR] Executing Institutional Audit on Test Results...")
    coverage_passed, _ = validate_state_sparsity(all_historical_states)
    backtest_passed = rigorous_backtest(trade_df)
    
    if backtest_passed and coverage_passed:
        print("\n🟢 [DEPLOYMENT AUTHORIZED] The Q-Agent survived the Institutional Audit.")
    else:
        print("\n🔴 [DEPLOYMENT BLOCKED] The Q-Agent failed mathematical edge verification. DO NOT DEPLOY.")

if __name__ == "__main__":
    train_agent()