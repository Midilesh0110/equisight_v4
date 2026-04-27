import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Q-Learning Hyperparameters (The strictly allowed 4 params)
ALPHA = 0.1      # Learning Rate
GAMMA = 0.9      # Discount Factor (Prioritizes long-term vs short-term reward)
EPSILON = 0.05   # Exploration Rate (5% chance to try a random move during training)
ACTIONS = ['HOLD', 'BUY', 'SELL']

def get_price_location(current_price, p10, p50, p90):
    """Discretizes the continuous price into 3 distinct zones based on the ATR cones."""
    # Calculate distance to boundaries
    dist_p10 = abs(current_price - p10)
    dist_p50 = abs(current_price - p50)
    dist_p90 = abs(current_price - p90)
    
    # Find the closest boundary
    min_dist = min(dist_p10, dist_p50, dist_p90)
    
    if min_dist == dist_p10:
        return 'NEAR_P10'
    elif min_dist == dist_p90:
        return 'NEAR_P90'
    else:
        return 'NEAR_P50'

def load_or_create_q_table():
    """Loads the Q-Table memory or creates a blank one if this is day 1."""
    q_table_path = os.path.join('models', 'q_table.csv')
    os.makedirs('models', exist_ok=True)
    
    if os.path.exists(q_table_path):
        # Changed index_col='State' to index_col=0 to bypass header naming issues
        q_table = pd.read_csv(q_table_path, index_col=0)
        q_table.index.name = 'State' # Re-assign the name in memory
        return q_table
    else:
        # Create an empty Q-Table with zeros for all 9 possible states
        regimes = ['BULLISH', 'BEARISH', 'SIDEWAYS']
        locations = ['NEAR_P10', 'NEAR_P50', 'NEAR_P90']
        
        states = [f"{r}_{l}" for r in regimes for l in locations]
        q_table = pd.DataFrame(np.zeros((len(states), len(ACTIONS))), index=states, columns=ACTIONS)
        q_table.index.name = 'State'
        q_table.to_csv(q_table_path)
        return q_table

def execute_q_inference():
    today_str = datetime.now().strftime('%Y%m%d')
    cones_file = os.path.join('data', 'active_targets', f'risk_cones_{today_str}.csv')
    regime_file = os.path.join('data', 'active_targets', f'regime_states_{today_str}.csv')
    
    if not os.path.exists(cones_file) or not os.path.exists(regime_file):
        print("⚠️ [Q-AGENT] Missing state data (Cones or Regime). Standing down.")
        sys.exit(0)
        
    cones_df = pd.read_csv(cones_file).set_index('Ticker')
    regime_df = pd.read_csv(regime_file).set_index('Ticker')
    
    q_table = load_or_create_q_table()
    print("🤖 [Q-AGENT] Q-Table loaded. Initiating decision matrix...")
    
    decisions = []
    
    for ticker in cones_df.index:
        if ticker not in regime_df.index:
            continue
            
        # 1. Construct the State
        regime = regime_df.loc[ticker, 'Regime']
        price = cones_df.loc[ticker, 'Current_Price']
        p10 = cones_df.loc[ticker, 'P10_Bear']
        p50 = cones_df.loc[ticker, 'P50_Base']
        p90 = cones_df.loc[ticker, 'P90_Bull']
        
        location = get_price_location(price, p10, p50, p90)
        current_state = f"{regime}_{location}"
        
        # 2. Consult the Q-Table for the action with the highest expected reward
        # (Since we are purely inferencing right now, we pick the max value, no exploration)
        state_rewards = q_table.loc[current_state]
        best_action = state_rewards.idxmax()
        
        # Fallback safety: If all rewards are 0 (untrained state), default to HOLD
        if state_rewards.max() == 0:
            best_action = 'HOLD'
            
        print(f"   -> {ticker} | State: [{current_state}] | Action Selected: {best_action}")
        
        decisions.append({
            'Ticker': ticker,
            'State': current_state,
            'Action': best_action
        })
        
    # Save today's decisions
    decisions_df = pd.DataFrame(decisions)
    decisions_path = os.path.join('data', 'active_targets', f'decisions_{today_str}.csv')
    decisions_df.to_csv(decisions_path, index=False)
    print(f"✅ [Q-AGENT] Action phase complete. Directives saved to {decisions_path}")

if __name__ == "__main__":
    execute_q_inference()