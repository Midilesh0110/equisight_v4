import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

def fetch_training_data():
    """Fetches historical data for the Nifty universe to train the models."""
    universe_path = 'config/nifty_universe.txt'
    if not os.path.exists(universe_path):
        print("❌ [TRAINER] Universe file missing at config/nifty_universe.txt")
        return {}

    # Extract Nifty 50 universe, filtering out HDFC
    with open(universe_path, 'r') as f:
        tickers = [line.strip() for line in f if line.strip() and 'HDFC' not in line.upper()]

    print(f"📡 [TRAINER] Fetching historical data for {len(tickers)} tickers...")
    training_data = {}
    
    for ticker in tickers:
        try:
            # 2 years of data provides sufficient market cycles for HMM and Q-Learning
            data = yf.download(ticker, period="2y", progress=False)
            if len(data) > 50:
                data['Returns'] = data['Close'].pct_change()
                data = data.dropna()
                training_data[ticker] = data
        except Exception as e:
            continue
            
    return training_data

def train_hmm(training_data, n_states=4):
    """Trains the Gaussian Hidden Markov Model to identify market regimes."""
    print(f"🧠 [TRAINER] Training Hidden Markov Model with {n_states} states...")
    
    # Concatenate all returns to train a generalized market HMM
    all_features = []
    for ticker, df in training_data.items():
        features = df[['Returns']].values
        all_features.append(features)
        
    X = np.concatenate(all_features)
    
    hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    hmm_model.fit(X)
    
    return hmm_model

def train_q_agent(training_data, hmm_model, n_states=4):
    """Trains the Q-Learning agent to map HMM states to optimal actions."""
    print("🤖 [TRAINER] Initiating Q-Learning Reinforcement loop...")
    
    # Action Space: 0 = HOLD, 1 = BUY, 2 = SELL
    n_actions = 3
    q_table = np.zeros((n_states, n_actions))
    
    # Hyperparameters
    alpha = 0.1      # Learning Rate
    gamma = 0.95     # Discount Factor
    epsilon = 1.0    # Exploration Rate
    epsilon_decay = 0.995
    min_epsilon = 0.05
    
    epochs = 10 
    
    for epoch in range(epochs):
        for ticker, df in training_data.items():
            features = df[['Returns']].values
            states = hmm_model.predict(features)
            returns = df['Returns'].values
            
            for i in range(len(states) - 1):
                current_state = states[i]
                next_state = states[i+1]
                future_return = returns[i+1]
                
                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = np.random.choice(n_actions)
                else:
                    action = np.argmax(q_table[current_state])
                
                # Objective Reward Function based on forward returns
                if action == 1: # BUY
                    reward = future_return
                elif action == 2: # SELL (Short)
                    reward = -future_return
                else: # HOLD
                    reward = 0
                
                # Bellman Equation update
                best_next_action = np.argmax(q_table[next_state])
                td_target = reward + gamma * q_table[next_state][best_next_action]
                q_table[current_state][action] += alpha * (td_target - q_table[current_state][action])
                
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
    return q_table

def build_v4_models():
    """Master function to generate and save V4 AI components."""
    print("🛠️ [TRAINER] Initiating V4 Model Generation...")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    data = fetch_training_data()
    if not data:
        return
        
    hmm_model = train_hmm(data)
    q_table = train_q_agent(data, hmm_model)
    
    # Save the architecture
    hmm_path = 'models/equisight_hmm_v4.pkl'
    q_table_path = 'models/equisight_q_table_v4.npy'
    
    with open(hmm_path, 'wb') as f:
        pickle.dump(hmm_model, f)
        
    np.save(q_table_path, q_table)
    
    print(f"✅ [TRAINER] Success! Models saved to:")
    print(f"   - {hmm_path}")
    print(f"   - {q_table_path}")

if __name__ == "__main__":
    build_v4_models()