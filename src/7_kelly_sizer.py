import pandas as pd
import os
import sys
from datetime import datetime

# Portfolio Settings
PORTFOLIO_BALANCE = 100000.0  # ₹1 Lakh starting capital
MAX_KELLY_FRACTION = 0.05     # The "Fractional Kelly" rule: never risk more than 5% on one trade

def calculate_kelly(win_rate, avg_win, avg_loss):
    """Calculates the optimal capital allocation using the Kelly Criterion."""
    if avg_loss == 0 or win_rate == 0:
        return 0
    
    p = win_rate
    q = 1.0 - p
    b = avg_win / abs(avg_loss)
    
    kelly_fraction = p - (q / b)
    
    # If edge is negative, risk zero.
    if kelly_fraction <= 0:
        return 0 
        
    # Cap the maximum risk to prevent black swan blowups
    return min(kelly_fraction, MAX_KELLY_FRACTION)

def log_final_signals_to_master(ledger_updates):
    """Appends the official AI decisions to the master V4 ledger."""
    ledger_file = "equisight_v4_ledger.csv"
    
    if not os.path.exists(ledger_file):
        print("⚠️ [SIZER] Master ledger not found. Cannot append final signals.")
        return

    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')
    
    new_rows = []
    for update in ledger_updates:
        new_rows.append({
            "Date": current_date,
            "Time": current_time,
            "Ticker": update['Ticker'],
            "Current Price": round(float(update['Current_Price']), 2),
            "Master Signal": update['Signal'], 
            "P10 Target": round(float(update['P10']), 2),
            "P50 Target": round(float(update['P50']), 2),
            "P90 Target": round(float(update['P90']), 2),
            "Reason": update['Reason']
        })
        
    if new_rows:
        df_append = pd.DataFrame(new_rows)
        df_append.to_csv(ledger_file, mode='a', header=False, index=False)
        print(f"📝 [LEDGER] Appended {len(new_rows)} final execution signals to the master ledger.")

def execute_sizing():
    today_str = datetime.now().strftime('%Y%m%d')
    decisions_file = os.path.join('data', 'active_targets', f'decisions_{today_str}.csv')
    cones_file = os.path.join('data', 'active_targets', f'risk_cones_{today_str}.csv')
    
    if not os.path.exists(decisions_file) or not os.path.exists(cones_file):
        print("⚠️ [SIZER] Missing agent decisions or risk parameters. Standing down.")
        sys.exit(0)
        
    decisions_df = pd.read_csv(decisions_file).set_index('Ticker')
    cones_df = pd.read_csv(cones_file).set_index('Ticker')
    
    print(f"⚖️ [SIZER] Initiating Risk Matrix on ₹{PORTFOLIO_BALANCE:,.2f} bankroll...")
    
    orders = []
    ledger_updates = [] # Stores every decision for the Master Ledger
    
    for ticker in decisions_df.index:
        action = decisions_df.loc[ticker, 'Action']
        state = decisions_df.loc[ticker, 'State']
        
        current_price = cones_df.loc[ticker, 'Current_Price']
        atr = cones_df.loc[ticker, 'ATR_14']
        
        # Recalculate bounds for the ledger
        p50 = current_price
        p10 = current_price - (atr * 1.5)
        p90 = current_price + (atr * 1.5)
        
        # 1. Check if the Agent commanded a HOLD
        if action == 'HOLD':
            print(f"   -> {ticker} | Agent Directive: HOLD. Capital deployed: ₹0.00")
            ledger_updates.append({
                'Ticker': ticker, 'Current_Price': current_price, 'Signal': 'HOLD', 
                'P10': p10, 'P50': p50, 'P90': p90, 'Reason': "Agent Directive: HOLD. Edge insufficient."
            })
            continue
            
        # 2. Baseline Q-Table testing stats
        win_rate = 0.55 
        avg_win = 1.5
        avg_loss = 1.0
        
        kelly_pct = calculate_kelly(win_rate, avg_win, avg_loss)
        
        # 3. Kelly Override Defense
        if kelly_pct <= 0:
            print(f"   -> {ticker} | Kelly mathematically rejected the {action} trade (Negative Edge).")
            ledger_updates.append({
                'Ticker': ticker, 'Current_Price': current_price, 'Signal': 'HOLD', 
                'P10': p10, 'P50': p50, 'P90': p90, 'Reason': f"Kelly rejected {action} trade. Negative Edge."
            })
            continue
            
        # 4. Calculate exact position sizing
        capital_at_risk = PORTFOLIO_BALANCE * kelly_pct
        shares_to_buy = int(capital_at_risk / atr)
        total_trade_value = shares_to_buy * current_price
        
        print(f"   -> {ticker} | Edge Detected. Sizing: {kelly_pct*100:.2f}% | Risking: ₹{capital_at_risk:.2f}")
        print(f"       Execution: {action} {shares_to_buy} shares @ ₹{current_price} (Total Margin: ₹{total_trade_value:,.2f})")
        
        orders.append({
            'Ticker': ticker,
            'Action': action,
            'State': state,
            'Kelly_Risk_Pct': round(kelly_pct * 100, 2),
            'Shares': shares_to_buy,
            'Risk_Amount': round(capital_at_risk, 2),
            'Total_Position_Value': round(total_trade_value, 2)
        })
        
        ledger_updates.append({
            'Ticker': ticker, 'Current_Price': current_price, 'Signal': action, 
            'P10': p10, 'P50': p50, 'P90': p90, 'Reason': f"Kelly Matrix {kelly_pct*100:.2f}%. Sizing: {shares_to_buy} shares."
        })
        
    # Save the internal execution orders
    if orders:
        orders_df = pd.DataFrame(orders)
        orders_path = os.path.join('data', 'active_targets', f'final_orders_{today_str}.csv')
        orders_df.to_csv(orders_path, index=False)
        print(f"✅ [SIZER] Final trade orders locked and saved to {orders_path}")
    else:
        print(f"✅ [SIZER] No actionable orders to execute today. Preserving capital.")

    # Push the final state to the public Master Ledger
    if ledger_updates:
        log_final_signals_to_master(ledger_updates)

if __name__ == "__main__":
    execute_sizing()