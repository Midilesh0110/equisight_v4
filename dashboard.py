import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Configure the page layout
st.set_page_config(page_title="EquiSight V4 Command Center", page_icon="🦅", layout="wide")

st.title("🦅 EquiSight V4: Quantitative Engine")
st.markdown("---")

today_str = datetime.now().strftime('%Y%m%d')

# Define file paths
targets_path = os.path.join('data', 'active_targets', f'top_5_alpha_{today_str}.csv')
cones_path = os.path.join('data', 'active_targets', f'risk_cones_{today_str}.csv')
regime_path = os.path.join('data', 'active_targets', f'regime_states_{today_str}.csv')
orders_path = os.path.join('data', 'active_targets', f'final_orders_{today_str}.csv')

# --- SECTION 1: Active Targets ---
st.subheader("🎯 Phase 1: High-Alpha Targets")
if os.path.exists(targets_path):
    targets_df = pd.read_csv(targets_path)
    # Bypass PyArrow by rendering as raw HTML
    st.markdown(targets_df.to_html(index=False), unsafe_allow_html=True)
else:
    st.warning("No active targets found for today.")

# --- SECTION 2: Brain Telemetry (HMM & Bouncer) ---
st.subheader("🧠 Phase 3 & 4: Market Context & Risk Cones")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Hidden Markov Model (Regime)**")
    if os.path.exists(regime_path):
        regime_df = pd.read_csv(regime_path)
        st.markdown(regime_df.to_html(index=False), unsafe_allow_html=True)
    else:
        st.info("Regime data unavailable.")

with col2:
    st.markdown("**Dynamic Volatility Cones (ATR)**")
    if os.path.exists(cones_path):
        cones_df = pd.read_csv(cones_path)
        st.markdown(cones_df.to_html(index=False), unsafe_allow_html=True)
    else:
        st.info("Risk cone data unavailable.")

# --- SECTION 3: Final Execution ---
st.markdown("---")
st.subheader("⚖️ Phase 7: Kelly Sizer & Final Directives")
if os.path.exists(orders_path):
    orders_df = pd.read_csv(orders_path)
    if not orders_df.empty:
        st.markdown(orders_df.to_html(index=False), unsafe_allow_html=True)
    else:
        st.info("Agent evaluated all states: CAPITAL PRESERVATION (HOLD). No actionable orders today.")
else:
    st.info("Agent evaluated all states: CAPITAL PRESERVATION (HOLD). No actionable orders today.")