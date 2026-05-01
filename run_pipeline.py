import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"🚀 EXECUTING: {script_path}")
    print(f"{'='*50}")
    
    try:
        # Run the script and stream the output to the console
        result = subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ [CRITICAL] Pipeline halted. {script_path} encountered an error or safely stood down.")
        sys.exit(1)

def execute_full_pipeline():
    print("🟢 INITIALIZING EQUISIGHT V4 PIPELINE...")
    
    # The exact chronological order of inference
    pipeline = [
        "src/0_daily_ledger.py",  # <--- NEW: Runs first to guarantee daily P10/P50/P90 logging
        "src/1_nse_screener.py",
        "src/2_data_fetcher.py",
        "src/3_dynamic_bouncer.py",
        "src/4_regime_detector.py",
        "src/5_q_agent.py",
        "src/7_kelly_sizer.py"
    ]
    
    for script in pipeline:
        run_script(script)
        
    print("\n🏁 [PIPELINE COMPLETE] All inference modules executed successfully.")

if __name__ == "__main__":
    execute_full_pipeline()