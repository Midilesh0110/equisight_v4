import sys
import datetime
import pandas_market_calendars as mcal

def check_market_open():
    """Validates if today is an active NSE trading day."""
    
    # The BSE (Bombay Stock Exchange) calendar mirrors the NSE holiday schedule
    nse = mcal.get_calendar('BSE')
    
    # Grab the current date
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Query the calendar to see if the market is open today
    schedule = nse.schedule(start_date=today, end_date=today)
    
    # If the schedule is empty, it's a weekend or a holiday
    if schedule.empty:
        print(f"🛑 [CALENDAR GATE] {today} is a market holiday or weekend. Terminating run.")
        sys.exit(0) # A zero exit code tells cloud pipelines to stop gracefully
    else:
        print(f"🟢 [CALENDAR GATE] Market is active on {today}. Ready for screener.")

if __name__ == "__main__":
    check_market_open()