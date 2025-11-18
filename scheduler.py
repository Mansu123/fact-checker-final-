import schedule
import time
from datetime import datetime
from news_agent import NewsCollectionAgent

# Initialize the agent
agent = NewsCollectionAgent()

def run_daily_collection_job():
    """Job to run daily news collection"""
    print("\n" + "="*70)
    print(f"🕒 Scheduled job started at {datetime.now()}")
    print("="*70 + "\n")
    
    try:
        agent.run_daily_collection()
        print(f"\n✓ Job completed successfully at {datetime.now()}")
    except Exception as e:
        print(f"\n✗ Job failed: {e}")

# Schedule the job to run every day at 6:00 AM
schedule.every().day.at("06:00").do(run_daily_collection_job)

# Alternative: Run every 12 hours
# schedule.every(12).hours.do(run_daily_collection_job)

# Run immediately on start for testing
print("🚀 News Collection Scheduler Started")
print(f"⏰ Current time: {datetime.now()}")
print("📅 Scheduled: Every day at 6:00 AM")
print("="*70)

# Run once immediately for testing
print("\n🧪 Running initial collection...")
run_daily_collection_job()

print("\n⏳ Waiting for next scheduled run...")
print("   Press Ctrl+C to stop the scheduler\n")

# Keep the scheduler running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
