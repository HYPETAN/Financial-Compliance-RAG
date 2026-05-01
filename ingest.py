from sec_edgar_downloader import Downloader
import time


def batch_download_sec_data():
    print("Initializing Enterprise SEC Downloader for Scale...")

    dl = Downloader("RAG_Project", "hypetanbm@gmail.com", "sec_data")

    # 10 Companies across 5 highly diverse domains
    target_companies = ["AAPL", "MSFT", "TSLA", "F",
                        "JPM", "GS", "JNJ", "PFE", "XOM", "CVX"]

    for ticker in target_companies:
        print(f"\n--- Fetching 10-K history for {ticker} ---")
        try:
            # limit=3 fetches the 3 most recent 10-K filings
            dl.get("10-K", ticker, limit=3, download_details=True)
            print(f"[SUCCESS] Downloaded 3 years of data for {ticker}")

            # Sleep for 2 seconds between companies to strictly respect SEC rate limits
            time.sleep(2)
        except Exception as e:
            print(f"[ERROR] Failed to download {ticker}: {e}")


if __name__ == "__main__":
    batch_download_sec_data()
