import os
from sec_edgar_downloader import Downloader


def fetch_10k_filings(tickers, download_folder="sec_data", limit=2):
    """
    Downloads the most recent 10-K (Annual Report) filings for a list of company tickers.
    """
    # The SEC strictly requires a User-Agent string to monitor traffic.
    # Format: "CompanyName", "EmailAddress"
    # If you leave this blank, the SEC API will rate-limit or ban your IP.
    dl = Downloader("PortfolioProject", "hypetanbm@gmail.com", download_folder)

    print(f"Initializing connection to SEC EDGAR database...")
    print(f"Target directory: ./{download_folder}/\n")

    for ticker in tickers:
        print(
            f"Fetching the {limit} most recent 10-K filings for {ticker}...")
        try:
            # "10-K" is the specific form type for comprehensive annual reports
            dl.get("10-K", ticker, limit=limit)
            print(f"[SUCCESS] Downloaded filings for {ticker}.\n")
        except Exception as e:
            print(f"[ERROR] Failed to download {ticker}. Reason: {e}\n")


if __name__ == "__main__":
    # We will start with two highly complex, data-rich companies: Apple and Microsoft.
    target_companies = ["AAPL", "MSFT"]

    fetch_10k_filings(target_companies, limit=2)

    print("Ingestion pipeline complete. Verify the 'sec_data' folder in your directory.")
