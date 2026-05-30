import yfinance as yf

def get_qa_transcript(ticker):
    """
    Fetches live market news to provide context for the RAG model.
    We use news as a proxy for management narrative.
    """
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news
        
        if not news_items:
            return "No recent news or management narrative available to explain the financial state."
            
        narrative = f"Recent Market News & Context for {ticker}:\n\n"
        
        # Grab the top 4 most recent news headlines
        for item in news_items[:4]:
            title = item.get('title', 'No Title')
            publisher = item.get('publisher', 'Unknown Publisher')
            narrative += f"- [{publisher}]: {title}\n"
            
        return narrative
        
    except Exception as e:
        return f"Could not fetch live context. System Error: {e}"