import yfinance as yf
import pandas as pd
import numpy as np

def get_live_features(ticker):
    print(f"📡 Fetching live data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Grab everything we can
        info = stock.info
        bs = stock.balance_sheet
        
        # --- MANUAL CALCULATIONS (The Backup Plan) ---
        # If Yahoo is missing the Current Ratio, we calculate it ourselves!
        try:
            current_assets = bs.loc['Total Current Assets'].iloc[0]
            current_liab = bs.loc['Total Current Liabilities'].iloc[0]
            manual_current_ratio = current_assets / current_liab
        except:
            manual_current_ratio = None
            
        try:
            total_debt = bs.loc['Total Debt'].iloc[0]
            total_equity = bs.loc['Stockholders Equity'].iloc[0]
            manual_debt_equity = total_debt / total_equity
        except:
            manual_debt_equity = None

        # --- THE SMART DICTIONARY ---
        # Logic: Use Yahoo Info -> OR Use Manual Math -> OR Use a Safe Neutral Value (Not 0!)
        data = {
            'Current Ratio': info.get('currentRatio') or manual_current_ratio or 1.5,
            'Quick Ratio': info.get('quickRatio') or (manual_current_ratio * 0.8 if manual_current_ratio else 1.2),
            'Debt Equity Ratio': (info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else None) or manual_debt_equity or 0.5,
            'ROE': info.get('returnOnEquity') or 0.15,
            'Return On Assets': info.get('returnOnAssets') or 0.05,
            'PE Ratio': info.get('trailingPE') or 20.0,
            'Price To Book Ratio': info.get('priceToBook') or 2.5,
            'Gross Profit Margin': info.get('grossMargins') or 0.30,
            'Operating Profit Margin': info.get('operatingMargins') or 0.15,
            'Net Profit Margin': info.get('profitMargins') or 0.10,
            'Revenue Per Share': info.get('revenuePerShare') or 50.0,
            'Cash Per Share': info.get('totalCashPerShare') or 20.0,
            'Book Value Per Share': info.get('bookValue') or 100.0,
            'Enterprise Value Over Ebitda': info.get('enterpriseToEbitda') or 12.0,
            'EV To Sales': info.get('enterpriseToRevenue') or 2.5,
            'Dividend Payout Ratio': info.get('payoutRatio') or 0.0,
            'Net Income Per Share': info.get('trailingEps') or 10.0,
            'Price Earnings To Growth Ratio': info.get('pegRatio') or 1.5
        }
        
        df = pd.DataFrame([data])
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()
