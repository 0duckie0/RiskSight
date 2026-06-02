import yfinance as yf
import pandas as pd
import numpy as np

def get_live_features(ticker):
    print(f"📡 Fetching live data for {ticker}...")
    
    # 1. Define the ultimate baseline fallback so the model NEVER gets an empty array
    fallback_data = {
        'Current Ratio': 1.5, 'Quick Ratio': 1.2, 'Debt Equity Ratio': 0.5,
        'ROE': 0.15, 'Return On Assets': 0.05, 'PE Ratio': 20.0,
        'Price To Book Ratio': 2.5, 'Gross Profit Margin': 0.30,
        'Operating Profit Margin': 0.15, 'Net Profit Margin': 0.10,
        'Revenue Per Share': 50.0, 'Cash Per Share': 20.0,
        'Book Value Per Share': 100.0, 'Enterprise Value Over Ebitda': 12.0,
        'EV To Sales': 2.5, 'Dividend Payout Ratio': 0.0,
        'Net Income Per Share': 10.0, 'Price Earnings To Growth Ratio': 1.5
    }

    try:
        stock = yf.Ticker(ticker)
        
        # 2. Grab everything safely (preventing 'NoneType' errors)
        info = stock.info
        if not isinstance(info, dict):
            info = {}
            
        bs = stock.balance_sheet
        if bs is None or bs.empty:
            bs = pd.DataFrame()
        
        # --- MANUAL CALCULATIONS (The Backup Plan) ---
        manual_current_ratio = None
        try:
            if 'Total Current Assets' in bs.index and 'Total Current Liabilities' in bs.index:
                current_assets = bs.loc['Total Current Assets'].iloc[0]
                current_liab = bs.loc['Total Current Liabilities'].iloc[0]
                if current_liab != 0: # Prevent division by zero
                    manual_current_ratio = current_assets / current_liab
        except:
            pass
            
        manual_debt_equity = None
        try:
            if 'Total Debt' in bs.index and 'Stockholders Equity' in bs.index:
                total_debt = bs.loc['Total Debt'].iloc[0]
                total_equity = bs.loc['Stockholders Equity'].iloc[0]
                if total_equity != 0: # Prevent division by zero
                    manual_debt_equity = total_debt / total_equity
        except:
            pass

        # Helper function to safely extract values and avoid pandas NaNs
        def safe_get(key, default_val):
            val = info.get(key)
            if val is None or pd.isna(val):
                return default_val
            return val

        # --- THE SMART DICTIONARY ---
        # Logic: Use Yahoo Info -> OR Use Manual Math -> OR Use a Safe Neutral Value
        
        # Yahoo Finance returns Debt to Equity as a percentage (e.g. 50 instead of 0.5). We format it here:
        raw_debt_equity = safe_get('debtToEquity', None)
        final_debt_equity = (raw_debt_equity / 100) if raw_debt_equity is not None else (manual_debt_equity or 0.5)

        data = {
            'Current Ratio': safe_get('currentRatio', manual_current_ratio or fallback_data['Current Ratio']),
            'Quick Ratio': safe_get('quickRatio', (manual_current_ratio * 0.8 if manual_current_ratio else fallback_data['Quick Ratio'])),
            'Debt Equity Ratio': final_debt_equity,
            'ROE': safe_get('returnOnEquity', fallback_data['ROE']),
            'Return On Assets': safe_get('returnOnAssets', fallback_data['Return On Assets']),
            'PE Ratio': safe_get('trailingPE', fallback_data['PE Ratio']),
            'Price To Book Ratio': safe_get('priceToBook', fallback_data['Price To Book Ratio']),
            'Gross Profit Margin': safe_get('grossMargins', fallback_data['Gross Profit Margin']),
            'Operating Profit Margin': safe_get('operatingMargins', fallback_data['Operating Profit Margin']),
            'Net Profit Margin': safe_get('profitMargins', fallback_data['Net Profit Margin']),
            'Revenue Per Share': safe_get('revenuePerShare', fallback_data['Revenue Per Share']),
            'Cash Per Share': safe_get('totalCashPerShare', fallback_data['Cash Per Share']),
            'Book Value Per Share': safe_get('bookValue', fallback_data['Book Value Per Share']),
            'Enterprise Value Over Ebitda': safe_get('enterpriseToEbitda', fallback_data['Enterprise Value Over Ebitda']),
            'EV To Sales': safe_get('enterpriseToRevenue', fallback_data['EV To Sales']),
            'Dividend Payout Ratio': safe_get('payoutRatio', fallback_data['Dividend Payout Ratio']),
            'Net Income Per Share': safe_get('trailingEps', fallback_data['Net Income Per Share']),
            'Price Earnings To Growth Ratio': safe_get('pegRatio', fallback_data['Price Earnings To Growth Ratio'])
        }
        
        df = pd.DataFrame([data])
        df = df.fillna(0) # Catch any lingering NaNs
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e} | Falling back to baseline matrix.")
        # Returning a safe, baseline 1x18 array so your Streamlit App/Model NEVER crashes!
        return pd.DataFrame([fallback_data])
