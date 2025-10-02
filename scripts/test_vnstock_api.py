#!/usr/bin/env python3
"""
Test script to call vnstock API methods and document their response formats.
This script will help generate documentation for all vnstock API response formats.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'vnstock'))

from vnstock import Vnstock, Quote, Company, Finance, Listing, Trading, Screener, Fund
import json
import pandas as pd
from datetime import datetime, timedelta

def test_quote_api():
    """Test Quote API methods"""
    print("=== QUOTE API ===")

    quote = Quote(symbol='VCB')

    # Test intraday data (replaces stock price)
    try:
        print("\n1. Intraday Data (symbol='VCB', page_size=5):")
        data = quote.intraday(symbol='VCB', page_size=5)
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
        else:
            print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            print(f"Sample: {str(data)[:200]}...")
    except Exception as e:
        print(f"Error: {e}")

    # Test historical price
    try:
        print("\n2. Historical Price (symbol='VCB', start='2024-01-01', end='2024-01-05'):")
        data = quote.history(symbol='VCB', start='2024-01-01', end='2024-01-05')
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

def test_company_api():
    """Test Company API methods"""
    print("\n=== COMPANY API ===")

    company = Company(source='vci', symbol='VCB')

    # Test company overview
    try:
        print("\n1. Company Overview:")
        data = company.overview()
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            print("Sample data:")
            for key, value in list(data.items())[:3]:
                print(f"  {key}: {str(value)[:100]}...")
        elif hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

    # Test company shareholders
    try:
        print("\n2. Company Shareholders:")
        data = company.shareholders()
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

def test_finance_api():
    """Test Finance API methods"""
    print("\n=== FINANCE API ===")

    finance = Finance(source='vci', symbol='VCB', period='quarter', get_all=True, show_log=False)

    # Test balance sheet
    try:
        print("\n1. Balance Sheet (symbol='VCB', period='year', lang='vi'):")
        data = finance.balance_sheet('VCB', period='year', lang='vi')
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

    # Test income statement
    try:
        print("\n2. Income Statement (symbol='VCB', period='year', lang='vi'):")
        data = finance.income_statement('VCB', period='year', lang='vi')
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

    # Test cash flow statement
    try:
        print("\n3. Cash Flow Statement (symbol='VCB', period='year', lang='vi'):")
        data = finance.cash_flow('VCB', period='year', lang='vi')
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

    # Test financial ratios
    try:
        print("\n4. Financial Ratios (symbol='VCB', period='year', lang='vi'):")
        data = finance.ratio('VCB', period='year', lang='vi')
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

def test_listing_api():
    """Test Listing API methods"""
    print("\n=== LISTING API ===")

    listing = Listing()

    # Test all symbols
    try:
        print("\n1. All Symbols:")
        data = listing.all_symbols()
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

    # Test symbols by group
    try:
        print("\n2. Symbols by Group (group='VN30'):")
        data = listing.symbols_by_group('VN30')
        print(f"Type: {type(data)}")
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            print(f"Sample: {data[:3]}")
    except Exception as e:
        print(f"Error: {e}")

def test_trading_api():
    """Test Trading API methods"""
    print("\n=== TRADING API ===")

    trading = Trading(source='vci', symbol='VCB')

    # Test trading stats
    try:
        print("\n1. Trading Stats (limit=5):")
        data = trading.trading_stats(start="2024-01-01", end="2024-12-31", limit=5)
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

    # Test foreign trade as alternative
    try:
        print("\n3. Foreign Trade (limit=5):")
        data = trading.foreign_trade(limit=5)
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

    # Test price board
    try:
        print("\n2. Price Board:")
        data = trading.price_board(symbols_list=['VCB', 'VNM'])
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

def test_screener_api():
    """Test Screener API methods"""
    print("\n=== SCREENER API ===")

    screener = Screener()

    # Test stock screener
    try:
        print("\n1. Stock Screener (limit=5):")
        data = screener.stock(limit=5)
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

def test_fund_api():
    """Test Fund API methods"""
    print("\n=== FUND API ===")

    fund = Fund()

    # Test fund listing
    try:
        print("\n1. Fund Listing:")
        data = fund.listing()
        print(f"Type: {type(data)}")
        if hasattr(data, 'to_dict'):
            print(f"Columns: {list(data.columns)}")
            print(f"Shape: {data.shape}")
            print("Sample data:")
            print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

    # Test fund top holding
    try:
        print("\n2. Fund Top Holding (first fund):")
        # Get first fund symbol from listing
        fund_list = fund.listing()
        if not fund_list.empty:
            first_fund_symbol = fund_list['short_name'].iloc[0]
            data = fund.details.top_holding(first_fund_symbol)
            print(f"Type: {type(data)}")
            if hasattr(data, 'to_dict'):
                print(f"Columns: {list(data.columns)}")
                print(f"Shape: {data.shape}")
                print("Sample data:")
                print(data.head(2).to_dict())
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function to run all tests"""
    print("VNSTOCK API RESPONSE FORMAT DOCUMENTATION")
    print("=" * 50)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test all APIs
    test_quote_api()
    test_company_api()
    test_finance_api()
    test_listing_api()
    test_trading_api()
    test_screener_api()
    test_fund_api()

    print("\n" + "=" * 50)
    print("API testing completed!")

if __name__ == "__main__":
    main()
