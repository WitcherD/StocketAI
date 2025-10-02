#!/usr/bin/env python3
"""
Debug script to compare different ways of fetching historical data from vnstock.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vnstock import Quote
from data_acquisition.vnstock_client import VNStockClient
import pandas as pd

def test_direct_quote_api():
    """Test direct Quote API call"""
    print("=== DIRECT QUOTE API ===")

    # Test with small date range (like the working test)
    print("\n1. Small date range (2024-01-01 to 2024-01-05):")
    try:
        quote = Quote(symbol='VCB')
        data = quote.history(symbol='VCB', start='2024-01-01', end='2024-01-05')
        print(f"Success! Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test with larger date range
    print("\n2. Larger date range (2023-01-01 to 2024-01-05):")
    try:
        quote = Quote(symbol='VCB')
        data = quote.history(symbol='VCB', start='2023-01-01', end='2024-01-05')
        print(f"Success! Shape: {data.shape}")
        print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    except Exception as e:
        print(f"Error: {e}")

def test_vnstock_client():
    """Test VNStockClient methods"""
    print("\n=== VNSTOCK CLIENT ===")

    client = VNStockClient()

    # Test with small date range
    print("\n1. Small date range via VNStockClient (2024-01-01 to 2024-01-05):")
    try:
        data = client.get_historical_data('VCB', '2024-01-01', '2024-01-05', '1D', 'VCI', False)
        print(f"Success! Shape: {data.shape}")
        print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test merged historical data with small range
    print("\n2. Merged historical data small range:")
    try:
        data = client.get_merged_historical_data('VCB', '2024-01-01', '2024-01-05', '1D')
        print(f"Success! Shape: {data.shape}")
        print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test with the problematic large range
    print("\n3. Large date range via VNStockClient (2015-10-02 to 2025-10-02):")
    try:
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        print(f"Date range: {start_date} to {end_date}")
        data = client.get_historical_data('VCB', start_date, end_date, '1D', 'VCI', False)
        print(f"Success! Shape: {data.shape}")
        print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    except Exception as e:
        print(f"Error: {e}")

def test_different_sources():
    """Test different data sources"""
    print("\n=== DIFFERENT SOURCES ===")

    client = VNStockClient()

    sources = ['VCI', 'TCBS', 'MSN']
    start_date = '2024-01-01'
    end_date = '2024-01-05'

    for source in sources:
        print(f"\nTesting source: {source}")
        try:
            data = client.get_historical_data('VCB', start_date, end_date, '1D', source, False)
            print(f"Success! Shape: {data.shape}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_direct_quote_api()
    test_vnstock_client()
    test_different_sources()
