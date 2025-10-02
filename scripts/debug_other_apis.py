#!/usr/bin/env python3
"""
Debug script to check if other API methods have similar issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vnstock import Finance, Company
from data_acquisition.vnstock_client import VNStockClient
import pandas as pd

def test_finance_api_direct():
    """Test direct Finance API calls"""
    print("=== DIRECT FINANCE API ===")

    try:
        # Test direct Finance API
        finance = Finance(source='VCI', symbol='VCB', period='year', get_all=True, show_log=False)
        data = finance.ratio('VCB', period='year', lang='vi')
        print(f"Direct Finance.ratio() - Success! Shape: {data.shape}")
    except Exception as e:
        print(f"Direct Finance.ratio() - Error: {e}")

def test_company_api_direct():
    """Test direct Company API calls"""
    print("\n=== DIRECT COMPANY API ===")

    try:
        # Test direct Company API
        company = Company(source='TCBS', symbol='VCB')
        data = company.overview()
        print(f"Direct Company.overview() - Success! Shape: {data.shape}")
    except Exception as e:
        print(f"Direct Company.overview() - Error: {e}")

def test_vnstock_client_methods():
    """Test VNStockClient methods that use stock instance approach"""
    print("\n=== VNSTOCK CLIENT METHODS ===")

    client = VNStockClient()

    # Test methods that use self.vnstock.stock().finance or .company
    test_methods = [
        ('get_stock_info', lambda: client.get_stock_info('VCB', 'VCI')),
        ('get_financial_ratios', lambda: client.get_financial_ratios('VCB', 'yearly', 'VCI')),
        ('get_balance_sheet', lambda: client.get_balance_sheet('VCB', 'yearly', 'VCI')),
        ('get_income_statement', lambda: client.get_income_statement('VCB', 'yearly', 'VCI')),
        ('get_cash_flow', lambda: client.get_cash_flow('VCB', 'yearly', 'VCI')),
    ]

    for method_name, method_func in test_methods:
        try:
            data = method_func()
            if isinstance(data, dict):
                print(f"{method_name} - Success! Keys: {len(data)}")
            else:
                print(f"{method_name} - Success! Shape: {data.shape}")
        except Exception as e:
            print(f"{method_name} - Error: {e}")

if __name__ == "__main__":
    test_finance_api_direct()
    test_company_api_direct()
    test_vnstock_client_methods()
