#!/usr/bin/env python3
"""
Test script for VN30 constituents loading
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from data_acquisition.vnstock_client import VNStockClient

def test_vn30_loading():
    """Test VN30 constituents loading functionality."""
    print("Testing VN30 constituents loading...")

    try:
        # Initialize client
        client = VNStockClient()
        print("✓ VNStock client initialized")

        # Load VN30 constituents
        data = client.get_vn30_constituents()
        print(f"✓ Successfully loaded {len(data)} constituents")

        # Display first 5 symbols
        first_5 = data['symbol'].head().tolist()
        print(f"✓ First 5 symbols: {first_5}")

        # Basic validation
        assert len(data) > 0, "No constituents loaded"
        assert 'symbol' in data.columns, "Symbol column missing"
        assert data['symbol'].notna().all(), "Some symbols are NaN"

        print("✓ All validation checks passed")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_vn30_loading()
    sys.exit(0 if success else 1)
