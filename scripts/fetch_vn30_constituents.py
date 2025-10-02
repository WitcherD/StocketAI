#!/usr/bin/env python3
"""
Fetch VN30 constituents data using the data acquisition module.

This script fetches current VN30 index constituents data and saves it to the data/raw directory.
It uses the VN30ConstituentsFetcher class from the data acquisition module.

Usage:
    python fetch_vn30_constituents.py [options]

Options:
    --output-path PATH    Path to save the constituents data (default: data/raw/vn30_constituents.json)
    --force-refresh       Bypass cache and fetch fresh data from the API
    --format FORMAT       Output format: json, csv, or excel (default: json)
    --help               Show this help message

Examples:
    python fetch_vn30_constituents.py
    python fetch_vn30_constituents.py --force-refresh --format excel --output-path "data/raw/constituents.xlsx"

Requirements:
    - Python environment with vnstock and data acquisition module installed
    - Project directory structure with src/ and data/ directories
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_acquisition import VN30ConstituentsFetcher

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fetch_vn30_constituents.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='Fetch VN30 constituents data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--output-path',
        default='data/raw/vn30_constituents.json',
        help='Path to save the constituents data'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Bypass cache and fetch fresh data from the API'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'excel'],
        default='json',
        help='Output format'
    )

    args = parser.parse_args()

    try:
        logger.info("Initializing VN30ConstituentsFetcher...")
        logger.info(f"Output Path: {args.output_path}")
        logger.info(f"Format: {args.format}")
        logger.info(f"Force Refresh: {args.force_refresh}")

        # Initialize fetcher
        fetcher = VN30ConstituentsFetcher()

        logger.info("Fetching VN30 constituents data...")

        # Fetch constituents data
        constituents_data = fetcher.fetch_constituents(force_refresh=args.force_refresh)

        logger.info(f"Successfully fetched {len(constituents_data.constituents)} constituents")

        # Prepare output data
        output_data = {
            'metadata': {
                'index_name': constituents_data.index_name,
                'last_updated': constituents_data.last_updated.isoformat(),
                'total_constituents': len(constituents_data.constituents),
                'total_market_cap': constituents_data.total_market_cap,
                'average_weight': constituents_data.average_weight,
                'exported_at': datetime.now().isoformat(),
                'exported_by': 'fetch_vn30_constituents.py'
            },
            'constituents': [c.to_dict() for c in constituents_data.constituents]
        }

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Export based on format
        if args.format.lower() == 'json':
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported to JSON: {args.output_path}")

        elif args.format.lower() == 'csv':
            import pandas as pd

            # Convert to DataFrame
            df_data = []
            for c in constituents_data.constituents:
                row = c.to_dict()
                row['last_updated'] = row['last_updated']
                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(args.output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Exported to CSV: {args.output_path}")

        elif args.format.lower() == 'excel':
            import pandas as pd

            # Convert to DataFrame
            df_data = []
            for c in constituents_data.constituents:
                row = c.to_dict()
                row['last_updated'] = row['last_updated']
                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_excel(args.output_path, index=False)
            logger.info(f"Exported to Excel: {args.output_path}")

        # Print summary
        print("\n=== VN30 Constituents Summary ===")
        print(f"Index: {constituents_data.index_name}")
        print(f"Total Constituents: {len(constituents_data.constituents)}")
        print(f"Total Market Cap: {constituents_data.total_market_cap:,.0f}")
        print(f"Average Weight: {constituents_data.average_weight:.2f}%")
        print(f"Last Updated: {constituents_data.last_updated}")

        print("\n=== Top 5 Constituents by Weight ===")
        sorted_constituents = sorted(constituents_data.constituents, key=lambda x: x.weight, reverse=True)
        for i, c in enumerate(sorted_constituents[:5], 1):
            market_cap_display = f"{c.market_cap:,.0f}" if c.market_cap is not None else "N/A"
            print(f"{i}. {c.symbol} - Weight: {c.weight:.2f}% - Market Cap: {market_cap_display}")

    except Exception as e:
        logger.error(f"Error occurred while fetching VN30 constituents: {str(e)}")
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
