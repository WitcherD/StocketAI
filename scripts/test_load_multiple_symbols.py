#!/usr/bin/env python3
"""
Test script for load_multiple_symbols method in VN30DataLoader.

This script tests the load_multiple_symbols functionality for the 'ACB' symbol
to ensure the data loading pipeline works correctly with real vnstock API calls.

Usage:
    python test_load_multiple_symbols.py [options]

Options:
    --symbol SYMBOL       Stock symbol to test (default: ACB)
    --output-dir DIR      Base directory for data output (default: temp_test_data)
    --force-reload        Force reload data even if files exist
    --cleanup             Remove test data directory after completion
    --verbose             Enable verbose logging
    --help               Show this help message

Examples:
    python test_load_multiple_symbols.py
    python test_load_multiple_symbols.py --symbol VCB --force-reload --verbose
    python test_load_multiple_symbols.py --cleanup

Requirements:
    - Python environment with vnstock and data acquisition module installed
    - Internet connection for API calls
    - Project directory structure with src/ directory
"""

import sys
import os
import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_acquisition.vn30_data_loader import VN30DataLoader

def setup_logging(verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def test_load_multiple_symbols(symbol, output_dir, force_reload=False):
    """Test the load_multiple_symbols method."""
    logger = logging.getLogger(__name__)

    logger.info(f"Testing load_multiple_symbols for symbol: {symbol}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Force reload: {force_reload}")

    try:
        # Initialize data loader
        logger.info("Initializing VN30DataLoader...")
        loader = VN30DataLoader()

        # Test load_multiple_symbols
        logger.info(f"Loading data for symbol: {symbol}")
        start_time = datetime.now()

        result = loader.load_multiple_symbols(
            symbols=[symbol],
            base_path=output_dir,
            force_reload=force_reload
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(".2f")
        logger.info(f"Result: {result}")

        # Validate results
        success = validate_results(result, symbol, output_dir)
        return success, result

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False, None

def validate_results(result, symbol, output_dir):
    """Validate the results of the load_multiple_symbols call."""
    logger = logging.getLogger(__name__)

    logger.info("Validating results...")

    # Check result structure
    required_keys = ['total_symbols', 'successful_loads', 'skipped_loads', 'failed_loads', 'success_rate']
    for key in required_keys:
        if key not in result:
            logger.error(f"Missing required key in result: {key}")
            return False

    # Check result values
    if result['total_symbols'] != 1:
        logger.error(f"Expected total_symbols=1, got {result['total_symbols']}")
        return False

    if result['successful_loads'] + result['skipped_loads'] + result['failed_loads'] != 1:
        logger.error("Result counts don't add up to total_symbols")
        return False

    # Check directory structure
    symbol_dir = output_dir / 'data' / 'symbols' / symbol / 'raw'
    if not symbol_dir.exists():
        logger.error(f"Symbol directory not created: {symbol_dir}")
        return False

    # Check for expected data files
    expected_files = [
        'historical_price.csv',
        'financial_ratios_yearly.csv',
        'financial_ratios_quarterly.csv',
        'balance_sheet_yearly.csv',
        'balance_sheet_quarterly.csv',
        'income_statement_yearly.csv',
        'income_statement_quarterly.csv',
        'cash_flow_yearly.csv',
        'cash_flow_quarterly.csv',
        'company_profile.csv'
    ]

    found_files = []
    for file in expected_files:
        file_path = symbol_dir / file
        if file_path.exists():
            found_files.append(file)
            logger.debug(f"Found data file: {file}")
        else:
            logger.warning(f"Missing data file: {file}")

    if not found_files:
        logger.error("No data files were created")
        return False

    logger.info(f"Successfully created {len(found_files)} data files")
    logger.info("Validation completed successfully")
    return True

def print_summary(success, result, symbol, output_dir):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if success:
        print("✅ TEST PASSED")
        print(f"Symbol tested: {symbol}")
        print(f"Output directory: {output_dir}")
        print(f"Result: {result}")
    else:
        print("❌ TEST FAILED")
        if result:
            print(f"Result: {result}")

    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Test load_multiple_symbols method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--symbol',
        default='VCB',
        help='Stock symbol to test (default: VCB)'
    )
    parser.add_argument(
        '--output-dir',
        default='temp_test_data',
        help='Base directory for data output (default: temp_test_data)'
    )
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Force reload data even if files exist'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove test data directory after completion'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        # Convert output_dir to Path
        output_dir = Path(args.output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting load_multiple_symbols test")
        logger.info(f"Symbol: {args.symbol}")
        logger.info(f"Output directory: {output_dir.absolute()}")
        logger.info(f"Force reload: {args.force_reload}")

        # Run the test
        success, result = test_load_multiple_symbols(
            symbol=args.symbol,
            output_dir=output_dir,
            force_reload=args.force_reload
        )

        # Print summary
        print_summary(success, result, args.symbol, output_dir)

        # Cleanup if requested
        if args.cleanup and output_dir.exists():
            logger.info(f"Cleaning up test directory: {output_dir}")
            shutil.rmtree(output_dir)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
