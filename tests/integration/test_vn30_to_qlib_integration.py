#!/usr/bin/env python3
"""
Integration test for VN30 to Qlib converter.

Tests the complete workflow of converting VCB raw data to qlib format
using the Vn30ToQlibConverter class with real data.
"""

import sys
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_processing.vn30_to_qlib_converter import Vn30ToQlibConverter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestVn30ToQlibIntegration:
    """Integration test class for VN30 to Qlib converter."""

    def __init__(self):
        self.test_dir = Path(__file__).parent / "integration_test_data" / "symbols"
        self.test_symbol = "VCB"
        self.test_results = {
            'data_validation': False,
            'data_cleaning': False,
            'qlib_format_preparation': False,
            'binary_conversion': False,
            'error_handling': False,
            'overall_success': False
        }

    def setup_test_environment(self):
        """Set up test environment with VCB data."""
        logger.info("Setting up test environment...")

        try:
            # Ensure test directory exists
            test_symbol_dir = self.test_dir / self.test_symbol
            raw_dir = test_symbol_dir / 'raw'
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Copy historical price data if it exists
            source_file = raw_dir / 'historical_price.csv'
            if source_file.exists():
                logger.info(f"✅ Test data already exists at {source_file}")
            else:
                logger.error(f"❌ Test data not found at {source_file}")
                return False

            # Verify data file exists and has content
            if source_file.exists() and source_file.stat().st_size > 0:
                logger.info(f"✅ Test data file exists with {source_file.stat().st_size} bytes")
                return True
            else:
                logger.error("❌ Test data file is empty or missing")
                return False

        except Exception as e:
            logger.error(f"❌ Error setting up test environment: {e}")
            return False

    def test_data_validation(self):
        """Test data validation functionality."""
        logger.info("Testing data validation...")

        try:
            # Initialize converter
            converter = Vn30ToQlibConverter(str(self.test_dir))

            # Test validation
            is_valid, error_msg = converter.validate_raw_data(self.test_symbol)

            if is_valid:
                logger.info("✅ Data validation passed")
                self.test_results['data_validation'] = True
                return True
            else:
                logger.error(f"❌ Data validation failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"❌ Data validation test failed: {e}")
            return False

    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        logger.info("Testing data cleaning...")

        try:
            # Initialize converter
            converter = Vn30ToQlibConverter(str(self.test_dir))

            # Test cleaning
            cleaned_df = converter.clean_raw_data(self.test_symbol)

            if cleaned_df is not None:
                logger.info(f"✅ Data cleaning successful: {len(cleaned_df)} records")

                # Verify cleaned data structure
                required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = set(required_columns) - set(cleaned_df.columns)

                if not missing_columns:
                    logger.info("✅ Cleaned data has all required columns")
                    self.test_results['data_cleaning'] = True
                    return True
                else:
                    logger.error(f"❌ Cleaned data missing columns: {missing_columns}")
                    return False
            else:
                logger.error("❌ Data cleaning returned None")
                return False

        except Exception as e:
            logger.error(f"❌ Data cleaning test failed: {e}")
            return False

    def test_qlib_format_preparation(self):
        """Test qlib format preparation."""
        logger.info("Testing qlib format preparation...")

        try:
            # Initialize converter
            converter = Vn30ToQlibConverter(str(self.test_dir))

            # Get cleaned data first
            cleaned_df = converter.clean_raw_data(self.test_symbol)
            if cleaned_df is None:
                logger.error("❌ Cannot test format preparation: cleaned data is None")
                return False

            # Test format preparation
            qlib_df = converter.prepare_qlib_format(self.test_symbol, cleaned_df)

            if qlib_df is not None:
                logger.info(f"✅ Qlib format preparation successful: {len(qlib_df)} records")

                # Verify qlib format structure
                expected_columns = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = set(expected_columns) - set(qlib_df.columns)

                if not missing_columns:
                    logger.info("✅ Qlib format has all required columns")

                    # Check data types
                    if pd.api.types.is_datetime64_any_dtype(qlib_df['time']):
                        logger.info("✅ Time column is datetime type")

                    # Check for numeric columns
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        if pd.api.types.is_numeric_dtype(qlib_df[col]):
                            logger.info(f"✅ {col} column is numeric type")

                    self.test_results['qlib_format_preparation'] = True
                    return True
                else:
                    logger.error(f"❌ Qlib format missing columns: {missing_columns}")
                    return False
            else:
                logger.error("❌ Qlib format preparation returned None")
                return False

        except Exception as e:
            logger.error(f"❌ Qlib format preparation test failed: {e}")
            return False

    def test_binary_conversion(self):
        """Test binary format conversion."""
        logger.info("Testing binary format conversion...")

        try:
            # Initialize converter
            converter = Vn30ToQlibConverter(str(self.test_dir))

            # Test conversion
            success = converter.convert_symbol_to_qlib(self.test_symbol)

            if success:
                logger.info("✅ Binary conversion successful")

                # Verify output files exist
                qlib_dir = self.test_dir / self.test_symbol / 'qlib'

                # Check if qlib directory was created
                if qlib_dir.exists():
                    logger.info(f"✅ Qlib directory created: {qlib_dir}")

                    # Look for output files (either .bin or .pkl depending on method used)
                    output_files = list(qlib_dir.glob("*.bin")) + list(qlib_dir.glob("*.pkl"))

                    if output_files:
                        logger.info(f"✅ Found output files: {[f.name for f in output_files]}")
                        self.test_results['binary_conversion'] = True
                        return True
                    else:
                        logger.error("❌ No output files found in qlib directory")
                        return False
                else:
                    logger.error("❌ Qlib directory was not created")
                    return False
            else:
                logger.error("❌ Binary conversion failed")
                return False

        except Exception as e:
            logger.error(f"❌ Binary conversion test failed: {e}")
            return False



    def test_complete_workflow(self):
        """Test complete workflow from raw data to qlib format."""
        logger.info("Testing complete workflow...")

        try:
            # Initialize converter
            converter = Vn30ToQlibConverter(str(self.test_dir))

            # Step 1: Validation
            is_valid, error_msg = converter.validate_raw_data(self.test_symbol)
            if not is_valid:
                logger.error(f"❌ Workflow failed at validation: {error_msg}")
                return False

            # Step 2: Cleaning
            cleaned_df = converter.clean_raw_data(self.test_symbol)
            if cleaned_df is None:
                logger.error("❌ Workflow failed at cleaning")
                return False

            # Step 3: Format preparation
            qlib_df = converter.prepare_qlib_format(self.test_symbol, cleaned_df)
            if qlib_df is None:
                logger.error("❌ Workflow failed at format preparation")
                return False

            # Step 4: Binary conversion
            success = converter.convert_symbol_to_qlib(self.test_symbol)
            if not success:
                logger.error("❌ Workflow failed at binary conversion")
                return False

            logger.info("✅ Complete workflow test passed")
            return True

        except Exception as e:
            logger.error(f"❌ Complete workflow test failed: {e}")
            return False

    def generate_integration_report(self):
        """Generate comprehensive integration test report."""
        logger.info("Generating integration test report...")

        try:
            # Get data statistics
            try:
                df = pd.read_csv(self.test_dir / self.test_symbol / 'raw' / 'historical_price.csv')
                data_stats = {
                    'total_records': len(df),
                    'date_range': f"{df['time'].min()} to {df['time'].max()}",
                    'columns': list(df.columns),
                    'missing_data': df.isnull().sum().sum()
                }
            except Exception:
                data_stats = {'error': 'Could not read data file'}

            report = f"""
VN30 to Qlib Integration Test Report

Test Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Test Symbol: {self.test_symbol}
Test Data Location: {self.test_dir / self.test_symbol}

Data Statistics:
- Total Records: {data_stats.get('total_records', 'N/A')}
- Date Range: {data_stats.get('date_range', 'N/A')}
- Columns: {data_stats.get('columns', 'N/A')}
- Missing Data: {data_stats.get('missing_data', 'N/A')}

Test Results Summary:
--------------------
Data Validation: {'PASS' if self.test_results['data_validation'] else 'FAIL'}
Data Cleaning: {'PASS' if self.test_results['data_cleaning'] else 'FAIL'}
Qlib Format Preparation: {'PASS' if self.test_results['qlib_format_preparation'] else 'FAIL'}
Binary Conversion: {'PASS' if self.test_results['binary_conversion'] else 'FAIL'}
Error Handling: {'PASS' if self.test_results['error_handling'] else 'FAIL'}

Overall Test Result: {'PASS' if all([v for k, v in self.test_results.items() if k != 'overall_success']) else 'FAIL'}

Test completed at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            print(report)
            logger.info("Integration test report generated successfully")

            return True

        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return False

    def run_integration_test(self):
        """Run the complete integration test."""
        logger.info("Starting VN30 to Qlib integration test...")

        try:
            # Setup test environment
            if not self.setup_test_environment():
                logger.error("❌ Test environment setup failed")
                return False

            # Run individual tests
            tests = [
                ('data_validation', self.test_data_validation),
                ('data_cleaning', self.test_data_cleaning),
                ('qlib_format_preparation', self.test_qlib_format_preparation),
                ('binary_conversion', self.test_binary_conversion),
                ('complete_workflow', self.test_complete_workflow)
            ]

            for test_name, test_func in tests:
                logger.info(f"Running {test_name}...")
                if not test_func():
                    logger.error(f"❌ {test_name} failed")
                    return False

            # Generate report
            if not self.generate_integration_report():
                logger.error("❌ Report generation failed")
                return False

            # Overall success
            self.test_results['overall_success'] = True

            logger.info("✅ VN30 to Qlib integration test completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False


def main():
    """Main function to run the integration test."""
    print("VN30 to Qlib Integration Test")
    print("=" * 50)

    # Create and run test
    test = TestVn30ToQlibIntegration()
    success = test.run_integration_test()

    if success:
        print("\n✅ VN30 to Qlib integration test PASSED!")
        return 0
    else:
        print("\n❌ VN30 to Qlib integration test FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
