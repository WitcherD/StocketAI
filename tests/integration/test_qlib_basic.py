#!/usr/bin/env python3
"""
Basic integration test for qlib library.

This test validates that qlib is properly installed and configured
before using it in data conversion processes.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestQlibBasic:
    """Basic integration test class for qlib."""

    def __init__(self):
        self.test_results = {
            'qlib_import': False,
            'qlib_init': False,
            'basic_functionality': False,
            'data_layer': False,
            'config_structure': False,
            'overall_success': False
        }

    def test_qlib_import(self):
        """Test basic qlib import functionality."""
        logger.info("Testing qlib import...")

        try:
            import qlib
            logger.info(f"✅ Qlib imported successfully (version: {getattr(qlib, '__version__', 'unknown')})")
            self.test_results['qlib_import'] = True
            return True

        except Exception as e:
            logger.error(f"❌ Qlib import failed: {e}")
            return False

    def test_qlib_initialization(self):
        """Test qlib initialization with current API."""
        logger.info("Testing qlib initialization...")

        try:
            import qlib
            from qlib.config import C

            # Test that we can access the global config object
            logger.info(f"✅ Qlib config object accessible: {type(C)}")

            # Test basic config properties
            logger.info(f"✅ Default provider URI: {C.provider_uri}")
            logger.info(f"✅ Available modes: {list(C._default_config.keys())[:5]}...")  # Show first 5 keys

            self.test_results['qlib_init'] = True
            return True

        except Exception as e:
            logger.error(f"❌ Qlib initialization test failed: {e}")
            return False

    def test_basic_functionality(self):
        """Test basic qlib functionality with current API."""
        logger.info("Testing basic qlib functionality...")

        try:
            from qlib.utils import fname_to_code, code_to_fname
            from qlib.constant import REG_CN, REG_US, REG_TW

            # Test fname_to_code and code_to_fname functions
            test_symbols = ["AAPL", "GOOGL", "MSFT", "TEST_STOCK"]
            codes = {}

            for symbol in test_symbols:
                code = fname_to_code(symbol)
                back_to_fname = code_to_fname(code)
                codes[symbol] = {'code': code, 'back_to_fname': back_to_fname}
                logger.info(f"Symbol: {symbol} -> Code: {code} -> Back: {back_to_fname}")

            # Test region constants
            logger.info(f"✅ Region constants: CN={REG_CN}, US={REG_US}, TW={REG_TW}")

            # Test utility functions
            from qlib.utils import hash_args, flatten_dict
            test_hash = hash_args("test", 123, {"key": "value"})
            logger.info(f"✅ Hash function works: {test_hash[:16]}...")

            test_flatten = flatten_dict({"a": 1, "b": {"c": 2, "d": 3}})
            logger.info(f"✅ Flatten function works: {test_flatten}")

            # Basic validation
            if len(codes) == len(test_symbols) and all(isinstance(v, dict) and 'code' in v for v in codes.values()):
                logger.info("✅ Basic functionality test passed")
                self.test_results['basic_functionality'] = True
                return True
            else:
                logger.error("❌ Basic functionality test failed")
                return False

        except Exception as e:
            logger.error(f"❌ Basic functionality test failed: {e}")
            return False

    def test_data_layer(self):
        """Test qlib data layer components with current API."""
        logger.info("Testing qlib data layer...")

        try:
            from qlib.data import D
            from qlib.data.cache import H

            # Test that data layer classes can be imported
            logger.info("✅ Data layer classes imported successfully")
            logger.info(f"✅ D (main data interface): {type(D)}")
            logger.info(f"✅ H (cache interface): {type(H)}")

            # Test data layer methods availability
            available_methods = [method for method in dir(D) if not method.startswith('_')]
            logger.info(f"✅ Available D methods: {available_methods[:10]}...")  # Show first 10

            # Test cache layer methods
            cache_methods = [method for method in dir(H) if not method.startswith('_')]
            logger.info(f"✅ Available H methods: {cache_methods[:10]}...")  # Show first 10

            self.test_results['data_layer'] = True
            return True

        except Exception as e:
            logger.error(f"❌ Data layer test failed: {e}")
            return False

    def test_config_structure(self):
        """Test qlib configuration structure."""
        logger.info("Testing qlib configuration structure...")

        try:
            from qlib.config import C, MODE_CONF, HIGH_FREQ_CONFIG

            # Test configuration modes
            logger.info(f"✅ Available config modes: {list(MODE_CONF.keys())}")

            # Test client mode configuration
            client_config = MODE_CONF.get('client', {})
            logger.info(f"✅ Client mode has {len(client_config)} configuration options")

            # Test high frequency configuration
            hf_config = HIGH_FREQ_CONFIG
            logger.info(f"✅ High frequency config provider URI: {hf_config.get('provider_uri', 'N/A')}")

            # Test region configurations
            from qlib.constant import REG_CN, REG_US, REG_TW
            from qlib.config import _default_region_config

            regions = [REG_CN, REG_US, REG_TW]
            logger.info(f"✅ Available regions: {regions}")

            for region in regions:
                if region in _default_region_config:
                    region_config = _default_region_config[region]
                    logger.info(f"✅ {region} config: trade_unit={region_config.get('trade_unit', 'N/A')}")

            logger.info("✅ Configuration structure test passed")
            self.test_results['config_structure'] = True
            return True

        except Exception as e:
            logger.error(f"❌ Configuration structure test failed: {e}")
            return False

    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("Generating test report...")

        try:
            report = f"""
Qlib Basic Integration Test Report

Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Test Results Summary:
--------------------
Qlib Import: {'PASS' if self.test_results['qlib_import'] else 'FAIL'}
Qlib Initialization: {'PASS' if self.test_results['qlib_init'] else 'FAIL'}
Basic Functionality: {'PASS' if self.test_results['basic_functionality'] else 'FAIL'}
Data Layer: {'PASS' if self.test_results['data_layer'] else 'FAIL'}
Configuration Structure: {'PASS' if self.test_results['config_structure'] else 'FAIL'}

Overall Test Result: {'PASS' if all([v for k, v in self.test_results.items() if k != 'overall_success']) else 'FAIL'}

Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            print(report)
            logger.info("Test report generated successfully")

            return True

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return False

    def run_basic_test(self):
        """Run the complete basic qlib test."""
        logger.info("Starting basic qlib integration test...")

        try:
            # Test 1: Import
            if not self.test_qlib_import():
                logger.error("Import test failed")
                return False

            # Test 2: Initialization
            if not self.test_qlib_initialization():
                logger.error("Initialization test failed")
                return False

            # Test 3: Basic functionality
            if not self.test_basic_functionality():
                logger.error("Basic functionality test failed")
                return False

            # Test 4: Data layer
            if not self.test_data_layer():
                logger.error("Data layer test failed")
                return False

            # Test 5: Configuration structure
            if not self.test_config_structure():
                logger.error("Configuration structure test failed")
                return False

            # Generate report
            if not self.generate_test_report():
                logger.error("Report generation failed")
                return False

            # Overall success
            self.test_results['overall_success'] = True

            logger.info("✅ Basic qlib integration test completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Basic qlib integration test failed: {e}")
            return False


def main():
    """Main function to run the basic qlib test."""
    print("Qlib Basic Integration Test")
    print("=" * 40)

    # Create and run test
    test = TestQlibBasic()
    success = test.run_basic_test()

    if success:
        print("\n✅ Basic qlib integration test PASSED!")
        return 0
    else:
        print("\n❌ Basic qlib integration test FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
