#!/usr/bin/env python3
"""
OpenCog Cognitive Test Runner

Runs comprehensive tests for all OpenCog cognitive subsystems,
ensuring real implementations are validated with actual data.
"""

import sys
import unittest
import argparse
from pathlib import Path

def discover_and_run_tests(subsystem=None, verbose=False):
    """Discover and run tests for specified subsystem or all."""
    
    test_root = Path(__file__).parent
    
    if subsystem:
        test_dir = test_root / subsystem
        if not test_dir.exists():
            print(f"❌ Test directory for {subsystem} not found")
            return False
    else:
        test_dir = test_root
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"✅ All tests passed for {subsystem or 'all subsystems'}")
        return True
    else:
        print(f"❌ Some tests failed for {subsystem or 'all subsystems'}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run OpenCog cognitive tests')
    parser.add_argument('--subsystem', choices=['pln', 'moses', 'relex', 'atomspace', 'ecan'],
                       help='Run tests for specific subsystem only')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose test output')
    
    args = parser.parse_args()
    
    print("🧠 OpenCog Cognitive Test Runner")
    print("=" * 50)
    
    success = discover_and_run_tests(args.subsystem, args.verbose)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
