#!/usr/bin/env python3
"""
Test runner for GUI automated tests.

Runs all GUI tests and provides comprehensive test reporting.
"""

import sys
import unittest
import os
from pathlib import Path
import argparse

# Add paths for testing
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent))

# Set working directory
os.chdir(current_dir)


def discover_tests(test_dir=None, pattern="test_*.py"):
    """Discover all test files."""
    if test_dir is None:
        test_dir = Path(__file__).parent
    
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern=pattern)
    return suite


def run_tests(verbosity=2, failfast=False, pattern="test_*.py"):
    """Run all GUI tests."""
    print("=" * 70)
    print("FLOWLIB GUI AUTOMATED TEST SUITE")
    print("=" * 70)
    
    # Discover and run tests
    test_suite = discover_tests(pattern=pattern)
    
    # Configure test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=failfast,
        stream=sys.stdout
    )
    
    # Run tests
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    print("=" * 70)
    
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run GUI automated tests")
    parser.add_argument(
        "-v", "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Test output verbosity (0=quiet, 1=normal, 2=verbose)"
    )
    parser.add_argument(
        "-f", "--failfast",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="test_*.py",
        help="Test file pattern (default: test_*.py)"
    )
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Run only UI component tests"
    )
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests"
    )
    
    args = parser.parse_args()
    
    # Determine test pattern
    if args.ui_only:
        pattern = "test_ui_components.py"
    elif args.integration_only:
        pattern = "test_integration.py"
    else:
        pattern = args.pattern
    
    # Run tests
    success = run_tests(
        verbosity=args.verbosity,
        failfast=args.failfast,
        pattern=pattern
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())