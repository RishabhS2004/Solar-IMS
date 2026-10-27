"""
tests/run_tests.py
==================
Standalone test runner for Solar IMS test suite.
Runs all test functions across test_simulation, test_diagnostics,
test_recommendations, test_data_loader, and test_integration.
"""

import sys
import os
import inspect
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Import test modules
import tests.test_simulation as t_sim
import tests.test_diagnostics as t_diag
import tests.test_recommendations as t_rec
import tests.test_data_loader as t_data
import tests.test_integration as t_integ

def run_module_tests(module):
    """Find and run all functions starting with test_ in the module."""
    test_funcs = [
        obj for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj) and name.startswith("test_")
    ]
    
    passed = 0
    failed = 0
    errors = []

    print(f"\n[MODULE] Running {module.__name__} ({len(test_funcs)} tests):")
    for func in test_funcs:
        test_name = func.__name__
        try:
            func()
            print(f"  [PASS] {test_name}")
            passed += 1
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  [FAIL] {test_name}: {e}\n{tb}")
            failed += 1
            errors.append((test_name, e))

    return passed, failed, errors

def main():
    modules = [t_sim, t_diag, t_rec, t_data, t_integ]
    total_passed = 0
    total_failed = 0
    all_errors = []

    print("=" * 70)
    print("SOLAR IMS -- AUTOMATED TEST SUITE EXECUTION")
    print("=" * 70)

    for m in modules:
        p, f, errs = run_module_tests(m)
        total_passed += p
        total_failed += f
        all_errors.extend(errs)

    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY:")
    print(f"Total Tests Run: {total_passed + total_failed}")
    print(f"Passed:          {total_passed}")
    print(f"Failed:          {total_failed}")
    print("=" * 70)

    if total_failed > 0:
        print("\nFailures:")
        for name, err in all_errors:
            print(f"- {name}: {err}")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)

if __name__ == "__main__":
    main()
