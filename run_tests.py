#!/usr/bin/env python3
"""Run all abseil tests in the project."""

import subprocess
import sys


def main():
    """Run all test modules and exit with appropriate code."""
    test_modules = [
        "src.llm_kernel.data_models_test",
        "src.llm_kernel.scorer_test",
        "src.llm_kernel.utils_test",
        "src.llm_kernel.signatures.signatures_test",
    ]

    print("Running all tests...")
    print("=" * 50)

    passed = 0
    total = len(test_modules)

    for module in test_modules:
        print(f"\nRunning {module}...")
        print("-" * 30)

        try:
            result = subprocess.run(
                [sys.executable, "-m", module], capture_output=False
            )
            if result.returncode == 0:
                print(f"✅ {module} passed")
                passed += 1
            else:
                print(f"❌ {module} failed")
                return 1
        except Exception as e:
            print(f"❌ Error running {module}: {e}")
            return 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
