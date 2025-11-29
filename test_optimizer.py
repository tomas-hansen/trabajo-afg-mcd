# test_optimizer.py
# Script to test the optimizer with all available engines

import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from optimizer import main


def capture_output(func, *args, **kwargs):
    """Capture stdout and stderr from a function call."""
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            return f"Error: {e}", None
    return f.getvalue(), result


def main_test():
    engines = ["auto", "lstm_simple", "lstm_full",
               "xgb", "lgbm", "pls", "lgbm_full"]

    print("Testing optimizer with all engines...\n")

    for engine in engines:
        print(f"=== Testing engine: {engine} ===")
        output, _ = capture_output(main, engine)
        # Extract key lines
        lines = output.split('\n')
        for line in lines:
            if 'engine=' in line or 'Pred base' in line or 'Mejora' in line or 'Error' in line:
                print(line)
        print("-" * 50)


if __name__ == "__main__":
    main_test()
