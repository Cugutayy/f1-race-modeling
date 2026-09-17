"""Compatibility entry point. Run --help for the explicit research workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from f1_research.cli import main

if __name__ == "__main__":
    main()
