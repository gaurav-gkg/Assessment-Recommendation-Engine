"""
conftest.py – pytest configuration
Ensures the project root is on sys.path so all imports work in tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
