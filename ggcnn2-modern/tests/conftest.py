"""Conftest for ggcnn2-modern tests — make sure the package is importable."""
import sys
import os

# Allow `import ggcnn2.*` without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
