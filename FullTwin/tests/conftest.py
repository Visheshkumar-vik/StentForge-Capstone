"""Make the FullTwin package and the reused project lanes importable in tests."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))          # FullTwin/
REPO = os.path.abspath(os.path.join(ROOT, ".."))          # Use_Capstone/
for p in (ROOT, os.path.join(REPO, "StenosisCFD"),
          os.path.join(REPO, "StentForge"),
          os.path.join(REPO, "StentForge", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
