#!/usr/bin/env python3
"""
refnet entry point.
"""

import sys
sys.path.insert(0, '.')

from refnet.cli import main

if __name__ == "__main__":
    sys.exit(main())
