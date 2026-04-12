"""Launcher for the HuggingFace Space demo from project root."""

import os
import sys
from pathlib import Path

# Set up paths so hf_space imports work
root = Path(__file__).parent.parent
hf_space = root / "hf_space"
sys.path.insert(0, str(hf_space))
sys.path.insert(0, str(root))
os.chdir(root)

# Import and run
from hf_space.app import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
