# Cell 1: Imports and Setup
# This cell sets up the environment for plotting learning curves

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
API_DIR = Path("/Users/thefool/Local/Project/WebMining/Books/app/backend/api")
CHECKPOINTS_DIR = API_DIR / "checkpoints"

print(f"Checkpoints directory: {CHECKPOINTS_DIR}")
print(f"Available models: {[d.name for d in CHECKPOINTS_DIR.iterdir() if d.is_dir()]}")
