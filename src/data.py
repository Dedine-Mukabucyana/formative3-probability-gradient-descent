from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "GaltonFamilies.csv"


def load_data(file_path=DATA_FILE):
    """Load the Galton Families dataset."""
    return pd.read_csv(file_path)


def prepare_heights(data):
    """Return mothers, children, and combined height observations."""
    mothers = data.drop_duplicates(subset="family")["mother"].to_numpy(dtype=float)
    children = data["childHeight"].to_numpy(dtype=float)
    heights = np.concatenate([mothers, children])
    return mothers, children, heights
