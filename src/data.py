<<<<<<< HEAD
﻿from pathlib import Path
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
=======
"""Part 1 data loading: Galton family heights."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_galton(path: str, parent: str = "mother"):
    """Return pooled heights, group labels and sub-group names.

    parent   : "mother" or "father" (the assignment allows either).
    heights  : one parent height per family followed by every child height.
    labels   : 0 = parent, 1 = child (kept aside, used only for evaluation).
    subgroup : "mother"/"father", "daughter" or "son" for each observation.
    """
    if parent not in ("mother", "father"):
        raise ValueError("parent must be 'mother' or 'father'")
    df = pd.read_csv(path)
    parents = df.drop_duplicates(subset="family")[parent].to_numpy(float)
    children = df["childHeight"].to_numpy(float)
    heights = np.concatenate([parents, children])
    labels = np.concatenate([np.zeros(len(parents), int), np.ones(len(children), int)])
    child_sex = df["gender"].map({"female": "daughter", "male": "son"}).to_numpy()
    subgroup = np.concatenate([np.full(len(parents), parent), child_sex])
    return heights, labels, subgroup
>>>>>>> origin/main
