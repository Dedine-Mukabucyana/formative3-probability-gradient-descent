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
