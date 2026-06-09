import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv("data/smart_parking.csv")  # adjust filename if different

# ── Basic Info ────────────────────────────────────────────────
print("Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())

# ── Missing Values ────────────────────────────────────────────
print("\nMissing Values:")
print(df.isnull().sum())

# ── Duplicates ────────────────────────────────────────────────
print("\nDuplicate Rows:", df.duplicated().sum())

# ── Basic Stats ───────────────────────────────────────────────
print("\nNumerical Summary:")
print(df.describe())