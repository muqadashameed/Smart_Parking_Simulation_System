import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv("../data/IIoT_Smart_Parking_Management.csv")
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


# ── Step 2: Feature Engineering ──────────────────────────────

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Extract useful time features
df["Hour"]       = df["Timestamp"].dt.hour
df["Day"]        = df["Timestamp"].dt.day_name()
df["Month"]      = df["Timestamp"].dt.month
df["Is_Weekend"] = df["Timestamp"].dt.dayofweek >= 5  # Saturday=5, Sunday=6

print("New columns added: Hour, Day, Month, Is_Weekend")
print(df[["Timestamp", "Hour", "Day", "Month", "Is_Weekend"]].head(10))

# ── Step 3: Univariate Analysis ───────────────────────────────

# Check unique values in categorical columns
cat_cols = ["Occupancy_Status", "Vehicle_Type", "User_Type",
            "Parking_Lot_Section", "Spot_Size", "Payment_Status"]

print("\n── Categorical Column Value Counts ──")
for col in cat_cols:
    print(f"\n{col}:\n{df[col].value_counts()}")



# ── Step 3: Visualizations ────────────────────────────────────
import os
os.makedirs("plots", exist_ok=True)  # creates notebooks/plots/ folder

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ── Plot 1: Occupancy Status Distribution ──
ax = df["Occupancy_Status"].value_counts().plot(kind="bar", color=["#e74c3c","#2ecc71"], edgecolor="black")
plt.title("Occupancy Status Distribution")
plt.xlabel("Status")
plt.ylabel("Count")
plt.xticks(rotation=0)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height() + 5), ha="center")
plt.tight_layout()
plt.savefig("plots/01_occupancy_status.png")
plt.close()
print("Saved: 01_occupancy_status.png")

# ── Plot 2: Vehicle Type Distribution ──
df["Vehicle_Type"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90,
                                        colors=["#3498db","#e67e22","#9b59b6"])
plt.title("Vehicle Type Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("plots/02_vehicle_type.png")
plt.close()
print("Saved: 02_vehicle_type.png")

# ── Plot 3: Parking Lot Section Usage ──
ax = df["Parking_Lot_Section"].value_counts().plot(kind="bar", color="#3498db", edgecolor="black")
plt.title("Parking Lot Section Usage")
plt.xlabel("Zone")
plt.ylabel("Count")
plt.xticks(rotation=0)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height() + 3), ha="center")
plt.tight_layout()
plt.savefig("plots/03_section_usage.png")
plt.close()
print("Saved: 03_section_usage.png")

# ── Plot 4: Hourly Occupancy (Peak Hours) ──
hourly = df.groupby("Hour")["Occupancy_Status"].apply(
    lambda x: (x == "Occupied").sum()
)
hourly.plot(kind="line", marker="o", color="#e74c3c", linewidth=2)
plt.title("Occupied Spots by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Number of Occupied Spots")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("plots/04_peak_hours.png")
plt.close()
print("Saved: 04_peak_hours.png")

# ── Plot 5: Day of Week Occupancy ──
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
daily = df.groupby("Day")["Occupancy_Status"].apply(
    lambda x: (x == "Occupied").sum()
).reindex(day_order)
daily.plot(kind="bar", color="#9b59b6", edgecolor="black")
plt.title("Occupied Spots by Day of Week")
plt.xlabel("Day")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/05_day_occupancy.png")
plt.close()
print("Saved: 05_day_occupancy.png")

# ── Plot 6: Payment Status ──
df["Payment_Status"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90,
                                          colors=["#2ecc71","#e74c3c","#f39c12"])
plt.title("Payment Status Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("plots/06_payment_status.png")
plt.close()
print("Saved: 06_payment_status.png")

# ── Plot 7: Parking Duration Distribution ──
df["Parking_Duration"].plot(kind="hist", bins=30, color="#1abc9c", edgecolor="black")
plt.title("Parking Duration Distribution")
plt.xlabel("Duration")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/07_parking_duration.png")
plt.close()
print("Saved: 07_parking_duration.png")

# ── Plot 8: Correlation Heatmap ──
plt.figure(figsize=(12, 8))
num_cols = df.select_dtypes(include=np.number).columns
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/08_correlation_heatmap.png")
plt.close()
print("Saved: 08_correlation_heatmap.png")

print("\n✅ All 8 plots saved in notebooks/plots/")    


# ── Step 4: Save Clean Dataset ────────────────────────────────

# Encode Occupancy_Status as binary (for ML later)
df["Occupancy_Label"] = df["Occupancy_Status"].map({"Occupied": 1, "Vacant": 0})

# Save cleaned dataset
df.to_csv("../data/cleaned_parking_data.csv", index=False)
print("✅ Cleaned dataset saved to data/cleaned_parking_data.csv")
print(f"   Shape: {df.shape}")
print(f"   New columns: Hour, Day, Month, Is_Weekend, Occupancy_Label")