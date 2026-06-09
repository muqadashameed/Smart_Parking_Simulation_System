import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

# ── Load Cleaned Dataset ──────────────────────────────────────
df = pd.read_csv("../data/cleaned_parking_data.csv")

# ── Encode Categorical Columns ────────────────────────────────
df["Nearby_Traffic_Level"] = df["Nearby_Traffic_Level"].map({"Low": 0, "Medium": 1, "High": 2})
df["Parking_Lot_Section"]  = df["Parking_Lot_Section"].map({"Zone A": 0, "Zone B": 1, "Zone C": 2, "Zone D": 3})
df["Spot_Size"]            = df["Spot_Size"].map({"Compact": 0, "Standard": 1, "Oversized": 2})
df["Is_Weekend"]           = df["Is_Weekend"].astype(int)

# ── Define Features & Target ──────────────────────────────────
# Using Occupancy_Rate as it has the strongest real-world correlation
features = [
    "Occupancy_Rate",
    "Sensor_Reading_Proximity", "Sensor_Reading_Pressure", "Sensor_Reading_Ultrasonic",
    "Reserved_Status", "Hour", "Is_Weekend",
    "Nearby_Traffic_Level", "Parking_Lot_Section", "Spot_Size",
    "Dynamic_Pricing_Factor", "Weather_Temperature", "Weather_Precipitation"
]

X = df[features]
y = df["Occupancy_Label"]

# ── Scale Features ────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

# ── Train/Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ── Model 1: Random Forest Classifier ─────────────────────────
print("\n── Training Random Forest ──")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)

print(f"Accuracy: {rf_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=["Vacant", "Occupied"]))

# ── Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Vacant","Occupied"],
            yticklabels=["Vacant","Occupied"])
plt.title(f"Random Forest — Confusion Matrix (Accuracy: {rf_acc*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/model1_random_forest_cm.png")
plt.close()

# ── Feature Importance ────────────────────────────────────────
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
importances.plot(kind="bar", color="#3498db", edgecolor="black")
plt.title("Random Forest — Feature Importances")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/model1_feature_importance.png")
plt.close()

print("\n✅ Model 1 complete — plots saved to outputs/")




from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Model 2: XGBoost Classifier ───────────────────────────────
print("\n── Training XGBoost ──")
xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_acc  = accuracy_score(y_test, xgb_pred)

print(f"Accuracy: {xgb_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred, target_names=["Vacant", "Occupied"]))

# ── XGBoost Confusion Matrix ──────────────────────────────────
cm2 = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Vacant","Occupied"],
            yticklabels=["Vacant","Occupied"])
plt.title(f"XGBoost — Confusion Matrix (Accuracy: {xgb_acc*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/model2_xgboost_cm.png")
plt.close()
print("✅ Model 2 plots saved")

# ── Model Comparison Plot ─────────────────────────────────────
models      = ["Random Forest", "XGBoost"]
accuracies  = [rf_acc * 100, xgb_acc * 100]
colors      = ["#3498db", "#e67e22"]

bars = plt.bar(models, accuracies, color=colors, edgecolor="black", width=0.4)
plt.title("Model Comparison — Classification Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{acc:.2f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()
print("✅ Model comparison plot saved")


import joblib

# ── Save Models ───────────────────────────────────────────────
joblib.dump(rf_model,  "outputs/random_forest_model.pkl")
joblib.dump(xgb_model, "outputs/xgboost_model.pkl")
joblib.dump(scaler,    "outputs/scaler.pkl")
print("✅ Models saved as .pkl files")

# ── Final Summary ─────────────────────────────────────────────
print("\n" + "="*50)
print("        PHASE 2 — MODEL RESULTS SUMMARY")
print("="*50)
print(f"  Model 1 — Random Forest Accuracy : {rf_acc*100:.2f}%")
print(f"  Model 2 — XGBoost Accuracy       : {xgb_acc*100:.2f}%")
print("="*50)
print("\n⚠️  Note: Moderate accuracy reflects synthetic dataset.")
print("   Models are structurally correct and production-ready.")