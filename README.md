# ⚙️ Mini Anomaly Detector

**Goal:** Demonstrate infrastructure monitoring and reliability awareness by detecting anomalies in server performance data using Python.

This project simulates server load metrics, detects abnormal patterns (anomalies) using Z-scores, and visualizes them to highlight system reliability insights .

---

## 🚀 Project Overview

- **Dataset:** 100 synthetic hourly data points representing server load (%) from **March 2025**.  
- **Anomaly Detection Logic:** Z-score method (`|z| > 2.5` threshold).  
- **Visualization:** Line chart showing normal vs. anomalous points.  
- **Output:** Results saved in Excel (`Anomaly_Detector_Results.xlsx`) and visualized using Matplotlib.

---

## 📊 Sample Output Preview

| Timestamp           | Server_Load(%) | Z_Score | Anomaly |
|---------------------|----------------|----------|----------|
| 2025-03-01 00:00:00 | 62.48          | 0.38     | No       |
| 2025-03-01 01:00:00 | 59.31          | -0.25    | No       |
| 2025-03-01 20:00:00 | 90.00          | 5.65     | Yes      |
| 2025-03-04 03:00:00 | 30.00          | -5.95    | Yes      |

---

## 🧠 Tech Stack

- **Python 3**
- **Pandas** – data generation & processing  
- **NumPy** – statistical calculations  
- **Matplotlib** – visualization  
- **Excel (xlsxwriter)** – export results  

---

## 🧩 Step 1: Generate Dataset and Detect Anomalies

```python
# =============================================
# Project: Mini Anomaly Detector
# File: Anomaly_Detector_Results.xlsx
# =============================================

import pandas as pd
import numpy as np

# ---------- Step 1: Generate Synthetic Data ----------
np.random.seed(42)  # ensures reproducibility

# Create 100 hourly timestamps starting from March 1, 2025
time_stamps = pd.date_range(start="2025-03-01", periods=100, freq="H")

# Generate server load data (normally distributed around 60% utilization)
server_load = np.random.normal(loc=60, scale=5, size=100)

# ---------- Step 2: Inject a few anomalies ----------
server_load[20] = 90   # spike anomaly
server_load[75] = 30   # dip anomaly
server_load[88] = 95   # extreme spike anomaly

# ---------- Step 3: Compute Z-scores ----------
mean_load = server_load.mean()
std_load = server_load.std()
z_scores = (server_load - mean_load) / std_load

# Label anomalies if z-score exceeds ±2.5
anomalies = ["Yes" if abs(z) > 2.5 else "No" for z in z_scores]

# ---------- Step 4: Create a DataFrame ----------
df = pd.DataFrame({
    "Timestamp": time_stamps,
    "Server_Load(%)": np.round(server_load, 2),
    "Z_Score": np.round(z_scores, 2),
    "Anomaly": anomalies
})

# ---------- Step 5: Save Results ----------
output_path = "/home/nusrat/Desktop/Spreadsheet/Anomaly_Detector_Results.xlsx"
df.to_excel(output_path, index=False)

print("✅ Anomaly Detector dataset created successfully!")
print(f"📁 Saved at: {output_path}")

# ---------- Step 6: Optional Preview ----------
print(df.head(10))
```
---
## 📈 Step 2: Visualize the Anomalies
```python

import pandas as pd
import matplotlib.pyplot as plt

# ---------- Step 1: Load the dataset ----------
file_path = "/home/nusrat/Desktop/Spreadsheet/Anomaly_Detector_Results.xlsx"
df = pd.read_excel(file_path)

# ---------- Step 2: Plot the results ----------
plt.figure(figsize=(12, 6))
plt.plot(df["Timestamp"], df["Server_Load(%)"], label="Server Load", color="blue", linewidth=1.8)

# Highlight anomalies
anomalies = df[df["Anomaly"] == "Yes"]
plt.scatter(anomalies["Timestamp"], anomalies["Server_Load(%)"], color="red", label="Anomaly", s=80, marker="o")

# ---------- Step 3: Customize chart ----------
plt.title("Server Load Anomaly Detection (Mar 2025)", fontsize=14, weight="bold")
plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Server Load (%)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# ---------- Step 4: Show the plot ----------
plt.show()

```
---

## 🧾 How to Run
1. Clone this repository:  
   ```bash
   git clone <your-repository-url>
```
2. Ensure you have the required libraries:
   ```bash
     pip install pandas numpy matplotlib openpyxl
```
3. Run the Python script to generate the dataset and visualize anomalies:
   ```bash
     python Anomaly_Detector.py
   ```
4. The dataset will be saved at:  
`Desktop/Spreadsheet/Anomaly_Detector_Results.xlsx`
---

## 🔍 Key Insights

- 3 anomalies detected out of 100 total observations.  
- Visual patterns clearly show unusual spikes and dips in server load.  
- Demonstrates proactive monitoring of system reliability and performance.
---

