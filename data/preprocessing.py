import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

print("Starting preprocessing pipeline...")

files = [f for f in os.listdir(RAW_DIR) if f.endswith(".parquet")]

dfs = []

for f in files:
    print("Loading:", f)
    df = pd.read_parquet(os.path.join(RAW_DIR, f))
    dfs.append(df)

data = pd.concat(dfs)

print("Dataset loaded:", data.shape)

# clean data
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

print("After cleaning:", data.shape)

# encode label
data["Label"] = data["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

X = data.drop("Label", axis=1)
y = data["Label"]

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

os.makedirs(PROCESSED_DIR, exist_ok=True)

np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

print("Preprocessing complete")
print("Files saved in data/processed")