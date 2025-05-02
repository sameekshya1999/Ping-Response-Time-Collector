import os
import time
import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------- PART 1: Collect Ping Data --------------------
domains = ["google.com", "amazon.com", "openai.com"]
ping_data = []

print("🔄 Collecting ping data...")

for domain in domains:
    for i in range(10):  # ping each domain 10 times
        start = time.time()
        os.system(f"ping -n 1 {domain} > nul")
        end = time.time()
        latency = (end - start) * 1000  # ms
        print(f"{domain} ping {i+1}: {latency:.2f} ms")
        ping_data.append([domain, latency])

# Save to CSV
with open("ping_latency.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["domain", "latency_ms"])
    writer.writerows(ping_data)

print("✅ Ping data saved to ping_latency.csv")

# -------------------- PART 2: Load and Prepare Data --------------------
df = pd.read_csv("ping_latency.csv")

# Encode domain names
le = LabelEncoder()
df['domain_code'] = le.fit_transform(df['domain'])

# Add simulated features
df['simulated_load'] = [random.uniform(0.1, 1.0) for _ in range(len(df))]
df['packet_size'] = [random.randint(64, 1500) for _ in range(len(df))]

# Prepare input and output
X = df[['domain_code', 'simulated_load', 'packet_size']].values
y = df['latency_ms'].values

# -------------------- PART 3: Scale Features & Target --------------------
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# -------------------- PART 4: Build and Train Model --------------------
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print("🧠 Training model...")
model.fit(X_scaled, y_scaled, epochs=100, verbose=1)

# -------------------- PART 5: Predict and Plot --------------------
predictions_scaled = model.predict(X_scaled)
df['predicted_latency'] = scaler_y.inverse_transform(predictions_scaled).flatten()

plt.figure(figsize=(10, 6))
plt.scatter(df['domain'], df['latency_ms'], color='blue', label='Actual')
plt.scatter(df['domain'], df['predicted_latency'], color='red', label='Predicted')
plt.ylabel("Latency (ms)")
plt.title("Actual vs Predicted Latency")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
