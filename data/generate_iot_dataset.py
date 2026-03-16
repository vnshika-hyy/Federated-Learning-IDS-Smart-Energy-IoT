import pandas as pd
import numpy as np
import random
import os

rows = 12000

devices = [f"meter_{i}" for i in range(50)]
protocols = ["MQTT", "CoAP", "HTTP"]

data = []

for i in range(rows):

    device = random.choice(devices)
    protocol = random.choice(protocols)

    sensor1 = np.random.normal(50, 10)
    sensor2 = np.random.normal(30, 5)

    latency = abs(np.random.normal(100, 40))
    packet_loss = abs(np.random.normal(1, 0.5))

    attack_prob = random.random()

    if attack_prob < 0.2:

        label = "Attack"

        latency = latency + random.uniform(200, 500)
        packet_loss = packet_loss + random.uniform(3, 8)

    else:

        label = "Normal"

    data.append([
        i,
        device,
        sensor1,
        sensor2,
        latency,
        packet_loss,
        protocol,
        label
    ])

df = pd.DataFrame(data, columns=[
    "timestamp",
    "device_id",
    "sensor_1",
    "sensor_2",
    "latency",
    "packet_loss",
    "protocol",
    "label"
])

os.makedirs("data/raw", exist_ok=True)

df.to_csv("data/raw/smart_energy_iot_attack_data.csv", index=False)

print("Dataset generated")
print(df.shape)