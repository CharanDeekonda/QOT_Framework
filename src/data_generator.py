import pandas as pd
import numpy as np
import random
import os

NUM_UNIQUE_LINKS = 50      
NUM_SAMPLES = 5000          
MIN_HOPS = 3                
MAX_HOPS = 10               
OUTPUT_PATH = "data/raw/network_data.csv"

link_registry = {}
for i in range(NUM_UNIQUE_LINKS):
    link_id = i
    length = np.random.randint(40, 100) 
    quality_factor = np.random.uniform(0.8, 2.5) 
    
    link_registry[link_id] = {
        "length": length,
        "quality_factor": quality_factor
    }

print(f"✅ Created Network Topology with {NUM_UNIQUE_LINKS} unique links.")

data = []

for _ in range(NUM_SAMPLES):
    num_hops = np.random.randint(MIN_HOPS, MAX_HOPS + 1)
    path_sequence = random.sample(range(NUM_UNIQUE_LINKS), num_hops)
    total_length = 0
    total_noise = 0
    
    for link_id in path_sequence:
        props = link_registry[link_id]
        total_length += props["length"]
        noise_contribution = (props["length"] * props["quality_factor"] * 2.5) + np.random.normal(0, 5)
        total_noise += noise_contribution
    base_signal = 35.0 
    calculated_gsnr = base_signal - (total_noise / 100)
    calculated_gsnr = max(5.0, min(30.0, calculated_gsnr))

    path_string = "-".join(map(str, path_sequence))

    data.append({
        "path_sequence": path_string,
        "num_hops": num_hops,
        "total_length_km": total_length,
        "GSNR_dB": round(calculated_gsnr, 2) 
    })

df = pd.DataFrame(data)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Dataset generated successfully!")
print(f"   - Saved to: {OUTPUT_PATH}")
print(f"   - Total Samples: {len(df)}")
print(f"   - Columns: {list(df.columns)}")
print("\nSample Data:")
print(df.head(3))