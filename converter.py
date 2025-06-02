import pandas as pd
from PIL import Image


import pandas as pd
'''
# Read your CSV
df = pd.read_csv('testing_landmarks.csv')

# Original columns example (fix yours if needed)
original_cols = df.columns.tolist()
print(original_cols)

# Build new columns with correct pattern
new_cols = ['image']
for i in range(68):
    new_cols.append(f'x{i}')
    new_cols.append(f'y{i}')

# Rename columns in df
df.columns = new_cols

# Save fixed CSV
df.to_csv('testing_landmarks.csv', index=False)
'''


def normalize_landmarks(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    
    normalized_data = []
    
    for idx, row in df.iterrows():
        img_path = row['image']
        img = Image.open(img_path)
        w, h = img.size
        
        landmarks = row[1:].values.astype('float32')
        # Normalize x and y coords
        landmarks[0::2] /= w  # x coords
        landmarks[1::2] /= h  # y coords
        
        normalized_data.append([img_path] + landmarks.tolist())
    
    cols = df.columns.tolist()
    normalized_df = pd.DataFrame(normalized_data, columns=cols)
    normalized_df.to_csv(csv_out, index=False)
    print(f"Saved normalized landmarks to {csv_out}")

# Usage:
normalize_landmarks("training_landmarks.csv", "training_landmarks_norm.csv")
normalize_landmarks("testing_landmarks.csv", "testing_landmarks_norm.csv")
