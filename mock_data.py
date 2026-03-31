import os
import cv2
import numpy as np
import pandas as pd

def create_mock_dataset(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    img_dir = os.path.join(data_dir, "train_images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Create some dummy 224x224 images and a train.csv
    records = []
    
    # Create 50 dummy images
    for i in range(50):
        img_id = f"mock_{i}"
        diagnosis = np.random.randint(0, 5)
        
        # random colored image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, f"{img_id}.png"), img)
        
        records.append({"id_code": img_id, "diagnosis": diagnosis})
        
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    print("Mock dataset created in", data_dir)

if __name__ == "__main__":
    create_mock_dataset()
