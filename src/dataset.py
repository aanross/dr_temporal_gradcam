import os
import subprocess
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold

def download_aptos2019(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    # Check if data already exists
    if os.path.exists(os.path.join(data_dir, "train.csv")) and os.path.exists(os.path.join(data_dir, "train_images")):
        print("Data already downloaded.")
        return
        
    zip_path = os.path.join(data_dir, "aptos2019-blindness-detection.zip")
    
    # If the zip is already present, just extract it
    if os.path.exists(zip_path):
        print(f"Found {zip_path}, extracting...")
        subprocess.run(["unzip", "-q", zip_path, "-d", data_dir], check=True)
        os.remove(zip_path)
        print("Extraction complete.")
        return
    
    print("Downloading APTOS 2019 dataset via Kaggle API...")
    try:
        subprocess.run(["kaggle", "competitions", "download", "-c", "aptos2019-blindness-detection", "-p", data_dir], check=True)
        if os.path.exists(zip_path):
            subprocess.run(["unzip", "-q", zip_path, "-d", data_dir], check=True)
            os.remove(zip_path)
            print("Download and extraction complete.")
    except Exception as e:
        print(f"Failed to download dataset: {e}. Ensure Kaggle API is set up properly.")


def generate_synthetic_sequences(csv_path, num_patients=1000, seed=42):
    """
    Creates synthetic longitudinal data from the static APTOS classification dataset.
    We'll build sequences of 3-4 visits per patient, simulating disease progression.
    """
    np.random.seed(seed)
    df = pd.read_csv(csv_path)
    
    # We group pool of images by diagnosis
    pool_by_dx = {dx: df[df['diagnosis'] == dx]['id_code'].tolist() for dx in range(5)}
    
    patients = []
    
    for p_id in range(num_patients):
        seq_len = np.random.choice([3, 4])
        # Decide if this patient progresses
        progresses = np.random.choice([True, False], p=[0.4, 0.6])
        
        if progresses:
            # E.g., dx path: 0 -> 1 -> 2
            start_dx = np.random.randint(0, 3)
            end_dx = np.random.randint(start_dx + 1, 5)
            dx_path = np.linspace(start_dx, end_dx, seq_len).round().astype(int)
        else:
            # Stable dx
            stable_dx = np.random.randint(0, 5)
            dx_path = np.array([stable_dx] * seq_len)
            
        visit_intervals = np.random.randint(3, 13, size=seq_len-1) # 3 to 12 months
        times = [0] + np.cumsum(visit_intervals).tolist()
        
        sequence_images = []
        for dx in dx_path:
            # random sample from the pool
            if len(pool_by_dx[dx]) > 0:
                img_id = np.random.choice(pool_by_dx[dx])
                sequence_images.append(img_id)
            else:
                # fallback if pool exhausted/empty
                sequence_images.append(None)
                
        # Progression target: will they worsen in 12 months after the LAST observed visit minus 1?
        # Actually, let's just define progression prediction: Does diagnosis[last] > diagnosis[0]?
        progression_label = 1 if dx_path[-1] > dx_path[0] else 0
        
        patients.append({
            'patient_id': p_id,
            'image_ids': sequence_images,
            'diagnoses': dx_path.tolist(),
            'visit_months': times,
            'progression_label': progression_label,
            'final_diagnosis': dx_path[-1]
        })
        
    return patients


class SyntheticLesionGenerator:
    """
    Generates dummy 'lesion masks' based on DR severity for GradCAM IoU evaluation.
    This simulates microaneurysms (dots), hemorrhages (larger spots), and exudates (irregular blob).
    """
    def __init__(self, img_size=(224, 224), seed=None):
        self.H, self.W = img_size
        if seed:
            np.random.seed(seed)

    def __call__(self, diagnosis):
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        
        if diagnosis == 0:
            return mask
            
        # Draw random circles roughly matching DR features based on severity
        num_mAs = diagnosis * np.random.randint(3, 8)
        for _ in range(num_mAs):
            cx, cy = np.random.randint(20, self.W-20), np.random.randint(20, self.H-20)
            r = np.random.randint(1, 4)
            cv2.circle(mask, (cx, cy), r, 1, -1)
            
        if diagnosis >= 2:
            num_exudates = (diagnosis - 1) * np.random.randint(2, 5)
            for _ in range(num_exudates):
                cx, cy = np.random.randint(30, self.W-30), np.random.randint(30, self.H-30)
                r = np.random.randint(4, 10)
                cv2.circle(mask, (cx, cy), r, 1, -1)
                
        if diagnosis >= 3:
            num_hemorrhages = (diagnosis - 2) * np.random.randint(2, 4)
            for _ in range(num_hemorrhages):
                pts = np.random.randint(20, self.W-20, size=(np.random.randint(4, 7), 2))
                cv2.fillPoly(mask, [pts], 1)
                
        return mask


class DRTemporalDataset(Dataset):
    def __init__(self, patients, img_dir, transform=None, max_seq_len=4, img_size=(224, 224)):
        self.patients = patients
        self.img_dir = img_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.lesion_gen = SyntheticLesionGenerator(img_size)

    def __len__(self):
        return len(self.patients)
        
    def __getitem__(self, idx):
        p_data = self.patients[idx]
        image_ids = p_data['image_ids']
        diagnoses = p_data['diagnoses']
        
        seq_len = len(image_ids)
        
        # We will pad to max_seq_len with zeros
        # Return shapes:
        # Padded Images: (max_seq_len, C, H, W)
        # Masks: (max_seq_len, 1, H, W)
        # Mask length / valid frames array
        
        img_list = []
        mask_list = []
        
        for i_id, dx in zip(image_ids, diagnoses):
            img_path = os.path.join(self.img_dir, f"{i_id}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
            else:
                # fallback black image
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                
            mask = self.lesion_gen(dx)
            
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            else:
                img = ToTensorV2()(image=img)['image'].float() / 255.0
                mask = torch.tensor(mask, dtype=torch.float32)

            img_list.append(img)
            mask_list.append(mask.unsqueeze(0))
            
        # Optional padding
        C, H, W = img_list[0].shape
        padded_imgs = torch.zeros((self.max_seq_len, C, H, W))
        padded_masks = torch.zeros((self.max_seq_len, 1, H, W))
        dx_padded = torch.zeros(self.max_seq_len, dtype=torch.long)
        
        for i, (img, mask, dx) in enumerate(zip(img_list, mask_list, diagnoses)):
            padded_imgs[i] = img
            padded_masks[i] = mask
            dx_padded[i] = dx
            
        return {
            'pixel_values': padded_imgs,         # (T, 3, H, W)
            'lesion_masks': padded_masks,        # (T, 1, H, W)
            'diagnoses': dx_padded,              # (T,)
            'seq_length': torch.tensor(seq_len), 
            'final_diagnosis': torch.tensor(p_data['final_diagnosis'], dtype=torch.long),
            'progression': torch.tensor(p_data['progression_label'], dtype=torch.float32)
        }


class DRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=4, num_workers=4, num_patients=1000, fold=0, num_folds=5, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "train.csv")
        self.img_dir = os.path.join(data_dir, "train_images")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_patients = num_patients
        self.fold = fold
        self.num_folds = num_folds
        self.seed = seed
        self.img_size = 224
        
    def prepare_data(self):
        download_aptos2019(self.data_dir)
        
    def setup(self, stage=None):
        # We generate synthetic longitudinal data once
        patients = generate_synthetic_sequences(self.csv_path, num_patients=self.num_patients, seed=self.seed)
        
        # 5-fold CV split
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        splits = list(kf.split(patients))
        train_idxs, val_idxs = splits[self.fold]
        
        self.train_patients = [patients[i] for i in train_idxs]
        self.val_patients = [patients[i] for i in val_idxs]
        
        self.train_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.train_dataset = DRTemporalDataset(self.train_patients, self.img_dir, transform=self.train_transform)
        self.val_dataset = DRTemporalDataset(self.val_patients, self.img_dir, transform=self.val_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
