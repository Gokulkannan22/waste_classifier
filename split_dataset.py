import os
import shutil
import random
from tqdm import tqdm

# Paths
base_dir = "C:/Users/gokul/OneDrive/Desktop/projects/waste_classifier/dataset"
classes = ["plastic", "paper", "metal", "E-waste", "organic"]
train_ratio = 0.8

# Create train/ and val/ folders
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# For each class
for cls in classes:
    class_dir = os.path.join(base_dir, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    split_idx = int(train_ratio * len(images))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Make class folders inside train/ and val/
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    print(f"\nProcessing class: {cls}")

    for img in tqdm(train_imgs, desc=f"Copying to train/{cls}"):
        shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, cls, img))

    for img in tqdm(val_imgs, desc=f"Copying to val/{cls}"):
        shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, cls, img))

print("\n✅ Dataset split complete!")
