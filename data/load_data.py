import os
import shutil
from sklearn.model_selection import train_test_split

def download_dog_dataset(dest_dir="data/all-dogs"):
    zip_filename = "all-dogs.zip"
    
    print("Downloading dataset...")
    os.system("gdown 1KXRTB_q4uub_XOHecpsQjE4Kmv76sZbV")
    os.system(f"unzip -q {zip_filename}")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    extracted_dir = "all-dogs"
    if os.path.exists(extracted_dir):
        for file in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, file)
            dst = os.path.join(dest_dir, file)
            if os.path.isfile(src):
                shutil.move(src, dst)
        shutil.rmtree(extracted_dir)
    
    os.remove(zip_filename)
    print("Download and extraction complete.")

def split_dataset(source_dir, train_dir, test_dir, test_size=0.2, random_state=42):
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

    print(f"Train files: {len(train_files)} | Test files: {len(test_files)}")

def prepare_dog_dataset(source_dir="data/all-dogs"):
    train_dir = os.path.join(source_dir, "train", "dogs")
    test_dir = os.path.join(source_dir, "test", "dogs")

    if not os.path.exists(source_dir) or not os.listdir(source_dir):
        print("Dog dataset not found, downloading...")
        download_dog_dataset(source_dir)
    else:
        print(f"Found dataset in '{source_dir}' with {len(os.listdir(source_dir))} files.")

    print("Splitting dataset into train and test sets...")
    split_dataset(source_dir, train_dir, test_dir)
    print("Dataset preparation complete.")
