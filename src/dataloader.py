import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from minicv.io import read_image, gray_to_rgb
from minicv.transforms import resize
from minicv.utils.normalize import normalize

def preprocess_image(image_path:str, target_size:tuple):
    # Read the image from defined path
    image = read_image(image_path)
    h,w = target_size
    # Check if the image is 2D (grayscale)
    if image.ndim ==2:
        image = gray_to_rgb(image)

    # Resize image to the target height and width and use bilinear interpolation for smoother results.
    resized_image = resize(image, h , w , "bilinear")

    # Normalization was applied using Min-Max scaling
    normalized_image = normalize(resized_image , "minmax")

    return normalized_image

def process_dataset_split(csv_filename:str, target_size:tuple, output_prefix: str):

    # Read the CSV
    df = pd.read_csv(csv_filename)

    # Create two empty lists: X (for the image arrays) and Y (for the integer labels).

    X_list = []
    Y_list = []

    total_images = len(df)

    # Loop through the rows. itertuples() is slightly faster than iterrows()
    for row in df.itertuples():
        # row.image_path accesses the column named 'image_path'
        try:
            # Try to process the image normally
            img_array = preprocess_image(row.image_path, target_size)
            X_list.append(img_array)
            Y_list.append(row.label)
            
        except ValueError as e:
            # If minicv throws an error (like the constant intensity one), catch it and skip
            print(f"\n[WARNING] Skipping corrupt image {row.image_path}: {e}")
            continue

        if row.Index % 100 == 0 and row.Index > 0:
            print(f"Processed {row.Index} / {total_images} images...")
        
    print("Converting lists to NumPy arrays (this might take a moment)...")
    
    X = np.array(X_list)
    Y = np.array(Y_list)

    print(f"Finished {output_prefix} split. Final shape: {X.shape}")

    # Disk Caching: Save these massive arrays to your hard drive
    # This way, you only ever run this slow function mra.

    np.save(f"X_{output_prefix}.npy",X)
    np.save(f"Y_{output_prefix}.npy",Y)
    print(f"Saved to X_{output_prefix}.npy and Y_{output_prefix}.npy\n")

    return X, Y

def main():
    target_size = (128, 128)

    # Process the Training Set
    X_train, Y_train = process_dataset_split("train_annotations.csv", target_size, "train")

    # Process the Validation Set
    X_val, Y_val = process_dataset_split("val_annotations.csv", target_size, "val")

    # Process the Testing Set
    X_test, Y_test = process_dataset_split("test_annotations.csv", target_size, "test")

if __name__ == "__main__":
    main()


        






    